import pandas as pd
import re
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus

# --------------------------------------------------------------------
# Helpers for parsing recipe strings that look like:
#    "Copper Frame (20x) 20/min"
# --------------------------------------------------------------------
def parse_item(item_str):
    """
    Parse a string of the form "Item (Nx) R/min" and return:
       (name, multiplier, rate)
    """
    pattern = r"(.+?)\s*\(([\d\.]+)x\)\s*([\d\.]+)/min"
    match = re.match(pattern, item_str.strip())
    if match:
        name = match.group(1).strip()
        multiplier = float(match.group(2))
        rate = float(match.group(3))
        return name, multiplier, rate
    return None

def parse_outputs(output_str):
    """
    Parse an output cell (which may have several lines) into a dictionary.
    For example, an output cell like:
      "Atlantum Ore (4x) 27.0/min 
       Limestone (6x) 40.5/min"
    will yield: {"Atlantum Ore": 27.0, "Limestone": 40.5}
    """
    outputs = {}
    for line in output_str.splitlines():
        if not line.strip():
            continue
        parsed = parse_item(line)
        if parsed:
            name, multiplier, rate = parsed
            outputs[name] = rate
    return outputs

# --------------------------------------------------------------------
# Load and parse the CSV file. (Adjust the filename as needed.)
# Remove rows that are not production recipes (for example, rows with just
# a single uppercase letter).
# --------------------------------------------------------------------
df = pd.read_csv("Techtonica Recipies - Sheet1.csv")
df = df[df['Ouput'].notna()]
df = df[~df['Ouput'].str.match(r"^[A-Z]\s*$")]

recipes = []
for index, row in df.iterrows():
    output_str = row['Ouput']
    machine = row['Machine']
    ingredients_str = row['Ingredients']
    technology = row['technology']
    
    outputs = parse_outputs(output_str)
    if not outputs:
        continue
    ingredients = {}
    if pd.notna(ingredients_str):
        for line in ingredients_str.splitlines():
            if not line.strip():
                continue
            parsed_ing = parse_item(line)
            if parsed_ing:
                ing_name, ing_multiplier, ing_rate = parsed_ing
                ingredients[ing_name] = ing_rate  
    recipes.append({
        'recipe_id': index,
        'machine': machine,
        'technology': technology,
        'outputs': outputs,         # dictionary: product -> production rate (per machine unit per minute)
        'ingredients': ingredients  # dictionary: ingredient -> consumption rate
    })

# --------------------------------------------------------------------
# Build a product network.
# --------------------------------------------------------------------
produced_products = {p for r in recipes for p in r['outputs'].keys()}
consumed_products = {p for r in recipes for p in r['ingredients'].keys()}
all_products = produced_products.union(consumed_products)

# Identify raw materials as those used only as ingredients (never produced onsight)
raw_materials = consumed_products - produced_products
print("Raw materials found:", raw_materials)

# --------------------------------------------------------------------
# Define your external (raw) resource limits if needed.
# (For now, we leave this dictionary empty or with sample values.)
# --------------------------------------------------------------------
available_raw = {
    # For example: "Copper Ore": 100, "Iron Ore": 100, "Limestone": 200, etc.
}

# --------------------------------------------------------------------
# Set up and solve the LP model.
#
# In this model, each recipe is run at some scaling factor x (machine units).
# Flow balance: for every product produced and used as an ingredient (except raw materials),
# production must equal consumption. For your target product, we require production to
# be at least the desired threshold.
# --------------------------------------------------------------------
target_product = "Research Core 590nm (Yellow)"
target_min_production = 1000  # desired items per minute

model = LpProblem("Recipe_Optimizer", LpMaximize)

# Decision variables for each recipe:
x = {r['recipe_id']: LpVariable(f"x_{r['recipe_id']}", lowBound=0) for r in recipes}

# Objective: maximize production of the target product
model += lpSum(x[r['recipe_id']] * r['outputs'].get(target_product, 0)
               for r in recipes), "Maximize_Target_Product"

# Flow balance constraints for each product that is produced (non-raw).
for product in all_products:
    if product in raw_materials:
        continue
    flow = []
    for r in recipes:
        if product in r['outputs']:
            flow.append(x[r['recipe_id']] * r['outputs'][product])
        if product in r['ingredients']:
            flow.append(- x[r['recipe_id']] * r['ingredients'][product])
    if product == target_product:
        model += lpSum(flow) >= target_min_production, f"flow_balance_{product}"
    else:
        model += lpSum(flow) == 0, f"flow_balance_{product}"

# Raw material supply constraints (if any values are set)
for product in raw_materials:
    if product in available_raw:
        consumption = []
        for r in recipes:
            if product in r['ingredients']:
                consumption.append(x[r['recipe_id']] * r['ingredients'][product])
        model += lpSum(consumption) <= available_raw[product], f"raw_supply_{product}"

model.solve()
print("\nOptimization status:", LpStatus[model.status])

# Print a summary plan for recipes with nonzero usage:
print("\nMachines to run (nonzero levels):")
for r in recipes:
    x_val = x[r['recipe_id']].varValue
    if x_val and x_val > 0:
        outs = ", ".join([f"{prod}: {x_val * rate:.2f}/min" for prod, rate in r['outputs'].items()])
        print(f" * Recipe {r['recipe_id']} ({r['machine']}, {r['technology']}): run at {x_val:.2f} units → produces {outs}")

# --------------------------------------------------------------------
# Now, build a production chain report.
#
# For a forward (raw-material–to–target) hierarchical report we assume that, at
# each stage, if a product is produced on-site, it is (typically) produced by a 
# single recipe. (If there are multiples, here we simply choose the first.)
# --------------------------------------------------------------------
# First, annotate each recipe with its solved operating level, if any.
for r in recipes:
    r['x_val'] = x[r['recipe_id']].varValue if x[r['recipe_id']].varValue else 0

# Build a mapping: for each product produced, choose one recipe that produces it.
produced_by = {}  # product -> recipe
for r in recipes:
    if r['x_val'] > 0:
        for prod, rate in r['outputs'].items():
            # (If multiple recipes make the same product, you could choose the one with higher throughput.)
            if prod not in produced_by:
                produced_by[prod] = r

# For each recipe that is used, we know its contributions to each product:
# flow (per minute) = machine level * rate

def print_chain(product, required_amount, indent=0):
    prefix = " " * indent
    # If the product is raw, show it as a starting input.
    if product in raw_materials:
        print(prefix + f"Raw material: {product} required: {required_amount:.2f}/min")
    elif product in produced_by:
        rec = produced_by[product]
        production_rate = rec['outputs'][product] * rec['x_val']
        # scale = factor needed to supply the required amount via this recipe
        if production_rate == 0:
            scale = 0
        else:
            scale = required_amount / production_rate
        # Report which machine (recipe) is run and at what scaled level
        print(prefix + f"→ Use machine '{rec['machine']}' ({rec['technology']}) at scaled level {rec['x_val']*scale:.2f} to produce {product} at {production_rate*scale:.2f}/min")
        # For each ingredient of this recipe, compute the required amount and print further.
        for ing, cons_rate in rec['ingredients'].items():
            ing_required = cons_rate * rec['x_val'] * scale
            print_chain(ing, ing_required, indent=indent+4)
    else:
        # In case there is no production path on-site
        print(prefix + f"(No on-site recipe for {product}; assume external supply) required: {required_amount:.2f}/min")

# Determine the total production of the target product from the LP flow.
target_production = sum(rec['outputs'][target_product] * rec['x_val'] for rec in recipes if target_product in rec['outputs'])
print(f"\nTotal production of {target_product}: {target_production:.2f}/min")

print("\nProduction Chain (from raw materials upward):")
print_chain(target_product, target_production)
