import pandas as pd
import re
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus

# --- Helper Functions ---
def parse_item(item_str):
    """
    Parse an item string like 'Copper Frame (20x) 20/min'.
    Returns a tuple (name, multiplier, rate).
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
    Parses multi-line outputs into a dictionary of {product: rate}.
    Example: "Atlantum Ore (4x) 27.0/min\nLimestone (6x) 40.5/min"
    Returns: {"Atlantum Ore": 27.0, "Limestone": 40.5}
    """
    outputs = {}
    for line in output_str.splitlines():
        parsed = parse_item(line)
        if parsed:
            name, multiplier, rate = parsed
            outputs[name] = rate
    return outputs

def parse_inputs(input_str):
    """
    Parses multi-line ingredients into a dictionary of {ingredient: rate}.
    Example: "Copper Frame (20x) 20/min\nElectrical Components (20x) 20/min"
    Returns: {"Copper Frame": 20.0, "Electrical Components": 20.0}
    """
    inputs = {}
    # Only process if the string is valid (non-NaN)
    if pd.notna(input_str):
        for line in input_str.splitlines():
            parsed = parse_item(line)
            if parsed:
                name, multiplier, rate = parsed
                inputs[name] = rate
    return inputs

# --- Load and Parse CSV ---
df = pd.read_csv("Techtonica Recipies - Sheet1.csv")

recipes = []
for index, row in df.iterrows():
    # Skip rows with missing critical columns
    if pd.isna(row['Ouput']) or pd.isna(row['Ingredients']) or pd.isna(row['Machine']):
        continue  # Skip invalid rows
    
    outputs = parse_outputs(row['Ouput'])
    inputs = parse_inputs(row['Ingredients'])
    machine = row['Machine']
    technology = row['technology']

    # Skip rows with parsing failures
    if not outputs or not inputs:
        continue

    recipes.append({
        'recipe_id': index,
        'machine': machine,
        'technology': technology,
        'outputs': outputs,         # Dictionary: {product: rate}
        'ingredients': inputs       # Dictionary: {ingredient: rate}
    })

# --- Identify Raw Materials ---
produced_products = {p for r in recipes for p in r['outputs'].keys()}
consumed_products = {p for r in recipes for p in r['ingredients'].keys()}
raw_materials = consumed_products - produced_products

# --- LP Optimization ---
target_product = "Research Core 590nm (Yellow)"
target_min_production = 1  # Desired production rate per minute

model = LpProblem("Recipe_Optimizer", LpMaximize)

# Create decision variables for each recipe (machine units)
x = {r['recipe_id']: LpVariable(f"x_{r['recipe_id']}", lowBound=0) for r in recipes}

# Objective function: Maximize production of the target product
model += lpSum(x[r['recipe_id']] * r['outputs'].get(target_product, 0) for r in recipes), "Maximize_Target"

# Add flow balance constraints for each product
for product in produced_products:
    flow_terms = []
    for r in recipes:
        if product in r['outputs']:
            flow_terms.append(x[r['recipe_id']] * r['outputs'][product])
        if product in r['ingredients']:
            flow_terms.append(-x[r['recipe_id']] * r['ingredients'][product])
    model += lpSum(flow_terms) == (target_min_production if product == target_product else 0), f"Flow_Balance_{product}"

# Solve the linear programming problem
model.solve()

# --- Display Machine Plan ---
print(f"Optimization status: {LpStatus[model.status]}")
print("\nMachines to run:")
for r in recipes:
    x_val = x[r['recipe_id']].varValue
    if x_val and x_val > 0:
        outputs_str = ", ".join([f"{prod}: {x_val * rate:.2f}/min" for prod, rate in r['outputs'].items()])
        print(f" * {r['machine']} ({r['technology']}): {x_val:.2f} units → Produces {outputs_str}")

# --- Generate Hierarchical Production Chain ---
def trace_production(product, required_amount, indent=0, visited=None):
    """
    Recursive function to trace the production chain for a given product.
    Adds cycle detection and checks for missing recipes to prevent infinite recursion.
    """
    prefix = " " * indent
    if visited is None:
        visited = set()  # Initialize the visited set

    # Stop if we've already visited this product (to prevent cycles)
    if product in visited:
        print(f"{prefix}(Cycle detected for {product}, skipping further tracing)")
        return
    visited.add(product)

    # Handle raw materials
    if product in raw_materials:
        print(f"{prefix}Raw Material: {product} → Needed: {required_amount:.2f}/min")
        return

    # Look for a recipe that produces the product
    producing_recipes = [r for r in recipes if product in r['outputs']]
    if not producing_recipes:
        print(f"{prefix}(No recipe produces {product}, ensure it is externally supplied)")
        return

    for r in producing_recipes:
        x_val = x[r['recipe_id']].varValue
        if x_val and x_val > 0:
            output_rate = r['outputs'][product] * x_val
            scale_factor = required_amount / output_rate if output_rate > 0 else 0
            print(f"{prefix}→ Use {r['machine']} ({r['technology']}) at {x_val * scale_factor:.2f} scale → Produces {product} at {output_rate * scale_factor:.2f}/min")
            for ing, ing_rate in r['ingredients'].items():
                trace_production(ing, ing_rate * x_val * scale_factor, indent=indent + 4, visited=visited)

    # Remove the product from visited (if you need to trace other branches separately)
    visited.remove(product)

print("\nProduction Chain Breakdown:")
trace_production(target_product, target_min_production)

# --- Final Summary ---
print("\nFinal Production Summary:")
summary = {}
for r in recipes:
    x_val = x[r['recipe_id']].varValue
    if x_val and x_val > 0:
        machine_type = r['machine']
        outputs = r['outputs']
        inputs = r['ingredients']
        for product, rate in outputs.items():
            summary_key = f"{machine_type} producing {product}"
            if summary_key not in summary:
                summary[summary_key] = {"count": 0, "inputs": {}}
            summary[summary_key]["count"] += x_val
            for ing, ing_rate in inputs.items():
                if ing not in summary[summary_key]["inputs"]:
                    summary[summary_key]["inputs"][ing] = 0
                summary[summary_key]["inputs"][ing] += ing_rate * x_val

# Print the summary
for task, details in summary.items():
    machine_count = details["count"]
    input_requirements = ", ".join([f"{ing}: {rate:.2f}/min" for ing, rate in details["inputs"].items()])
    print(f"- {task}: {machine_count:.2f} units needing {input_requirements}")

# --- Calculate Resources Not Consumed ---
print("\nUnused Resources (Not Fully Consumed):")
unused_resources = {}
for product in produced_products:
    # Calculate total production of the product
    total_produced = sum(x[r['recipe_id']].varValue * r['outputs'].get(product, 0) for r in recipes if x[r['recipe_id']].varValue)
    # Calculate total consumption of the product
    total_consumed = sum(x[r['recipe_id']].varValue * r['ingredients'].get(product, 0) for r in recipes if x[r['recipe_id']].varValue)
    # Calculate leftover resource
    if total_produced > total_consumed:
        unused_resources[product] = total_produced - total_consumed

# Display unused resources
if unused_resources:
    for product, excess in unused_resources.items():
        print(f"- {product}: {excess:.2f}/min not consumed")
else:
    print("All produced resources are fully consumed.")
