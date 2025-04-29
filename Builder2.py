import pandas as pd
import re
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus

# --- Parsing helpers ---
def parse_item(item_str):
    """Parses an item string like 'Copper Frame (20x) 20/min'"""
    pattern = r"(.+?)\s*\(([\d\.]+)x\)\s*([\d\.]+)/min"
    match = re.match(pattern, item_str.strip())
    if match:
        name = match.group(1).strip()
        multiplier = float(match.group(2))
        rate = float(match.group(3))
        return name, multiplier, rate
    return None

def parse_outputs(output_str):
    """Parses multi-line outputs"""
    outputs = {}
    for line in output_str.splitlines():
        parsed = parse_item(line)
        if parsed:
            name, multiplier, rate = parsed
            outputs[name] = rate
    return outputs

def parse_inputs(input_str):
    """Parses multi-line ingredient inputs"""
    inputs = {}
    for line in input_str.splitlines():
        parsed = parse_item(line)
        if parsed:
            name, multiplier, rate = parsed
            inputs[name] = rate
    return inputs

# --- Load and parse CSV ---
df = pd.read_csv("Techtonica Recipies - Sheet1.csv")
df = df[df['Ouput'].notna()]

recipes = []
for index, row in df.iterrows():
    outputs = parse_outputs(row['Ouput'])
    machine = row['Machine']
    inputs = parse_inputs(row['Ingredients'])
    technology = row['technology']

    if outputs and inputs:
        recipes.append({
            'recipe_id': index,
            'machine': machine,
            'technology': technology,
            'outputs': outputs,
            'ingredients': inputs
        })

# --- Identify raw materials ---
produced_products = {p for r in recipes for p in r['outputs'].keys()}
consumed_products = {p for r in recipes for p in r['ingredients'].keys()}
raw_materials = consumed_products - produced_products

# --- LP Optimization Setup ---
target_product = "Research Core 590nm (Yellow)"
target_min_production = 1  # Desired 1 unit per minute

model = LpProblem("Recipe_Optimizer", LpMaximize)
x = {r['recipe_id']: LpVariable(f"x_{r['recipe_id']}", lowBound=0) for r in recipes}

model += lpSum(x[r['recipe_id']] * r['outputs'].get(target_product, 0) for r in recipes), "Maximize_Target"

for product in produced_products:
    flow_terms = []
    for r in recipes:
        if product in r['outputs']:
            flow_terms.append(x[r['recipe_id']] * r['outputs'][product])
        if product in r['ingredients']:
            flow_terms.append(-x[r['recipe_id']] * r['ingredients'][product])
    model += lpSum(flow_terms) == (target_min_production if product == target_product else 0), f"Flow_{product}"

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
def trace_production(product, required_amount, indent=0):
    prefix = " " * indent
    if product in raw_materials:
        print(f"{prefix}Raw Material: {product} → Needed: {required_amount:.2f}/min")
    else:
        for r in recipes:
            if product in r['outputs']:
                x_val = x[r['recipe_id']].varValue
                if x_val and x_val > 0:
                    output_rate = r['outputs'][product] * x_val
                    scale_factor = required_amount / output_rate
                    print(f"{prefix}→ Use {r['machine']} ({r['technology']}) at {x_val * scale_factor:.2f} scale → Produces {product} at {output_rate * scale_factor:.2f}/min")
                    for ing, ing_rate in r['ingredients'].items():
                        trace_production(ing, ing_rate * x_val * scale_factor, indent=indent + 4)

print("\nProduction Chain Breakdown:")
trace_production(target_product, target_min_production)
