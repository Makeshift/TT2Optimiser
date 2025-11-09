import streamlit as st
import pandas as pd
from components import render_results, render_ingredient_input, render_importance_input
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from config import get_ingredient_images, get_default_importance_scores
from graph_visualisation import render_graph_visualization
from inventory_tracking import track_inventory_from_formatted_combos, create_transition_df_from_inventory
from utils import highlight_changes

ingredient_images = get_ingredient_images()

# Load the CSV file
file_path = 'TT2 Alchemy Event.csv'
df = pd.read_csv(file_path, index_col=0)

# Combined function to extract loot (including currency and handling specific keywords)
def extract_loot(value, importance_keys):
    if isinstance(value, str):
        parts = value.split()
        try:
            amount = int(parts[0])
            item_type = ' '.join(parts[1:])
            for key in importance_keys:
                if key in item_type:
                    return (key, amount)
            return (item_type, amount)
        except (ValueError, IndexError):
            pass
    return ('Unknown', 0)

# Default importance scores
default_importance_scores = {
    "Currency": 0,
    "Shards": 0,
    "Perks": 0,
    "Raid Cards": 0,
    "Common Equipment": 0,
    "Rare Equipment": 0,
    "Event Equipment": 0,
    "Dust": 0,
    "Skill Points": 0,
    "Pet Eggs": 0,
    "Clan Eggs": 0,
    "Wildcards": 0,
    "Clan Scrolls": 0,
    "Hero Weapons": 0
}

# Apply the function to the dataframe
loot_df = df.applymap(lambda x: extract_loot(x, default_importance_scores.keys()))

# Extracting relevant data for optimization
items = list(df.index)
combinations = [(i, j) for i in items for j in items if i <= j]

# Map each ingredient to the combos that yield it so we can enforce crafting goals
ingredient_output_map = {}
for ingredient in items:
    combos_producing = []
    for combo in combinations:
        product = df.loc[combo]
        if isinstance(product, str) and product == ingredient:
            combos_producing.append(combo)
    ingredient_output_map[ingredient] = combos_producing

# Streamlit inputs
st.set_page_config(layout="wide")
st.title("TT2 Alchemy Event Optimizer")

# Editable dataframe for the CSV data
with st.expander("Edit CSV Data", expanded=False):
    edited_df = st.data_editor(df)

# Create input columns for the number of ingredients and the importance
st.header("Input the number of ingredients and importance scores:")

with st.expander("Alchemy Ingredient List", expanded=False):
    st.write(", ".join(items))

ensure_all_ingredients = st.checkbox("Ensure each ingredient is brewed at least once", value=False)

if ensure_all_ingredients:
    missing_recipes = [ingredient for ingredient, combos_for_item in ingredient_output_map.items() if not combos_for_item]
    if missing_recipes:
        st.warning(f"No recipes found for: {', '.join(missing_recipes)}")

col1, col2 = st.columns(2)

ingredient_counts = {}
importance_scores = {}

with col1:
    st.subheader("Number of Ingredients")
    ingredient_data = pd.DataFrame({
        "Ingredient": items,
        "Count": [0] * len(items)
    })
    edited_ingredient_data = st.data_editor(ingredient_data, num_rows="fixed", use_container_width=True, hide_index=True)
    for index, row in edited_ingredient_data.iterrows():
        ingredient_counts[row["Ingredient"]] = int(row["Count"])

with col2:
    st.subheader("Importance Scores")
    importance_data = pd.DataFrame({
        "Loot Type": list(default_importance_scores.keys()),
        "Importance": list(default_importance_scores.values())
    })
    # make "Importance" a float
    importance_data["Importance"] = importance_data["Importance"].astype(float)

    edited_importance_data = st.data_editor(importance_data, num_rows="fixed", use_container_width=True, hide_index=True)
    for index, row in edited_importance_data.iterrows():
        importance_scores[row["Loot Type"]] = float(row["Importance"])

# Button to trigger the optimization
# if st.button("Optimize"):

# Create a new LP problem
prob = LpProblem("Maximize Loot Score", LpMaximize)

# Define variables
combo_vars = LpVariable.dicts("Combo", combinations, lowBound=0, cat='Integer')

# Objective function: sum of (importance score * loot amount * variable) for each combination
prob += lpSum([
    importance_scores.get(extract_loot(df.loc[combo], importance_scores.keys())[0], 0) * extract_loot(df.loc[combo], importance_scores.keys())[1] * combo_vars[combo]
    for combo in combinations
])

# Constraints for each item
for item in items:
    # Used items constraints
    used = lpSum([combo_vars[combo] for combo in combinations if combo[0] == item]) + \
           lpSum([combo_vars[combo] for combo in combinations if combo[1] == item])

    # Created items constraints
    created = lpSum([combo_vars[combo] for combo in combinations if df.loc[combo] == item])

    prob += used <= ingredient_counts[item] + created

if ensure_all_ingredients:
    for ingredient, combos_producing in ingredient_output_map.items():
        if combos_producing:
            prob += lpSum([combo_vars[combo] for combo in combos_producing]) >= 1

# Solve the problem
prob.solve()

# Extract results
combos_used = [(combo, value(var), df.loc[combo]) for combo, var in combo_vars.items() if value(var) > 0]

# Calculate total loot and score
total_loot = {}
formatted_combos = []
total_score = 0

# and then order the ouput by the order of the items
combos_used = sorted(combos_used, key=lambda x: items.index(x[0][0]))

# reorder the combos_used to have the ingredients first
combos_used = sorted(combos_used, key=lambda x: any(key in x[2] for key in importance_scores.keys()))

for combo, count, product in combos_used:
    product_name, product_amount = extract_loot(product, importance_scores.keys())
    if product_name in total_loot:
        total_loot[product_name] += product_amount * count
    else:
        total_loot[product_name] = product_amount * count
    total_score += importance_scores.get(product_name, 0) * product_amount * count
    formatted_combos.append({
        'input1': combo[0],
        'input2': combo[1],
        'count': count,
        'result': product,
        'is_ingredient': not any(key in product for key in importance_scores.keys() if isinstance(product, str))
    })

# st.header("Results")
# st.write(f"Maximum score: {total_score}")

# st.write(combos_used)
from render_combo import render_results
render_results(total_score, combos_used, total_loot, ingredient_images)

if ensure_all_ingredients:
    coverage_rows = []
    for ingredient, combos_producing in ingredient_output_map.items():
        produced = sum(value(combo_vars[combo]) for combo in combos_producing) if combos_producing else 0
        coverage_rows.append({
            "Ingredient": ingredient,
            "Goal": 1 if combos_producing else 0,
            "Produced": int(produced) if produced is not None else 0,
            "Goal Met": bool(produced >= 1) if combos_producing else False
        })
    coverage_df = pd.DataFrame(coverage_rows)
    st.subheader("Ingredient Goal Coverage")
    st.dataframe(coverage_df, use_container_width=True)

st.subheader("Check brews:")
st.write("Changes in the quantities are highlighted in yellow")
# Get your dataframe
df = track_inventory_from_formatted_combos(ingredient_counts, formatted_combos)

# Apply highlighting and display
styled_df = highlight_changes(df)

# Display the styled dataframe
st.write(styled_df, use_container_width=True)

# with st.expander("Check items", expanded=False):
#     for combo, count, product in combos_used:
#         product_name, product_amount = extract_loot(product, importance_scores.keys())
#         st.write(f"{count} x {combo} = {product}")

with st.expander("Visualise results - (Experimental)", expanded=False):
    # graph visualisation
    inventory_df = render_graph_visualization(combos_used, ingredient_counts, total_loot, formatted_combos)
