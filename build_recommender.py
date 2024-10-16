import argparse
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools
from keras.models import load_model
import random
from collections import defaultdict
import os

initial_objects = [
    "Death's Embrace", "Death's Temper", 'Diamond Arrow', 'Ornate Arrow',
    'Sundering Axe', 'Axe of Animosity', 'Manikin Mace', 'Manikin Hidden Blade',
    "Bumba's Spear", "Bumba's Hammer", 'Pendulum of Ages', 'The Alternate Timeline',
    "Sentinel's Boon", "Sentinel's Embrace", 'Compassion', 'Heroism',
    'Tainted Amulet', 'Tainted Breastplate', 'Seer of the Jungle', 'Protector of the Jungle',
    'Sigil of The Old Guard', 'Infused Sigil', "Lono's Mask","Rangda's Mask",
    'Bluestone Brooch', 'Corrupted Bluestone', 'Spartan Flag', 'War Banner',
    "Hunter's Cowl", "Leader's Cowl","Archmage's Gem","Gem of Focus","Blood-soaked Shroud","Sacrificial Shroud"
]

# Connect to the SQLite database
def connect_db():
    db_path = os.path.join('database', 'smite_players.db')
    return sqlite3.connect(db_path)

# Function to get character data from the database
def get_character_data(cursor, name):
    cursor.execute('SELECT class_distance, type_dmg, type_dmgform, tags FROM gods WHERE name = ?', (name,))
    data = cursor.fetchone()
    return data if data else ("N/A", "N/A", "N/A", "N/A")

# Load DataFrame from SQLite
def load_dataframe():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM combined_data5")
    column_names = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=column_names)
    print(f"Number of records obtained: {len(rows)}")
    conn.close()
    return df

# Initialize the scaler and load the pre-trained model
def initialize_model_and_scaler(columns):
    # Ensure all columns are strings and replace 'N/A'
    columns = [str(col) if col != 'N/A' else 'missing' for col in columns]
    scaler = StandardScaler().fit(pd.DataFrame(0, index=[0], columns=columns))
    model_path = os.path.join( 'models', 'nn_build_score_model.h5')
    model = load_model(model_path, compile=False)
    return scaler, model

# Evaluate a build with the prediction model
def evaluate_build(build, model, scaler, columns, character_encoded):
    build_df = character_encoded.copy()
    for item in build:
        column_name = f'{item}'
        if column_name in build_df.columns:
            build_df[column_name] = 1
    build_df = build_df.reindex(columns=columns, fill_value=0)
    build_scaled = scaler.transform(build_df)
    score = model.predict(build_scaled)
    return score[0][0],

# Function to get common items for a god
def get_common_items(god, df, processed_items):
    df_god = df[df['character_name'] == god]
    common_items = set()
    for build in df_god['build']:
        items = build.split(', ')
        filtered_items = [item for item in items if item in processed_items]
        common_items.update(filtered_items)
    return list(common_items)

# Build a pheromone-influenced build
def build_influenced_by_pheromones(initial_items_god, non_initial_items, pheromones):
    initial = random.choices(
        population=initial_items_god,
        weights=[pheromones[obj] for obj in initial_items_god],
        k=1
    )[0]
    remaining_build = []
    for _ in range(5):
        candidates = [obj for obj in non_initial_items if obj not in remaining_build]
        weights = [pheromones[obj] for obj in candidates]
        next_item = random.choices(candidates, weights=weights, k=1)[0]
        remaining_build.append(next_item)
    return [initial] + remaining_build

# Update pheromones based on score
def update_pheromones(build, score, pheromones, evaporation, pheromone_intensity):
    for obj in build:
        pheromones[obj] += score * pheromone_intensity
    for obj in pheromones:
        pheromones[obj] *= (1 - evaporation)

# Run ACO to find the best build
def run_aco(model, scaler, columns, character_encoded, initial_items_god, non_initial_items, num_ants=60, num_iterations=30, evaporation=0.1, pheromone_intensity=1.0):
    pheromones = defaultdict(lambda: 1.0)
    best_build = None
    best_score = float('-inf')
    
    for iteration in range(num_iterations):
        for _ in range(num_ants):
            build = build_influenced_by_pheromones(initial_items_god, non_initial_items, pheromones)
            
            # Check the structure of the build
            
            score = evaluate_build(build, model, scaler, columns, character_encoded)[0]
            
            # Check the score
            
            if score > best_score:
                best_score = score
                best_build = build
            
            # Check pheromones before and after the update
            update_pheromones(build, score, pheromones, evaporation, pheromone_intensity)
        

    return best_build, best_score

# Main function
def main(character_name, enemies):
    conn = connect_db()
    cursor = conn.cursor()

    # Get data for the character and enemies
    character_data = {
        'character_name': character_name,
        'character_class_distance': get_character_data(cursor, character_name)[0],
        'character_type_dmg': get_character_data(cursor, character_name)[1],
        'character_type_dmgform': get_character_data(cursor, character_name)[2],
    }
    for i, enemy in enumerate(enemies):
        character_data[f'enemy_{i+1}_class_distance'] = get_character_data(cursor, enemy)[0]
        character_data[f'enemy_{i+1}_type_dmg'] = get_character_data(cursor, enemy)[1]
        character_data[f'enemy_{i+1}_type_dmgform'] = get_character_data(cursor, enemy)[2]
    
    # Load DataFrame, model, and columns
    df = load_dataframe()
    X_columns = pd.read_csv("X_columns.csv", header=None).squeeze().tolist()
    X_columns = [str(col) for col in X_columns]
    scaler, model = initialize_model_and_scaler(X_columns)
    
    # Encode character and enemy data
    character_encoded = pd.DataFrame(0, index=[0], columns=X_columns)
    for key, value in character_data.items():
        column_name = f"{key}_{value}"
        if column_name in character_encoded.columns:
            character_encoded[column_name] = 1

    # Load items and run ACO
    with open("items_processed.txt", "r", encoding="utf-8") as file:
        processed_items = [line.strip() for line in file]
    common_items = get_common_items(character_name, df, processed_items)
    
    # Define initial and non-initial items
    initial_items_god = [obj for obj in common_items if obj in initial_objects]
    non_initial_items = [obj for obj in common_items if obj not in initial_objects]
    
    # Run ACO optimization
    best_build, best_score = run_aco(model, scaler, X_columns, character_encoded, initial_items_god, non_initial_items)
    
    conn.close()
    return best_build, best_score

# Run the script with command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize the build for a Smite character against three enemies.")
    parser.add_argument("character_name", type=str, help="Name of the main character")
    parser.add_argument("enemies", type=str, nargs=3, help="Names of the three enemies")
    
    args = parser.parse_args()
    
    best_build, best_score = main(args.character_name, args.enemies)
    print("The best build found is:", best_build)
    print("Score:", best_score)
