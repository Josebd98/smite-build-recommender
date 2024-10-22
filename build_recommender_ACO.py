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
    "Hunter's Cowl", "Leader's Cowl", "Archmage's Gem", "Gem of Focus", "Blood-soaked Shroud", "Sacrificial Shroud"
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
    item_counts = defaultdict(int)
    for build in df_god['build']:
        items = build.split(', ')
        for item in items:
            if item in processed_items:
                item_counts[item] += 1
    # Sort items by frequency (most common first)
    sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
    return sorted_items

# Build pheromones influenced by frequency of common items
def initialize_pheromones(common_items, df_god, processed_items):
    pheromones = defaultdict(lambda: 1.0)  # Valor base de feromonas
    item_counts = defaultdict(int)
    
    # Contar la frecuencia de cada ítem en las builds del dios
    for build in df_god['build']:
        items = build.split(', ')
        for item in items:
            if item in processed_items:
                item_counts[item] += 1
    
    # Calibrar feromonas en función de la frecuencia
    total_builds = len(df_god)
    for item, count in item_counts.items():
        pheromones[item] = 1.0 + (count / total_builds)  # Mayor frecuencia => mayor valor de feromona
    
    return pheromones

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

# Run ACO to find the best build with pheromones initialization
def run_aco(model, scaler, columns, character_encoded, initial_items_god, non_initial_items, df_god, processed_items, 
                              num_ants=30, max_iterations=1000, evaporation=0.2, pheromone_intensity=1.5, 
                              convergence_threshold=10, min_improvement=1e-5):
    pheromones = initialize_pheromones(initial_items_god + non_initial_items, df_god, processed_items)
    best_build = None
    best_score = float('-inf')
    
    sorted_initial_items = sorted(initial_items_god, key=lambda x: pheromones[x], reverse=True)
    sorted_non_initial_items = sorted(non_initial_items, key=lambda x: pheromones[x], reverse=True)

    iterations_since_last_improvement = 0
    for iteration in range(max_iterations):
        iteration_best_build = None
        iteration_best_score = float('-inf')
        
        for _ in range(num_ants):
            build = build_influenced_by_pheromones(sorted_initial_items, sorted_non_initial_items, pheromones)
            score = evaluate_build(build, model, scaler, columns, character_encoded)[0]
            
            if score > iteration_best_score:
                iteration_best_score = score
                iteration_best_build = build
        
        # Actualizar la mejor build global
        if iteration_best_score > best_score + min_improvement:
            best_score = iteration_best_score
            best_build = iteration_best_build
            iterations_since_last_improvement = 0  # Resetear contador de estancamiento
        else:
            iterations_since_last_improvement += 1  # No hubo mejora significativa

        # Detener si ha habido estancamiento
        if iterations_since_last_improvement >= convergence_threshold:
            print(f"Converged after {iteration} iterations.")
            break

        # Actualizar feromonas según el mejor resultado de esta iteración
        update_pheromones(iteration_best_build, iteration_best_score, pheromones, evaporation, pheromone_intensity)
    
    return best_build, best_score


# Main function with modified pheromone initialization
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
    df_god = df[df['character_name'] == character_name]  # Filtrar por el dios específico
    pheromones = initialize_pheromones(common_items, df_god, processed_items)
    initial_items_god = sorted([obj for obj in common_items if obj in initial_objects], key=lambda x: pheromones[x], reverse=True)
    non_initial_items = sorted([obj for obj in common_items if obj not in initial_objects], key=lambda x: pheromones[x], reverse=True)
    
    # Run ACO optimization with pheromone initialization
    best_build, best_score = run_aco(model, scaler, X_columns, character_encoded, initial_items_god, non_initial_items, df_god, processed_items)
    
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
