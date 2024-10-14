from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import subprocess
import os
import traceback
import re


os.environ["PYTHONIOENCODING"] = "utf-8"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/recommend")
async def recommend(character_name: str = Form(...), enemy_1: str = Form(...), enemy_2: str = Form(...), enemy_3: str = Form(...)):
    enemies = [enemy_1, enemy_2, enemy_3]
    
    python_path = "C:\\Users\\joseb\\anaconda3\\python.exe"
    script_path = os.path.join(os.getcwd(), 'build_recommender.py')
    
    try:
        # Run the script to get the build
        result = subprocess.run(
            [python_path, script_path, character_name, enemy_1, enemy_2, enemy_3],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )

        if result.stdout is None:
            raise ValueError("No output was obtained from stdout.")
        
        output = result.stdout.strip()
        print("Script output:")
        print(output)

        build_data = {
            "character_name": character_name,
            "enemies": enemies,
            "build_images": [],
        }

        # Extract the build from the script output
        try:
            lines = output.splitlines()
            build_line = next(line for line in lines if "The best build found is:" in line)
            build_items = re.findall(r'\'(.*?)\'|\"(.*?)\"', build_line.split(":")[1].strip().strip("[]"))
            build_items = [item[0] if item[0] else item[1] for item in build_items]

            
            # Connect to the database to get the URLs of the images, preserving the order
            db_path = os.path.join('database', 'smite_gods.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Search for each item in the correct order and add the URL to `build_images`
            for item in build_items:
                cursor.execute("SELECT name, link FROM items_images WHERE name = ?", (item,))
                row = cursor.fetchone()
                if row:
                    build_data["build_images"].append({"name": row[0], "url": row[1]})
            
            conn.close()
            
        except (ValueError, StopIteration):
            error_msg = "Could not interpret the script output."
            print(error_msg)
            build_data["error"] = error_msg

        return JSONResponse(content=build_data)

    except subprocess.CalledProcessError as e:
        print("Error running the script:")
        print(e.stderr)
        traceback.print_exc()
        return JSONResponse(content={"error": "Error running the script.", "details": e.stderr}, status_code=500)
    
    except Exception as e:
        print("Unexpected error:")
        traceback.print_exc()
        return JSONResponse(content={"error": "Unexpected error.", "details": str(e)}, status_code=500)

@app.get("/get_gods")
async def get_gods():
    db_path = os.path.join('database', 'smite_gods.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, link FROM gods_images")
    gods = [{"name": row[0], "link": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return JSONResponse(content=gods)

# New route to get item data
@app.get("/get_items")
async def get_items():
    db_path = os.path.join('database', 'smite_gods.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name, link FROM items_images")
    items = [{"name": row[0], "link": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return JSONResponse(content=items)
