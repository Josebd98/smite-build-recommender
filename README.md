# Smite Build Recommender

This project provides build recommendations for the game **Smite**. The information used to build this recommender system was gathered through web scraping from the Smite Guru website. Using this data, a machine learning model was developed to predict damage. Additionally, an ant colony optimization method or a bayesian optimization was applied to explore different possible builds based on the chosen characters. Bayesian Optimization is now working better than ACO. Please note, this project is still in testing, and it’s not yet fully functional.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Credits](#credits)

## Overview

The Smite Build Recommender aims to assist players in optimizing their builds based on historical game data and the characters selected for each match. This tool provides:
- **API with FastAPI** to serve build recommendations.
- **Machine Learning Models** to predict build scores, using damage as a key performance indicator.
- **Ant Colony Optimization** method to test various builds based on selected characters and enemies.
- - **Bayesian Optimization** other method to test various builds based on selected characters and enemies.
- **Web Scraping** and **Automation**: Utilizes Selenium and PyAutoGUI to gather data directly from the web and automate tasks.

## Requirements

The project requires several dependencies, all listed in `requirements.txt`. You can install them all at once by running:

```bash
pip install -r requirements.txt
```

**Dependencies List**:
- sqlite3
- os
- selenium
- time
- re
- random
- beautifulsoup
- requests
- pyautogui
- keyboard
- pandas
- numpy
- scikit-learn
- tensorflow
- fastapi
- optuna
- keras
- catboost
- deap
- argparse
- subprocess
- traceback

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/smite-build-recommender.git
   ```
   
2. **Navigate to the project directory**:
   ```bash
   cd smite-build-recommender
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up**:
   - If you are using **Selenium**, ensure that **ChromeDriver** is installed and that the `chromedriver.exe` file is located in the correct path specified in the scripts.
   - Place the database files in the `database` folder and make sure the notebooks are in the `notebooks` folder as described in the project structure below.

## Usage

### Running the FastAPI Server

To start the FastAPI server, execute the following command from the root of the project:
```bash
uvicorn app:app --reload
```
This will start the development server at `http://127.0.0.1:8000/static/index.html`, where you can access the API.

### Running Jupyter Notebooks

For data analysis and model training:
1. Navigate to the `notebooks` folder:
   ```bash
   cd notebooks
   ```
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

### Running Build Optimization Scripts

The repository includes multiple Python scripts for training models and optimizing builds. You can execute these scripts directly from the command line. Make sure file paths are configured correctly, and all dependencies are installed.

### Testing the Tool

To test the build recommender, you need to install the dependencies and start the application by running `app.py`. Once the FastAPI server is running, you can use an HTTP client such as **Postman** to test the endpoints and retrieve build recommendations.

## Project Structure

```plaintext
├── database                
│   └── smite_players.db
│   └── smite_gods.db
├── notebooks               
│   └── ML_models.ipynb
│   └── Database.ipynb
├── build_recommender.py               
├── app.py    
├── static      
│   └── index.html          
├── requirements.txt        
└── README.md     
```

## Credits

This project was developed by (https://github.com/Josebd98). Special thanks to **Hi-Rez Studios** for making **Smite** data accessible and to **Smite Guru** for providing valuable data for this tool.


