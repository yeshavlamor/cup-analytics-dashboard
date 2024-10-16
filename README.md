# Cup Analytics Dashboard

<!-- ![Vekin Logo](/screenshots/vekin.png) -->

<div style="text-align:center">
  <img src="/screenshots/padded_vekin_2.png" alt="Alt text" style="width:100%;">
</div>

A comprehensive analytics dashboard developed in Streamlit for analyzing cup usage and error data.

The Cup Analytics Dashboard provides a user-friendly interface to visualize and analyze data related to cup usage, user statistics, and error types. It offers real-time filtering, interactive charts, and machine learning model training and prediction capabilities for users to interact with.

<p align="center">
    <img src="/screenshots/main.png" alt="Dashboard Preview">
</p>

## Features

- **Branch-based Filtering:** Select specific branches to analyze their data.
- **Key Performance Indicators (KPIs):** Displays important metrics for each branch: total cups, unique users, average cup count, success rate, and error types.
- **Data Visualization:**
  - Cup Count Distribution histogram
  - Top N Error Types analysis
- **Further Analysis:**
  - Cup Status comparison
  - Machine Learning Model Training with customizable hyperparameters
  - Success Probability Prediction based on input variables

## Live Demo
Explore the live demo here: [**Cup Analytics Dashboard**](https://cup-analytics-dashboard.streamlit.app/)


## Usage

1. **Branch Selection:** Use the sidebar to select a specific branch for analysis.
2. **KPI Overview:** View key metrics at the top of the dashboard.
3. **Data Visualization:** Explore the Cup Count Distribution and Top N Error Types charts.
4. **Further Analysis:**
   - Compare cup counts by status
   - Train a machine learning model with custom hyperparameters
   - Predict success probability based on input variables

## Project Structure

<!-- - `main.py`: Main Streamlit application file
- `database.py`: Module for database operations (not provided in the code snippet)
- `style.css`: Custom CSS styles for the dashboard
- `assets/`: Directory containing images and icons -->

cup-analytics-dashboard/
│
├── src/
│   ├── main.py            # Main Streamlit application 
│   └── database.py        # Module for database operations
│
├── .streamlit/
│   └── secrets.toml       # Configuration file 
│
├── assets/            # Directory containing images and icons
├── screenshots/           # Directory for dashboard screenshots 
├── style.css              # Custom CSS styles for the dashboard
├── requirements.txt       # List of Python dependencies
├── README.md              # Project documentation (this file)
└── .gitignore             # Specifies intentionally untracked files to ignore

## Setup and Configuration
To run the dashboard locally, follow these steps: 

1. Clone the Repository:
   ```
   git clone https://github.com/yeshavlamor/cup-analytics-dashboard.git
   cd cup-analytics-dashboard
   ```

2. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure Secrets:
Create a secrets.toml file in the .streamlit directory and add your database credentials. The structure should be similar to:
   ```
   [postgres]
   host = "your_host"
   port = "your_port"
   database = "your_database"
   user = "your_username"
   password = "your_password"
   ```

4. Run the Streamlit App:
   ```
   streamlit run main.py
   ```