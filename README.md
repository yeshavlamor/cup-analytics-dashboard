# Cup Analytics Dashboard

<!-- ![Vekin Logo](/screenshots/vekin.png) -->

<div style="text-align:center">
  <img src="/screenshots/padded_vekin_2.png" alt="Alt text" style="width:70%;">
</div>

The Cup Analytics Dashboard was developed to aid Vekin in analysing cup usage and error data. I developed this dashboard using the open-source Python framework,  [Streamlit](https://docs.streamlit.io/).

 provides a user-friendly interface to visualize and analyze data related to cup usage, user statistics, and error types. It offers real-time filtering, interactive charts, and machine learning model training and prediction capabilities for users to interact with.

<p align="center">
    <img src="/screenshots/main.png" alt="Dashboard Preview">
</p>

<p align="center">
    <img src="/screenshots/model.png" alt="Dashboard Preview">
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

## Setup and Configuration

1. Clone the Repository:
   ```
   git clone https://github.com/yeshavlamor/cup-analytics-dashboard.git
   cd cup-analytics-dashboard
   ```

2. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit App:
   ```
   streamlit run main.py
   ```

## Usage

1. **Branch Selection:** Use the sidebar to select a specific branch for analysis.
2. **KPI Overview:** View key metrics at the top of the dashboard.
3. **Data Visualization:** Explore the Cup Count Distribution and Top N Error Types charts.
4. **Further Analysis:**
   - Compare cup counts by status
   - Train a machine learning model with custom hyperparameters
   - Predict success probability based on input variables

## Project Structure

- `main.py`: Main Streamlit application file
- `database.py`: Module for database operations (not provided in the code snippet)
- `style.css`: Custom CSS styles for the dashboard
- `assets/`: Directory containing images and icons

## Dependencies

- streamlit
- pandas
- plotly
- matplotlib
- seaborn
- scikit-learn