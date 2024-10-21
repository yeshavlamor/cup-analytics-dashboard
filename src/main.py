import streamlit as st
st.set_page_config(page_title="Cup Analytics Dashboard", layout="wide") 

# import rest of libraries
import pandas as pd
import plotly.express as px # for visualisations 
import plotly.graph_objects as go # provides more fine-tuned control over indiv elements of graph 
import matplotlib.pyplot as plt
import seaborn as sns
import database 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# import custom css design 
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# get data
all_data = database.get_all_processed_data()
error_df = all_data['error_data']
cup_df = all_data['cup_data']

# main area 
st.title("Cup Analytics Dashboard")

# sidebar area  
with st.sidebar:
    st.sidebar.image("assets/vekin_logo.png", width=150)
    st.sidebar.title("Please Filter Here: ")

# filter is the branch_code
branch = st.sidebar.selectbox("Select Branch:", sorted(cup_df['branch_code'].unique()),index=7) # default branch is 1900

def filter_dataframe(df, branch):
    filtered_cup_df = df.copy()    
    if branch: 
        filtered_cup_df = filtered_cup_df[filtered_cup_df['branch_code'] == branch]
    return filtered_cup_df

# filtered dfs
filtered_cup_df = filter_dataframe(cup_df, branch)
filtered_error_df = filter_dataframe(error_df, branch)


# update KPI metrics with filtered data
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    kpi_col1, kpi_col2 = st.columns([1,2]) # splits each column into sub-columns, where first subcolumn takes up one unit and the next one takes up 3 units of space
    kpi_col1.image('assets/total_cups_icon.png', width=30) 
    kpi_col2.metric("Total Cups", filtered_cup_df['total_cup'].sum())

with col2:
    kpi_col1, kpi_col2 = st.columns([1,2]) # splits each column into sub-columns, where first subcolumn takes up one unit and the next one takes up 3 units of space
    kpi_col1.image('assets/unique_users_icon.png', width=30) 
    kpi_col2.metric("Unique Users", filtered_cup_df['user_count'].nunique())

with col3:
    kpi_col1, kpi_col2 = st.columns([1,2]) # splits each column into sub-columns, where first subcolumn takes up one unit and the next one takes up 3 units of space
    kpi_col1.image('assets/avg_cup_count_icon.png', width=30) 
    kpi_col2.metric("Avg Cup Count", filtered_cup_df['cup_count'].mean().round(2))

with col4:
    kpi_col1, kpi_col2 = st.columns([1,2]) # splits each column into sub-columns, where first subcolumn takes up one unit and the next one takes up 3 units of space
    kpi_col1.image('assets/success_rate_icon.png', width=30) 
    kpi_col2.metric("Success Rate", f"{(filtered_cup_df['status'] == 'SUCCESS').mean()*100:.1f}%")

with col5:
    kpi_col1, kpi_col2 = st.columns([1,2]) # splits each column into sub-columns, where first subcolumn takes up one unit and the next one takes up 3 units of space
    kpi_col1.image('assets/error_type_icon.png', width=30) 
    kpi_col2.metric("Error Types", filtered_error_df['error_type'].nunique())


# charts 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cup Count Distribution")

    cleaned_cup_df = filtered_cup_df[filtered_cup_df['cup_count'] < 250] # from main script data analysis, <250 to take out anomaly (261)
    fig = px.histogram(cleaned_cup_df, x='cup_count', nbins=10, title='Histogram of Cup Count', 
                       color_discrete_sequence=['#469B9D'])  # dark teal color
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True) # 2nd argument sets width to match that of the conatiner

with col2:
    st.subheader("Analysing Top N Error Types")

    # make separate df of error types and their counts 
    error_counts = filtered_error_df['error_type'].value_counts() # value counts returns a series with the values and their correspnding frequency 
    error_counts_df = pd.DataFrame({'error_type': error_counts.index, 'count': error_counts.values})
    total_errors = error_counts_df['count'].sum()
    error_counts_df['percentage'] = error_counts_df['count'] / total_errors * 100
    
    # sort by percentage in descending order
    error_counts_df = error_counts_df.sort_values('percentage', ascending=False)

    # slider for selecting top N error types
    error_types = filtered_error_df['error_type'].nunique()

    # first check if error types == 0. less frequent case as first test
    if(error_types == 0):
        # display error message 
        st.markdown(
            """
            <div class="error-message">
                <p class="medium-font">⚠️ Selected branchId cannot be found in Error Log!</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # placeholder for error type chart (empty in this case)
        st.write("No error types to display.")

    else:
         # display slider according to number of error types 
         if(error_types>=3): top_n = st.slider("Select top N error types to display: ", min_value=0, max_value=len(error_counts.index), value=3)
         else: top_n = st.slider("Select top N error types to display: ", min_value=0, max_value=len(error_counts.index), value=1)
         top_n_df = error_counts_df.head(top_n)
 
         fig = go.Figure(go.Bar(
             y=top_n_df['error_type'],
             x=top_n_df['count'],
             orientation='h',
             text=top_n_df['percentage'].apply(lambda x: f'{x:.1f}%'),
             textposition='outside', # outside is easier to read imo
             marker_color='salmon',
             hoverinfo='text',
             hovertext=[f"{error}<br>Count: {count}<br>Percentage: {percentage:.1f}%" 
                     for error, count, percentage in zip(top_n_df['error_type'], top_n_df['count'], top_n_df['percentage'])]
         ))

         fig.update_layout(
             title=f"Top {top_n} Error Type(s) in this branch:",
             xaxis_title="Count",
             yaxis_title="Error Type",
             height=max(400, top_n * 30),  # Adjust height based on number of bars
             yaxis={'categoryorder':'total ascending'}
         )
         st.plotly_chart(fig)       

        
# additional analysis 
st.header("Further Analysis")
tab1, tab2, tab3 = st.tabs(["Cup Status", "Model Training", "Model Prediction"])

with tab1:
    st.subheader("Comparing Cup Counts by Status")
    selected_status = st.selectbox('Select status from the given options:', ['All'] + list(cup_df['status'].unique()))
    # filter data based on status 
    if selected_status != 'All':
        selected_df = filtered_cup_df[filtered_cup_df['status'] == selected_status]
    else:
        selected_df = filtered_cup_df

    # aggregate data for better performance
    agg_df = selected_df.groupby('status').agg({
        'total_cup': 'sum', 
        'cup_count': 'sum',
        'user_count': 'sum'
    }).reset_index()
    # st.write(agg_df)

    # define shades of teal
    teal_shades = ['#004d4d', '#009688', '#4db6b6']

    # bar chart
    fig = go.Figure(data=[
        go.Bar(name='Sum of total_cup', x=agg_df['status'], y=agg_df['total_cup'], text=agg_df['total_cup'], 
               textposition='outside', marker=dict(color=teal_shades[0])),  # First shade of teal),
        go.Bar(name='Sum of cup_count', x=agg_df['status'], y=agg_df['cup_count'], text=agg_df['cup_count'], 
               textposition='outside', marker=dict(color=teal_shades[1])),
        go.Bar(name='Sum of user_count', x=agg_df['status'], y=agg_df['user_count'], text=agg_df['total_cup'], 
               textposition='outside', marker=dict(color=teal_shades[2]))
    ])

    fig.update_layout(
        barmode='group',
        xaxis_title='Status',
        yaxis_title='Count',
        legend_title='Metric'
    )
    st.plotly_chart(fig)

with tab2: 
    st.subheader("Time to Train a Model!")   
    X = cup_df[['cup_count', 'total_cup', 'user_count', 'match_status', 'warning_status']]
    y = (cup_df['status'] == 'SUCCESS').astype(int)
    st.markdown("Edit the following hyperparameters and observe how the performance changes:")

    # separate into 2 columns, sel_col is for selecting hyperparameters. disp_col is for results 
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('max_depth of the model:', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox('Number of trees:', options=[100,200,300,'No Limit'], index=0) # default value is first element 
    
    bootstrap = bool(sel_col.selectbox('Bootstrap:', options=['True', 'False']))
    criterion = sel_col.selectbox('Select the criterion:', options=['entropy','gini','log_loss'])

    # make model based on chosen hyperparameters
    if n_estimators == 'No Limit':
        model = RandomForestClassifier(max_depth=max_depth, criterion=criterion, bootstrap=bootstrap)
    else: 
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, bootstrap=bootstrap)    

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # scale 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) # scale separately to prevent data leakage 

    # train and evaluate model 
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1_score = f1_score(y_test, predictions)

    # display confusion matrix 
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(3,2))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Greens') # cmap colors: https://python-graph-gallery.com/92-control-color-in-seaborn-heatmaps/
    plt.title('Confusion Matrix', fontsize=5)
    plt.ylabel('Actual', fontsize=5)
    plt.xlabel('Predicted', fontsize=5)
    # plt.tick_params(labelsize=8)
    plt.tight_layout() # adjust the plot to fill the figure 
    disp_col.pyplot(fig)

    # results
    disp_col.subheader(f"Model Accuracy: {accuracy: .2f}")
    disp_col.subheader(f"F1 Score: {f1_score: .2f}")

with tab3:
    st.subheader("Predict Success Probability")
    st.markdown("Edit the following input variables to predict the success probability:")
    cup_count = st.number_input("Cup Count", min_value=0, max_value=100, value=1)
    total_cup = st.number_input("Total Cup", min_value=0, max_value=100, value=1)
    user_count = st.number_input("User Count", min_value=0, max_value=100, value=1)
    st.markdown("Set match_status = True?")
    match_status = st.checkbox("Yes", key=2)
    st.markdown("Set warning_status = True?")
    warning_status = st.checkbox("Yes", key=3)

    if st.button("Predict"):
        prediction = model.predict_proba([[cup_count, total_cup, user_count, match_status, warning_status]])
        st.write(f"Probability of Success: {prediction[0][1]:.2f}")
