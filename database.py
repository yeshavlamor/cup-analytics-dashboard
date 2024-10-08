import streamlit as st 
import pandas as pd

# function to establish database connection 
@st.cache_resource # decorator to cache functions that return global resources 
def init_connection(): 
     return st.connection("or_cup_uat", type="sql",autocommit=True)
conn = init_connection()

# functions to get data and clean 
@st.cache_data # decorator that caches the output of the function, if input argument does not change in future runs, cached data is used instead of re-executing the function 
def get_error_data():
     return conn.query('SELECT * from "ErrorLog"')

@st.cache_data
def get_cup_data():
     cup_data = conn.query('SELECT * from "CupLog"')
     columns_to_keep = [
          "id",
          "total_cup",
          "cup_size",
          "previous_total_cup",
          "carbon_saving",
          "cup_count",
          "branchId",
          "crop_status", 
          "cero",
          "points",
          "no_tube",
          "size_not_correct",
          "not_single_tube",
          "warning_status",
          "match_status",
          "user_count",
          "status"
          ]
     return cup_data[columns_to_keep]

@st.cache_data
def get_branch_data():
     branch_data = conn.query('SELECT * from "Branch"')
     return branch_data.rename(columns={'id': 'branchId', 'code': 'branch_code'})
     
@st.cache_data
def get_all_processed_data():          
     error_data = get_error_data()
     cup_data = get_cup_data()
     branch_data = get_branch_data()

     # merge cup_data with branch_data to add branch_code
     cup_data = pd.merge(cup_data, branch_data[['branchId', 'branch_code']], on='branchId', how='left')

     return {
          'error_data': error_data,
          'cup_data': cup_data,
          'branch_data': branch_data
     }

# check 

# all_data = get_all_processed_data()
# error_data = all_data['error_data']
# cup_data = all_data['cup_data']
# st.table(cup_data)
# st.table(error_data)

# pd.merge:
# - cup_data is the left df, branch_data is the right df where we are using 2 columns 
# - we join right df to left df on branchid
# - we do this through a left join, which means:
#      - all rows from left df will be included
#      - rows from right df will only be included if they have a matcing branchid found in left df

