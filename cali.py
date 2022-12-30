# Load required packages
import streamlit as st 
from sklearn.datasets import fetch_california_housing

# Load the dataset
def california_housing():
    st.session_state.california_housing = fetch_california_housing(as_frame=True)

if 'california_housing' not in st.session_state:
    california_housing()

st.write(st.session_state.california_housing.DESCR)
