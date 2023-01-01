# Load required packages
import streamlit as st 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the dataset
def california_housing():
    st.session_state.california_housing = fetch_california_housing(as_frame=True)

if 'california_housing' not in st.session_state:
    california_housing()

st.write('The detailed description of the data is listed below')
st.write(st.session_state.california_housing.DESCR)

st.subheader('Creating a Pandas Dataframe from the original dataset')
st.table(st.session_state.california_housing.frame.head())

st.write(st.session_state.california_housing.frame.columns)

# creating test and training set
X = st.session_state.california_housing.frame.loc[:, st.session_state.california_housing.frame.columns != 'MedHouseVal'].values
y = st.session_state.california_housing.frame.loc[:, st.session_state.california_housing.frame.columns == 'MedHouseVal'].values
x_train, x_test, y_train, y_test = train_test_split (X,y, test_size = 0.25, random_state=37)