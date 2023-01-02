# Load required packages
import streamlit as st 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
def california_housing():
    st.session_state.california_housing = fetch_california_housing(as_frame=True)

if 'california_housing' not in st.session_state:
    california_housing()

st.write('The detailed description of the data is listed below')
st.write(st.session_state.california_housing.DESCR)

st.subheader('Creating a Pandas Dataframe from the original dataset')
st.table(st.session_state.california_housing.frame.head())

# creating test and training set
X = st.session_state.california_housing.frame.loc[:, st.session_state.california_housing.frame.columns != 'MedHouseVal'].values
y = st.session_state.california_housing.frame.loc[:, st.session_state.california_housing.frame.columns == 'MedHouseVal'].values
x_train, x_test, y_train, y_test = train_test_split (X,y, test_size = 0.25, random_state=37)

st.subheader('Setting the benchmark: Fitting a simple linear regression model')
st.caption('Note that this model is previously fitted and loaded here, due to performance reasons')
st.write('Below the source code of the linear model can be reviewed')
st.code('''
# creating test and training set
X = st.session_state.california_housing.frame.loc[:, st.session_state.california_housing.frame.columns != 'MedHouseVal'].values
y = st.session_state.california_housing.frame.loc[:, st.session_state.california_housing.frame.columns == 'MedHouseVal'].values
x_train, x_test, y_train, y_test = train_test_split (X,y, test_size = 0.25, random_state=37)

# linear model fit
reg = LinearRegression().fit(x_train, y_train)
reg.score(x_test,y_test)

# Save model to disk
joblib.dump(reg, 'lin_reg.pkl')''')

def lin_reg():
    st.session_state.lin_reg = joblib.load('lin_reg.pkl')
    st.session_state.r_sq = st.session_state.lin_reg.score(x_train,y_train)
    st.session_state.r_sq_t = st.session_state.lin_reg.score(x_test,y_test)
    st.session_state.y_pred = st.session_state.lin_reg.predict(x_test)

if 'lin_reg' not in st.session_state:
    lin_reg()

st.write('The benchmark model on this dataset yields the following results:')
st.write('R-squared: Training Set',round(st.session_state.r_sq,2))
st.write('R-squared: Test Set',round(st.session_state.r_sq_t,2))
st.write(f'Since the benchmark model has an R-square of {round(st.session_state.r_sq_t,2)} on the test set, we will continue with a linear kernel')