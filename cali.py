# Load required packages
import streamlit as st 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
# import matplotlib as mpl
from sklearn import metrics
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from io import BytesIO
from graphviz import Digraph

# mpl.rcParams['figure.figsize'] = 160, 48

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

st.subheader('Setting the benchmark: Fit a simple linear regression- model')
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
    st.session_state.rmse = np.sqrt(metrics.mean_squared_error(y_test, st.session_state.y_pred))

if 'lin_reg' not in st.session_state:
    lin_reg()

st.write('The benchmark model on this dataset yields the following results:')
st.write('R-squared: Training Set',round(st.session_state.r_sq,2))
st.write('R-squared: Test Set',round(st.session_state.r_sq_t,2))
st.write(f'Since the benchmark model has an R-square of {round(st.session_state.r_sq_t,2)} on the test set, we will continue with a linear kernel')
st.write('For easier comparison with other models, the Root Mean Squared Error (RMSE) score is also reported')
st.write('RMSE of baseline model on test set:', round(st.session_state.rmse,2))

# Building the dashboard on XGBOOST model:
st.subheader('Model the California Housing Dataset using XGBOOST')
st.write('Below the source code of the XGB model can be reviewed')
st.code('''# Define the search space
params = { 
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.0001,0.001, 0.01, 0.1, 1] ,
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": range(3,21,3),
    # Gamma specifies the minimum loss reduction required to make a split.
    "gamma": [i/10.0 for i in range(0,5)],
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [i/10.0 for i in range(3,10)],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # the minimum number of samples that a node can represent in order to be split further.
    "min_child_weight" : range(1,9,2)
    }
# Initiate the xgboost regressor (note GPU training is enabled)
rs_model=RandomizedSearchCV(xgb_regressor,param_distributions=params,n_iter=param_comb, scoring="neg_root_mean_squared_error",n_jobs=None,cv=3,verbose=3,random_state=37) 
# Perform a 3-fold cross-validated random-search of the grid for 18.900 possible combinations, evaluating ~25% of the total grid.
param_comb = 18900
rs_model=RandomizedSearchCV(xgb_regressor,param_distributions=params,n_iter=param_comb, scoring="neg_root_mean_squared_error",n_jobs=None,cv=3,verbose=3,random_state=37)
# Actual fitting procedure:
rs_model.fit(x_train,y_train)
# Perform 10-fold cross validation on the best model. 
print(np.mean(cross_val_score(rs_model.best_estimator_,x_test,y_test,cv=10)))
# Save the best model
rs_model.best_estimator_.save_model("xgboost_model.json)"''')

# Load the cross-validated tuned XGBOOST model
def xgb_loaded():
    st.session_state.xbg_loaded = xgb.XGBRegressor()
    st.session_state.xbg_loaded.load_model("xgboost_model.json")
    st.session_state.xbg_loaded.set_params(tree_method ='hist')
    st.session_state.xbg_loaded.set_params(n_jobs = '-1')
    st.session_state.loaded_predictions = st.session_state.xbg_loaded.predict(x_test)
    st.session_state.xgb_rmse = np.sqrt(metrics.mean_squared_error(y_test, st.session_state.loaded_predictions))

if 'xgb_loaded' not in st.session_state:
    xgb_loaded()

st.caption('Note that this model is previously fitted and loaded here, and set to CPU mode due to performance reasons')

st.write(f'RMSE of the XGBoost model on the test set: {round(st.session_state.xgb_rmse,2)}') 
st.write(f'This means that the model, on average, has an error in predicting the median house value of: {round(st.session_state.xgb_rmse,2)}  Times $1.000.')
st.write(f'This model scores  {abs(round(((round(st.session_state.xgb_rmse,2)- round(st.session_state.rmse,2))/ round(st.session_state.rmse,2))*100,2))} percent better on the unseen test data than the benchmark model.')
    
st.title('Explaining the XGBoost model.. to a wider audience')
st.write('below, all seperate decision trees that have been build by training the model can be reviewed')
ntree=st.number_input('Select the desired record for detailed explanation on the training set'
                        , min_value=min(range(st.session_state.xbg_loaded.best_iteration))
                                       , max_value=max(range(st.session_state.xbg_loaded.best_iteration+1))
                                       )

if st.button('click to see the selected tree'):
    graph = xgb.to_graphviz(st.session_state.xbg_loaded,num_trees=ntree)
    tree = graph.render('tree', format='jpg')
    st.image(tree, width= round(17587/2))

st.write('Using the standard XGBOOST importance plot feature, exposes the fact that the most important feature is not stable, select'
             ' different importance types using the selectbox below')
importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
importance_plot = xgb.plot_importance(st.session_state.xbg_loaded,importance_type=importance_type)
plt.title ('xgboost.plot_importance(best XGBoost model) importance type = '+ str(importance_type))
st.pyplot(importance_plot.figure)
plt.clf()