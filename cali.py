# Load required packages
import streamlit as st 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import shap
from sklearn import metrics
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import umap
import umap.plot
import altair as alt
import plotly.express as px

st.title('Assessing Best Model Features on the California Housing Set')
st.subheader('Created by: Stephan de Goede')

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
if st.button('click here to view source code',key=1):
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
if st.button('click here to view source code'):
    st.code(''' # Define the search space
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
    xgb_regressor = xgb.XGBRegressor(n_jobs=None,tree_method='gpu_hist',seed=37)
    # Perform a 3-fold cross-validated random-search of the grid for 1000 possible combinations, evaluating just a fraction of the total grid.
    param_comb = 1000
    rs_model = RandomizedSearchCV(xgb_regressor,param_distributions=params,n_iter=param_comb,scoring="neg_root_mean_squared_error",n_jobs=None,cv=3,verbose=3,random_state=37) 
    # Actual fitting procedure:
    rs_model.fit(x_train,y_train)
    # Perform 10-fold cross validation on the best model. 
    print(np.mean(cross_val_score(rs_model.best_estimator_,x_test,y_test,cv=10)))
    # Save the best model
    rs_model.best_estimator_.save_model("xgboost_model.json),"''',)

# Load the cross-validated tuned XGBOOST model
def xgb_loaded():
    st.session_state.xbg_loaded = xgb.XGBRegressor()
    st.session_state.xbg_loaded.load_model("xgboost_model.json")
    st.session_state.xbg_loaded.set_params(tree_method ='hist')
    st.session_state.xbg_loaded.set_params(n_jobs = '-1')
    st.session_state.loaded_predictions = st.session_state.xbg_loaded.predict(x_test)
    st.session_state.xgb_rmse = np.sqrt(metrics.mean_squared_error(y_test, st.session_state.loaded_predictions))
    st.session_state.loaded_train_predictions = st.session_state.xbg_loaded.predict(x_train)

if 'xgb_loaded' not in st.session_state:
    xgb_loaded()

st.caption('Note that this model is previously fitted and loaded here, and set to CPU mode due to performance reasons')

st.write(f'RMSE of the XGBoost model on the test set: {round(st.session_state.xgb_rmse,2)}') 
st.write(f'This means that the model, on average, has an error in predicting the median house value of: {round(st.session_state.xgb_rmse,2)}  Times $1.000.')
st.write(f'This model scores  {abs(round(((round(st.session_state.xgb_rmse,2)- round(st.session_state.rmse,2))/ round(st.session_state.rmse,2))*100,2))} percent better on the unseen test data than the benchmark model.')
    
st.title('Explaining the XGBoost model.. to a wider audience')
st.write('below, all separate decision trees that have been build by training the model can be reviewed')
ntree=st.number_input('Select the desired record for detailed explanation on the training set'
                        , min_value=min(range(st.session_state.xbg_loaded.best_iteration))
                        , max_value=max(range(st.session_state.xbg_loaded.best_iteration+1))
                                       )

if st.button('click to see the selected tree'):
    graph = xgb.to_graphviz(st.session_state.xbg_loaded,num_trees=ntree)
    tree = graph.render('tree', format='jpg')
    st.image(tree, width= 17587)
    st.write('Too large? try zooming out using your browser')
    

st.write('Using the standard XGBOOST importance plot feature, exposes the fact that the most important feature is not stable, select'
             ' different importance types using the select box below')
importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
importance_plot = xgb.plot_importance(st.session_state.xbg_loaded,importance_type=importance_type)
plt.title ('xgboost.plot_importance(best XGBoost model) importance type = '+ str(importance_type))
st.pyplot(importance_plot.figure)
plt.clf()

def shap_explainer():
    st.session_state.explainer = pickle.load(open('explainer.sav', 'rb'))
    st.session_state.shap_values = pickle.load(open('shap_values.sav', 'rb'))
    
if 'explainer' not in st.session_state:
    shap_explainer()

st.write('To handle this inconsistency, SHAP values give robust details, among which is feature importance')
st_shap(shap.summary_plot(st.session_state.shap_values,x_train,plot_type="bar",feature_names=st.session_state.california_housing.feature_names))
st.caption('Note that this calculation is loaded from disk and loaded here, due to performance reasons')
st.write('Below the source code of the calculation can be reviewed')
if st.button('click here to view source code',key=2):
    st.code(''' # Load the XGB Model from disk
    xbg_loaded = xgb.XGBRegressor()
    xbg_loaded.load_model("xgboost_model.json")
    # Compute SHAP Values
    explainer = shap.TreeExplainer(xbg_loaded)
    shap_values = explainer.shap_values(x_train)
    # Save calculations to disk
    pickle.dump(explainer, open('explainer.sav', 'wb'))
    pickle.dump(shap_values, open('shap_values.sav', 'wb'))''')
st.write('''SHAP values can also be used to represent the distribution of the training set of the respectable
            SHAP value in relation with the Target value, in this case the median house value for California districts (MedHouseVal)''')
st_shap(shap.summary_plot(st.session_state.shap_values,x_train,feature_names=st.session_state.california_housing.feature_names))

st.write('''Another example of SHAP values is for GDPR regulation, one should be able to give detailed information as to'
               why a specific prediction was made.''')
expectation = st.session_state.explainer.expected_value
individual = st.number_input('Select the desired record from the training set for detailed explanation.'
                                           , min_value=min(range(len(x_train)))
                                           , max_value=max(range(len(x_train))))
predicted_values = st.session_state.loaded_train_predictions
real_value = y_train[individual]
st.write('The real median house value for this individual record is: '+str(real_value))
st.write('The predicted median house value for this individual record is: '+str(predicted_values[individual]))
st.write(f'''This prediction is calculated as follows:
              The average median house value: {str(expectation)} + the sum of the SHAP values. ''')
st.write(f'''For this individual record the sum of the SHAP values is: {str(sum(st.session_state.shap_values[individual,:]))}''')
st.write(f'''This yields to a predicted value of median house value of: {str(expectation)} + {str(sum(st.session_state.shap_values[individual,:]))}
                = {expectation+(sum(st.session_state.shap_values[individual,:]))}''')
st.write('Which features caused this specific prediction? features in red increased the prediction, in blue decreased them')
st_shap(shap.force_plot(st.session_state.explainer.expected_value, st.session_state.shap_values[individual],x_train[individual],feature_names=st.session_state.california_housing.feature_names))
st.write('''In the plot above, the feature values are shown. The SHAP values are represented by the length of the specific bar.
              However, it is not quite clear what each single SHAP value is exactly, this can be seen below, if wanted.''')
def shap_table():
    st.session_state.shap_table = pd.DataFrame(st.session_state.shap_values,columns=st.session_state.california_housing.feature_names)

if st.button('Click here to see a drilldown of the SHAP values'):
    if 'shap_table' not in st.session_state:
        shap_table()
    st.table(st.session_state.shap_table.iloc[individual])

st.subheader('Developing a deeper understanding of the data using SHAP: Interaction effects')

st.write('''When selecting features below, note that the algorithm automatically plots the selected feature, with the feature that'
              ' it most likely interacts with. However, the final judgement lies in the eyes of the beholder. Typically, when there is'
              ' an interaction effect, points diverge strongly''')

st.write('''In the slider below, select the number of features to inspect for possible interaction effects.'
              'These are ordered based on feature importance in the model.''')

ranges = st.slider('Please select the number of features',min_value=min(range(len(st.session_state.california_housing.feature_names)))+1, max_value=max(range(len(st.session_state.california_housing.feature_names)))+1,value=1)
if ranges-1 == 0:
    st.write('you have selected the most importance feature')
elif ranges == len(st.session_state.california_housing.feature_names):
    st.write('you have selected all possible features')
else:
    st.write('you have selected the top:',ranges,'important features')
for rank in range(ranges):
    ingest=('rank('+str(rank)+')')
    st_shap(shap.dependence_plot(ingest,st.session_state.shap_values,x_train,feature_names=st.session_state.california_housing.feature_names))

st.write('Conclusion: It is to my best judgement that there are no significant interaction effects within the features of this model.')

st.subheader(' Understanding groups: Dimensionality reduction using UMAP and plotting the embeddings in 2D, focussing on potential clusters' )

st.write('''Below is an interactive UMAP plot. If you drag your mouse whilst holding the left mouse button, characteristics of 
              all non-spatial features are automatically shown as histograms. Herewith one can get meaningful insights of different groups in for example:
              targeting and communication in a marketing context, or to understand the distribution of the features, for the selected segment.
            Note that the target variable here is the sum of all the SHAP-values for that given datapoint. Furthermore, it is binned into 4 equal groups,for better interpretability. Each separate color stands for a specific group,'
             ' where pink signals the highest 25% of (predicted) median house value for California districts. In addition, underneath the graphs, a filter is added to quickly select a specific quadrant''')

# mapper = umap.UMAP().fit(st.session_state.california_housing.data)
def umap_embeddings():
   st.session_state.umap_embeddings = pickle.load(open('umap_embeddings.sav','rb'))
   st.session_state.umap_dataframe = pd.DataFrame(st.session_state.umap_embeddings, columns = ['x','y'])
   st.session_state.umap_dataframe['TARGET'] = st.session_state.shap_values.sum(1).astype(np.float64)
   labels = ['lowest 25%','25 to 50%','50-75%','highest 25%']
   st.session_state.umap_dataframe['TARGET_BINNED'] = pd.cut(st.session_state.umap_dataframe['TARGET'], bins=4,labels=labels).astype(str)
   st.session_state.umap_dataframe['MedInc'] = x_train[:,0]
   st.session_state.umap_dataframe['HouseAge'] = x_train[:,1]
   st.session_state.umap_dataframe['AveRooms'] = x_train[:,2]
   st.session_state.umap_dataframe['AveBedrms'] = x_train[:,3]
   st.session_state.umap_dataframe['Population'] = x_train[:,4]
   st.session_state.umap_dataframe['latitude'] = st.session_state.california_housing.frame['Latitude']
   st.session_state.umap_dataframe['longitude'] = st.session_state.california_housing.frame['Longitude']

if 'umap_embeddings' not in st.session_state:
        umap_embeddings()

st.caption('Note that these embeddings are loaded from disk, due to performance reasons')
st.write('Below the source code of the UMAP embeddings can be reviewed')
if st.button('click here to view source code', key=3):
    st.code(''' # Create UMAP embedding
embedding = umap.UMAP(n_neighbors=25,
                      min_dist=0.0,
                      metric='euclidean').fit_transform(st.session_state.shap_values)
# Save embeddings to disk
pickle.dump(embedding, open('umap_embeddings.sav', 'wb'))''')

input_dropdown = alt.binding_select(options=st.session_state.umap_dataframe['TARGET_BINNED'].unique(), name='Filter_group_SHAP_Values')
group_selection = alt.selection_single(fields=['TARGET_BINNED'], bind=input_dropdown)
brush = alt.selection(type='interval',resolve='global')
points_UMAP = alt.Chart(st.session_state.umap_dataframe).mark_point().encode(
           x='x:Q',
           y='y:Q',
           color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
           brush).properties(
        width=750,
        height=500
    ).add_selection(
    group_selection
).transform_filter(
    group_selection)


MedInc = alt.Chart(st.session_state.umap_dataframe).mark_bar().encode(
    x='MedIncbin:N',
    y="MedInc:Q"
).properties(
    width=300,
    height=300
).transform_filter(
    brush
).transform_bin(
    "MedIncbin",
    field="MedInc",
    bin=alt.Bin(maxbins=20)
).add_selection(
    group_selection
).transform_filter(
    group_selection)

HouseAge = alt.Chart(st.session_state.umap_dataframe).mark_bar().encode(
    x='HouseAgebin:N',
    y="HouseAge:Q"
).properties(
    width=300,
    height=300
).transform_filter(
    brush
).transform_bin(
    "HouseAgebin",
    field="HouseAge",
    bin=alt.Bin(maxbins=20)
).add_selection(
    group_selection
).transform_filter(
    group_selection)

AveRooms = alt.Chart(st.session_state.umap_dataframe).mark_bar().encode(
    x='AveRoomsbin:N',
    y="AveRooms:Q"
).properties(
    width=300,
    height=300
).transform_filter(
    brush
).transform_bin(
    "AveRoomsbin",
    field="AveRooms",
    bin=alt.Bin(maxbins=20)
).add_selection(
    group_selection
).transform_filter(
    group_selection)

AveBedrms = alt.Chart(st.session_state.umap_dataframe).mark_bar().encode(
    x='AveBedrmsbin:N',
    y="AveBedrms:Q"
).properties(
    width=300,
    height=300
).transform_filter(
    brush
).transform_bin(
    "AveBedrmsbin",
    field="AveBedrms",
    bin=alt.Bin(maxbins=20)
).add_selection(
    group_selection
).transform_filter(
    group_selection)

Population = alt.Chart(st.session_state.umap_dataframe).mark_bar().encode(
    x='Populationbin:N',
    y="Population:Q"
).properties(
    width=300,
    height=300
).transform_filter(
    brush
).transform_bin(
    "Populationbin",
    field="Population",
    bin=alt.Bin(maxbins=20)
).add_selection(
    group_selection
).transform_filter(
    group_selection)

st.altair_chart(points_UMAP & MedInc & HouseAge & AveRooms & AveBedrms & Population)

def geospatial():
    st.session_state.geospatial =  st.session_state.umap_dataframe.query("TARGET_BINNED in ['highest 25%','lowest 25%']")

if 'geospatial' not in st.session_state:
        geospatial()

st.subheader('Location, Location, Location......?')

st.write('''So, as the famous real-estate mantra goes when assessing property value, it's all about the location. So, why don't look at the geospatial data from the dataset? 
Below, the geographical representation of the California Housing dataset is plotted. Both the minimum and the maximum SHAP values groups are plotted on the map. 
As can be seen, using this information alone, it is very difficult to see a clear distinguish, if possible at all.''' )

geo_fig = px.scatter_geo(st.session_state.geospatial, lat='latitude', lon='longitude', color='TARGET_BINNED', locationmode='USA-states', center=dict(lat=37.8600, lon=-122.2200), scope='usa')
st.plotly_chart(geo_fig)


st.write('''As you came this far, hopefully you have enjoyed the app :) ''')
st.write('Source code: https://github.com/sgoede/streamlit-california-app')
st.write('connect with me @ https://nl.linkedin.com/in/stephandegoede')
st.write('Deployed on Streamlit Share')
