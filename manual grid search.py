#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().ast_node_interactivity = 'all'
import pandas as pd
import numpy
from datetime import datetime as dt

from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
get_ipython().ast_node_interactivity = 'all'
import pandas 
import numpy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.ensemble import BaggingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[8]:


from sklearn.neural_network import MLPRegressor


# In[2]:


df= pd.read_csv( "final_ml_data.csv")


# In[10]:


df['articleDate'] = pd.to_datetime(df['articleDate'])
df['day_of_week'] = df['articleDate'].dt.day_name()
df['month'] = df['articleDate'].dt.month
df['year'] = df['articleDate'].dt.year


# In[29]:


df['percentchange_marketcap'] = ((df['close_marketcap']/df['open_marketcap']) - 1) * 100


# In[30]:


def replace_source(source):
    if df['source'].value_counts()[source] <= 50:
        return 'other'
    else:
        return source

# Apply the function to the 'source' column
df['source'] = df['source'].apply(replace_source)


# In[31]:


df.dtypes


# In[45]:


#Regression
X_train, X_test, y_train, y_test = train_test_split(df, df.percentchange_marketcap, test_size = .5)
lstNum = ["n_companies", "positive_sentiment", "negative_sentiment", "neutral_sentiment", "title_polarity", "title_subjectivity", "open_marketcap"]
lstCat = ["StockSymbol", "source", "day_of_week", "country_isUS"]

pipeNum = Pipeline( [
('selector', ColumnTransformer([ ('selector', 'passthrough', lstNum) ] ) ),
('scaler', StandardScaler() )
])
pipeCat = Pipeline([
('selector', ColumnTransformer([ ('selector', 'passthrough', lstCat) ] )),
('encoder', OneHotEncoder( dtype=int, drop='first', sparse_output=False ) )
])
preprocessor = FeatureUnion([
('cat', pipeCat ),
('num', pipeNum )
])

#from sklearn.neural_network import MLPRegressor

#pipeNnTune = Pipeline( [
#('preprocessor', preprocessor ),
#('model', MLPRegressor( hidden_layer_sizes = (15,15,15), activation = 'relu', max_iter = 5000, learning_rate = 'adaptive') ) 
#    ])


#pipeNnTune.fit( X_train, y_train )
#predTrainNnTune = pipeNnTune.predict( X_train )
#predTestNnTune = pipeNnTune.predict( X_test )

#sklearn.metrics.r2_score( y_train, predTrainNnTune )



# In[ ]:


"month", "year"


# In[76]:


from sklearn.neural_network import MLPRegressor

pipeNnTune = Pipeline( [
('preprocessor', preprocessor ),
('model', MLPRegressor( max_iter = 5000, activation = 'identity', hidden_layer_sizes = (5,5,5)) ) 
    ])


pipeNnTune.fit( X_train, y_train )
predTrainNnTune = pipeNnTune.predict( X_train )
predTestNnTune = pipeNnTune.predict( X_test )

sklearn.metrics.r2_score( y_train, predTrainNnTune )
sklearn.metrics.r2_score( y_test, predTestNnTune )


# In[50]:


sklearn.metrics.r2_score( y_train, predTrainNnTune )


# In[49]:


paramGridMLP =  {'hidden_layer_sizes': [ (5,5),(10,10,10),(20,20,20),(40,40),(60,60),(30,30,30) ],
'activation': ['identity', 'logistic', 'tanh', 'relu']} 



# In[ ]:


#optimal model
pipeNnOptimal1 = Pipeline( [
('preprocessor', preprocessor ),
('model', MLPRegressor( max_iter = 5000, activation = 'relu', hidden_layer_sizes = (10,10,10)) ) 
    ])


pipeNnOptimal1.fit( X_train, y_train )
predTrainNnOptimal1 = pipeNnTune.predict( X_train )
predTestNnOptimal1 = pipeNnTune.predict( X_test )

sklearn.metrics.r2_score( y_train, predTrainNnOptimal1 )
sklearn.metrics.r2_score( y_test, predTestNnOptimal1 )


# In[ ]:


#second best model
pipeNnOptimal2 = Pipeline( [
('preprocessor', preprocessor ),
('model', MLPRegressor( max_iter = 5000, activation = 'tanh', hidden_layer_sizes = (20,20,20)) ) 
    ])


pipeNnOptimal2.fit( X_train, y_train )
predTrainNnOptimal2 = pipeNnOptimal2.predict( X_train )
predTestNnOptimal2 = pipeNnOptimal2.predict( X_test )

sklearn.metrics.r2_score( y_train, predTrainNnOptimal2 )
sklearn.metrics.r2_score( y_test, predTestNnOptimal2 )


# In[ ]:


#third best model
pipeNnOptimal3 = Pipeline( [
('preprocessor', preprocessor ),
('model', MLPRegressor( max_iter = 5000, activation = 'logistic', hidden_layer_sizes = (20,20,20)) ) 
    ])


pipeNnOptimal3.fit( X_train, y_train )
predTrainNnOptimal3 = pipeNnOptimal3.predict( X_train )
predTestNnOptimal3 = pipeNnOptimal3.predict( X_test )

sklearn.metrics.r2_score( y_train, predTrainNnOptimal3 )
sklearn.metrics.r2_score( y_test, predTestNnOptimal3 )

