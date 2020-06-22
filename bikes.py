import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
filePath = '/cxldata/datasets/project/bikes.csv'
bikesData = pd.read_csv(filePath)
print(bikesData.info())
columnsToDrop = ['instant','casual','registered','atemp','dteday']
def set_day(df):
         days = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thr", "Fri"]
    temp = ['d']*df.shape[0]
    i = 0
    indx = 0
    cur_day = df.weekday[0]
    for day in df.weekday:
        temp[indx] = days[(day-cur_day+7)%7]
        indx += 1
    df['dayWeek'] = temp
    return df
def mnth_cnt(df):
       import itertools
    yr = df['yr'].tolist()
    mnth = df['mnth'].tolist()
    out = [0] * df.shape[0]
    indx = 0
    for x, y in zip(mnth, yr):
        out[indx] = x + 12 * y
        indx += 1
    return out


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


bikesData = bikesData.drop(columnsToDrop,1)
columnsToScale=['temp','hum','windspeed']
scaler = StandardScaler()
bikesData[columnsToScale]=scaler.fit_transform(bikesData[columnsToScale])
bikesData['dayCount']=pd.Series(range(bikesData.shape[0]))/24
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(bikesData,test_size=0.3,random_state=42)
train_set.sort_values('dayCount',axis=0,inplace=True)
train_set.sort_values('dayCount',axis=0,inplace=True)
print(len(train_set),"train+",len(test_set),"test")
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
trainingCols = train_set.drop(['cnt'], axis=1)
trainingLabels = train_set['cnt']
dec_reg = DecisionTreeRegressor(random_state = 42)


dt_mae_scores = -cross_val_score(dec_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")


display_scores(dt_mae_scores)


dt_mse_scores = np.sqrt(-cross_val_score(dec_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))


display_scores(dt_mse_scores)
lin_reg = LinearRegression()
lr_mae_scores= -cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")
display_scores(lr_mae_scores)
lr_mse_scores=(np.sqrt(-cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error")))
display_scores(lr_mae_scores)
forest_reg = RandomForestRegressor(n_estimators=40,random_state = 42)
rf_mae_scores = -cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error")
display_scores(rf_mae_scores)
rf_mse_scores = np.sqrt(-cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error"))
display_scores(rf_mse_scores)
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [120, 150], 'max_features': [10, 12], 'max_depth': [15, 28]},
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(trainingCols, trainingLabels)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
final_model = grid_search.best_estimator_
test_set.sort_values('dayCount', axis= 0, inplace=True)
test_x_cols = (test_set.drop(['cnt'], axis=1)).columns.values
X_test = test_set.loc[:,test_x_cols]
test_y_cols='cnt'
y_test = test_set.loc[:,test_y_cols]
test_set.loc[:,'predictedCounts_test'] = final_model.predict(X_test)
mse = mean_squared_error(y_test, test_set.loc[:,'predictedCounts_test'])
final_mse=mse
print(final_mse)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
columnsToDrop = ['instant','casual','registered','atemp','dteday']
bikesData = bikesData.drop(columnsToDrop,1)


bikesData['isWorking'] = np.where(np.logical_and(bikesData.workingday==1,bikesData.holiday==0),1,0)
bikesData['monthCount'] = mnth_cnt(bikesData)
bikesData['xformHr'] = np.where(bikesData.hr>4,bikesData.hr-5,bikesData.hr+19)
bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24
bikesData['xformWorkHr'] = bikesData.isWorking*24 + bikesData.xformHr
bikesData = set_day(bikesData)
bikesData.describe()


columnsToScale = ['temp','hum','windspeed']
scaler = StandardScaler()
bikesData[columnsToScale] = scaler.fit_transform(bikesData[columnsToScale])
arry = bikesData[columnsToScale].as_matrix()
bikesData[columnsToScale] = preprocessing.scale(arry)




from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        isWorking = np.where(np.logical_and(X.loc[:,'workingday']==1,X.loc[:,'holiday']==0),1,0)
        xformHr = np.where(X.loc[:,'hr']>4,X.loc[:,'hr']-5,X.loc[:,'hr']+19)
        xformWorkHr = isWorking*24 + xformHr
        return np.c_[X, isWorking, xformHr, xformWorkHr]


attr_adder = CombinedAttributesAdder()
bikesData1 = attr_adder.transform(bikesData)
bikesData = pd.DataFrame(bikesData1, columns=list(bikesData.columns)+["isWorking", "xformHr", "xformWorkHr"])
bikesData.head()