#code for Arkon test
#libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score

from tqdm import tqdm

#load data
data_original = pd.read_csv('train_set.csv')
data = data_original.copy()

#print the headers from data
headers = data.columns

#view some patterns
examples = data.loc[0:5]

#clean data (pre-processing)
#1. quit id column and plan_duration (this not shown in test data)
data = data.drop(data.columns[[0]], axis=1) 
data = data.drop(['plan_duration'], axis=1) 
#1.2 EXTRACT feature from datetime ISO 8601
start_time = pd.to_datetime(data.loc[:,'start_time']).to_frame()
end_time = pd.to_datetime(data.loc[:,'end_time']).to_frame()
#quit datatimes columns
data = data.drop(['start_time','end_time'], axis=1) 
#START_TIME
# day
start_time['Start_Day'] = start_time['start_time'].dt.day
# month
start_time['Start_Month'] = start_time['start_time'].dt.month
# year
start_time['Start_Year'] = start_time['start_time'].dt.year
# hour
start_time['Start_hour'] = start_time['start_time'].dt.hour
# minute
start_time['Start_minute'] = start_time['start_time'].dt.minute
# Monday is 0 and Sunday is 6
start_time['Start_weekday'] = start_time['start_time'].dt.weekday
# week of the year
start_time['Start_week_of_year'] = start_time['start_time'].dt.week
#END_TIME
# day
end_time['End_Day'] = end_time['end_time'].dt.day
# month
end_time['End_Month'] = end_time['end_time'].dt.month
# year
end_time['End_Year'] = end_time['end_time'].dt.year
# hour
end_time['End_hour'] = end_time['end_time'].dt.hour
# minute
end_time['End_minute'] = end_time['end_time'].dt.minute
# Monday is 0 and Sunday is 6
end_time['End_weekday'] = end_time['end_time'].dt.weekday
# week of the year
end_time['End_week_of_year'] = end_time['end_time'].dt.week
#join new features
data = pd.concat([data, start_time.iloc[:,1:], end_time.iloc[:,1:]], axis=1)


#2. impute missing values using mean and median
print(data.isnull().sum())
missing_col = ['start_lat','start_lon','end_lat','end_lon']
for i in missing_col:
    data.loc[data.loc[:,i].isnull(),i] = data.loc[:,i].mean()
#data_2c = data.copy()

#======= VERSION 1, each class for each passholder_type
#cat to num target
targets = data["passholder_type"].unique()
targets = np.delete(targets, 5)
data['passholder_type'] = pd.factorize(data['passholder_type'])[0]
#impute missing targets
i = "passholder_type"
data.loc[data.loc[:,i] == -1,i] = data.loc[:,i].median()
print("\n", data.isnull().sum())
#sort target
data_t = data['passholder_type']
data = data.drop(['passholder_type'], axis=1)
data.loc[:,i] = data_t

targets_num = data["passholder_type"].unique()
data['passholder_type'].replace(targets_num, targets, inplace=True)

examples = data.loc[0:5]
print(data.iloc[:,data.shape[1]-1].value_counts())

#======= VERSION 2 OPTIONAL, just 2 classes, Clase A: Monthly Pass. Class B: the rest
# data_2c = pd.get_dummies(data_2c, columns=['passholder_type'])
# #view headers
# print(data_2c.columns)
# data_2c = data_2c.drop(['passholder_type_Annual Pass','passholder_type_Flex Pass','passholder_type_One Day Pass','passholder_type_Testing','passholder_type_Walk-up'], axis=1) 
# #check missing values
# print("\n", data_2c.isnull().sum())
# examples_2c = data_2c.loc[0:5]
# print(data_2c.iloc[:,data_2c.shape[1]-1].value_counts())

#Tarea 1
cols = ['Start_Day','Start_Month','Start_Year','Start_hour','Start_weekday','Start_week_of_year']
plt.figure(figsize=(12,8))
for i in range(len(cols)):
    plt.subplot(3,3,i+1)
    plt.hist(data[cols[i]], bins = len(data[cols[i]].unique()))
    plt.title(cols[i])
    plt.grid()

plt.tight_layout()
plt.show()
 
plt.figure(figsize=(8,6))
cols = ['start_station','end_station']
for i in range(len(cols)):
    plt.subplot(1,2,i+1)
    plt.hist(data[cols[i]], bins = len(data[cols[i]].unique()))
    plt.title(cols[i])
    plt.grid()

plt.tight_layout()
plt.show()

data['trip_route_category'].replace([0,1], ['Round Trip','One Way'], inplace=True)
data['trip_route_category'].value_counts().plot(kind='bar')

round_trip = data[data['trip_route_category'] == 'Round Trip']
one_way = data[data['trip_route_category'] == 'One Way']

plt.figure(figsize=(12,6))
cols = ['Start_Year','End_Year']
for i in range(len(cols)):
    plt.subplot(1,2,i+1)
    plt.hist(round_trip[cols[i]], bins = len(round_trip[cols[i]].unique()))
    plt.title(cols[i])
    plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
cols = ['Start_Year','End_Year']
for i in range(len(cols)):
    plt.subplot(1,2,i+1)
    plt.hist(one_way[cols[i]], bins = len(one_way[cols[i]].unique()))
    plt.title(cols[i])
    plt.grid()

plt.tight_layout()
plt.show()

Annual_Pass = data[data['passholder_type'] == 'Annual Pass']
plt.figure(figsize=(12,6))
cols = ['Start_Year','End_Year']
for i in range(len(cols)):
    plt.subplot(1,2,i+1)
    plt.hist(Annual_Pass[cols[i]], bins = len(Annual_Pass[cols[i]].unique()))
    plt.title(cols[i])
    plt.grid()

plt.tight_layout()
plt.show()

walk_up = data[data['passholder_type'] == 'Walk-up']
Monthly_Pass = data[data['passholder_type'] == 'Monthly Pass']
One_Day_Pass = data[data['passholder_type'] == 'One Day Pass']
Flex_Pass = data[data['passholder_type'] == 'Flex Pass']
Testing = data[data['passholder_type'] == 'Testing']

cols = ['Start_Day','Start_Month','Start_Year','Start_hour','Start_weekday','Start_week_of_year']
plt.figure(figsize=(30,15))
for i in range(len(cols)):
    plt.subplot(3,3,i+1)
    plt.hist(walk_up[cols[i]], bins = len(walk_up[cols[i]].unique()), alpha=0.5, label="walk_up")
    plt.hist(Monthly_Pass[cols[i]], bins = len(Monthly_Pass[cols[i]].unique()), alpha=0.5, label="Monthly_Pass")
    plt.hist(One_Day_Pass[cols[i]], bins = len(One_Day_Pass[cols[i]].unique()), alpha=0.5, label="One_Day_Pass")
    plt.hist(Flex_Pass[cols[i]], bins = len(Flex_Pass[cols[i]].unique()), alpha=0.5, label="Flex_Pass")
    plt.hist(Testing[cols[i]], bins = len(Testing[cols[i]].unique()), alpha=0.5, label="Testing")
    
    plt.title(cols[i])
    plt.legend(loc='upper right')
    plt.grid()

plt.tight_layout()
plt.show()


### ============================================
#Tarea 2
#convert cat to num
data['trip_route_category'] = pd.factorize(data['trip_route_category'])[0]
data['bike_id'] = pd.factorize(data['bike_id'])[0]
data['passholder_type'].replace(targets, targets_num, inplace=True)

#create matrix of data
X = np.array(data.iloc[:,0:data.shape[1]-1])
Y = np.array(data.iloc[:,data.shape[1]-1])
#Y_2c
#Y = np.array(data_2c.iloc[:,data.shape[1]-1])

#===== applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
pca.score(X)

#colores = ['red','blue']
colores=['red','green','blue','cyan','yellow','black']
asignar = []
for row in Y:
    asignar.append(colores[row])

fig = plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=asignar, s=2)
plt.show()

#===== appplying feature selection
nf = 5
test = SelectKBest(score_func=f_classif, k=nf)
fit = test.fit(X, Y)
features = fit.transform(X)
X = features
#targets selected
cols = test.get_support(indices=True)
feature_selected = data.columns[cols]


#========================
#con k fold estratificado
kf = StratifiedKFold(n_splits=5, shuffle=True)
predictions = []
true_test_targets = []
fold_Acu = []
fold_BA = []
fold_f1score = []

for train_index, test_index in tqdm(kf.split(X, Y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    true_test_targets.append(y_test)
        
    #create the model #xgb
    model = xgb.XGBClassifier()
    
    #learning phase
    model.fit(X_train, y_train)
        
    #classification phase
    prediction = model.predict(X_test)
    predictions.append(prediction)
    #BA and f1 score
    fold_Acu.append(metrics.accuracy_score(y_test, prediction))
    fold_BA.append(balanced_accuracy_score(y_test, prediction))
    fold_f1score.append(metrics.f1_score(y_test,prediction,average='macro'))
    
accuracy = round(sum(fold_Acu) / len(fold_Acu), 4)
BA_model = round(sum(fold_BA) / len(fold_BA), 4)
f1_score_model = round(sum(fold_f1score) / len(fold_f1score), 4)
print("N.Features: ", nf)
print("Accuracy: ", accuracy)
print("BA: ", BA_model)
print("F1 Score: ", f1_score_model)
#==== RESULTS
# N.Features:  10
# Accuracy:  0.7111
# BA:  0.2959
# F1 Score:  0.3165

#feature selected


#=======
#load and clean the test data to predictions
dataTest_original = pd.read_csv('test_set.csv')
dataTest = dataTest_original.copy()

#1. convert cat to num
dataTest['trip_route_category'] = pd.factorize(dataTest['trip_route_category'])[0]
dataTest['bike_id'] = pd.factorize(dataTest['bike_id'])[0]
#1.2 extract feature from datetime
start_time_test = pd.to_datetime(dataTest.loc[:,'start_time']).to_frame()
end_time_test = pd.to_datetime(dataTest.loc[:,'end_time']).to_frame()
#START_TIME
# day
start_time_test['Start_Day'] = start_time_test['start_time'].dt.day
# month
start_time_test['Start_Month'] = start_time_test['start_time'].dt.month
# year
start_time_test['Start_Year'] = start_time_test['start_time'].dt.year
# hour
start_time_test['Start_hour'] = start_time_test['start_time'].dt.hour
# minute
start_time_test['Start_minute'] = start_time_test['start_time'].dt.minute
# Monday is 0 and Sunday is 6
start_time_test['Start_weekday'] = start_time_test['start_time'].dt.weekday
# week of the year
start_time_test['Start_week_of_year'] = start_time_test['start_time'].dt.week
#END_TIME
# day
end_time_test['End_Day'] = end_time_test['end_time'].dt.day
# month
end_time_test['End_Month'] = end_time_test['end_time'].dt.month
# year
end_time_test['End_Year'] = end_time_test['end_time'].dt.year
# hour
end_time_test['End_hour'] = end_time_test['end_time'].dt.hour
# minute
end_time_test['End_minute'] = end_time_test['end_time'].dt.minute
# Monday is 0 and Sunday is 6
end_time_test['End_weekday'] = end_time_test['end_time'].dt.weekday
# week of the year
end_time_test['End_week_of_year'] = end_time_test['end_time'].dt.week
#join new features
dataTest = pd.concat([dataTest, start_time_test.iloc[:,1:], end_time_test.iloc[:,1:]], axis=1)

#2. set the features selected
XTest = np.array(dataTest[feature_selected])

#3. train the complete train_test with the best model
model = xgb.XGBClassifier()
model.fit(X, Y)

#4. predict the test_set
YTest = pd.DataFrame(model.predict(XTest), columns=['passholder_type'])
prediction = dataTest['trip_id'].to_frame()

#assign targets to predictions
targets_num = data["passholder_type"].unique()

YTest['passholder_type'].replace(targets_num, targets, inplace=True)
#2 classes
#YTest['passholder_type'].replace([0, 1], ['NO_Monthly Pass', 'Monthly Pass'], inplace=True)

prediction.insert(1,'passholder_type', YTest)

prediction.to_csv(r'submission.csv', index=False)





