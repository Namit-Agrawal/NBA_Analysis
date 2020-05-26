import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

#Read in the college data and rookie data
rookie_data = pd.read_csv('rookie_records.csv')
ncaa_data = pd.read_csv('toppick_records.csv')
ncaa_2019_data = pd.read_csv('toppick2019_records.csv')
#list to get index of players that will be dropped
list = []
#list that specifies the stats to predict
list2= ['PTS', 'AST', 'TRB', 'STL', 'BLK']
#list needed for feature selection
list3 = ['GP','MIN','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT','OFF','DEF','TRB','AST','STL','BLK','PF','TOV','PTS','PTS2']
nba_pred = []
final_pred = []
rookie_stats=np.array(rookie_data)

#Filter out all the players who played fewer than 20 NBA games during rookie
#seasons. Players who played fewer than 20 games might disrupt the training
#model
i = 0
while i<len(rookie_data):
    if rookie_stats[i][1]<20:
        list.append(i)
    i+=1
val = 0
#drop from both college and rookie datasets
for a in range(0, len(list)):
    rookie_data = rookie_data.drop(rookie_data.index[list[a]-val])
    ncaa_data = ncaa_data.drop(ncaa_data.index[list[a]-val])
    val+=1


#Drop the features that won't be needed to predict stats and convert to mumpy array
nba_pred.append(np.array(ncaa_2019_data['player name']))
ncaa_data=ncaa_data.drop(columns=['player name','year','pick','position','GS','WS','eFG%','PER'])
ncaa_2019_data = ncaa_2019_data.drop(columns=['player name','year','pick','position','GS','WS','eFG%','PER'])
cols = ncaa_data.columns
#Normalize the features
sc = StandardScaler()
for co in cols:
    ncaa_data[[co]] = sc.fit_transform(ncaa_data[[co]])
    ncaa_2019_data[[co]] = sc.fit_transform(ncaa_2019_data[[co]])

#This code was used to visualize the features and how they correlate to one
#another
'''
pointss = np.array(rookie_data[list2[0]], dtype='float32')
ncaa_data['PTS2'] = pointss
#print(ncaa_data)
correlation = ncaa_data[list3].corr()
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
map = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlation, mask=mask, cmap=map, vmax=.3, center=0, square=True, linewidths=.01, cbar_kws={"shrink": .5})
plt.show()
'''
#Loop to predict the NBA stats for players
for a in range(0, len(list2)):
    #Normalize the labels
    sc6 = StandardScaler()
    rookie_data[[list2[a]]] = sc6.fit_transform(rookie_data[[list2[a]]])
    ncaa_stats=np.array(ncaa_data, dtype='float32')
    points = np.array(rookie_data[list2[a]], dtype='float32')
    #Use RFE function for feature selection to use the best data to predict
    #stats. Used LinearRegression to train
    model = LinearRegression()
    rfe = RFE(model, 12)
    reg = rfe.fit(ncaa_stats, points)
    cols_to_use = reg.support_
    cols_to_use_2 = []
    cols_to_use_3=[]
    cols_2019 = []
    cols_2019_2 = []
    num = 0
    ncaa_2019 = np.array(ncaa_2019_data, dtype='float32')
    #Transform the data into a readable format
    for val in cols_to_use:
        if val:
            cols_to_use_2.append(ncaa_data[list3[num]].values)
            cols_2019.append(ncaa_2019_data[list3[num]].values)
        num+=1
    for col in range(0, len(cols_to_use_2[0])):
        for row in range(0, len(cols_to_use_2)):
            cols_to_use_3.append(cols_to_use_2[row][col])

    for col in range(0, len(cols_2019[0])):
        for row in range(0, len(cols_2019)):
            cols_2019_2.append(cols_2019[row][col])
    #Convert to numpy array
    pointss = np.array(rookie_data[list2[a]], dtype='float32')
    dummy_array = np.array(cols_to_use_3)
    cols_to_use_3 = np.reshape(dummy_array, (-1, 12))
    dummy_array2 = np.array(cols_2019_2)
    cols_2019_3 = np.reshape(dummy_array2, (-1, 12))
    #line used to split the data and determine what training algorithm to use
    #train_features, test_features, train_labels, test_labels = train_test_split(cols_to_use_3, pointss, test_size = 0.2, random_state = 0)
    #Eventually used Random Forest Algorithm as it produced the best results
    #based on mean square error and r2 score
    rf = RandomForestRegressor(n_estimators=1000, random_state=0)
    rf.fit(cols_to_use_3, pointss);
    #inverse transform to get the right data
    predictions = sc6.inverse_transform(rf.predict(cols_2019_3))
    nba_pred.append(predictions)

#Neatly gather all the information and reshape the data into a readable form
for col in range(0, len(nba_pred[0])):
    for row in range(0, len(nba_pred)):
        final_pred.append(nba_pred[row][col])

dummy_array = np.array(final_pred)
final_pred = np.reshape(dummy_array, (-1, 6))



#Export the predictions to a csv file
df = pd.DataFrame(final_pred, columns=['player name']+list2)
df.set_index('player name', inplace=True)
df.to_csv('predictions.csv')
