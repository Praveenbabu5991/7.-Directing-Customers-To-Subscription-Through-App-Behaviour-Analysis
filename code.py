# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:35:43 2018

@author: Bolt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

dataset=pd.read_csv('appdata10.csv')

#EDA
dataset.head()
dataset.hour.head(3)
dataset.describe
#DATA CLEANING
dataset['hour']=dataset.hour.str.slice(1,3).astype(int)

dataset2=dataset.copy().drop(columns=['user','screen_list','enrolled_date','first_open','enrolled'])
dataset2.head()
dataset2.columns.values
dataset2.iloc[:,3].unique()
#histogram
## Histograms
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
#    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


#correlation with responce


dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reposnse variable',
                  fontsize = 15, rot = 45,
                  grid = True) #some are positively correlated

corr=dataset2.corr()

'''This module offers a generic date/time string parser which is able to parse
 most known formats to represent a date and/or time.
 The isinstance() function checks if the object (first argument)
 is an instance or subclass of classinfo class (second argument).
 
 '''
dataset.dtypes
dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
dataset.dtypes

dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')
response_hist = plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

#by putting range in between zero to one hundred
plt.hist(dataset["difference"].dropna(), color='#3F5D7D', range = [0, 100])
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

dataset.loc[dataset.difference > 48,'enrolled']=0 #setting enrolled 1 to 0 for difference greater than 48
dataset = dataset.drop(columns=['enrolled_date', 'difference', 'first_open'])

# Load Top Screens
top_screens = pd.read_csv('top_screens.csv').top_screens.values
top_screens
top_screens.head(5)
pd.set_option('display.max_columns', 500)

# Mapping Screens to Fields
dataset["screen_list"] = dataset.screen_list.astype(str) + ','
dataset.shape
dataset.head(3)

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)  #putting each word as a column
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",", "")#removing the name of column in screanlist

dataset['Other'] = dataset.screen_list.str.count(",") #counting the no of removing words in screan list
dataset = dataset.drop(columns=['screen_list']) #removing the column screen_list

# Funnels(means collecting a string into list)
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns=savings_screens)
'''
However, the numpy sum result is the exact opposite of what I was thinking.

'''
cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

#### Saving Results ####
dataset.head()
dataset.describe()
dataset.columns

dataset.to_csv('new_appdata11.csv', index = False)

#################################################
import time

dataset=pd.read_csv("new_appdata11.csv")

responce=dataset["enrolled"]
dataset=dataset.drop(columns="enrolled")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,responce,test_size=0.2,random_state=0)

train_identifier=X_train['user']
X_train=X_train.drop(columns='user')
test_identifier=X_test['user']
X_test=X_test.drop(columns='user')

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train2=pd.DataFrame(sc.fit_transform(X_train)) #if we didnot use data frame it will be matrix
X_test2=pd.DataFrame(sc.transform(X_test)) 
X_train2.columns=X_train2.columns.values
X_test2.columns=X_test2.columns.values
X_train2.index=X_train2.index.values
X_test2.index=X_test2.index.values
X_train=X_train2
X_test=X_test2

#model building
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,penalty='l1')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)


from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()

