import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

def inference():
	main_df=pd.read_csv('train.csv')
	main_df=main_df.fillna('no')
	main_df.head()
	files = list()

	for i in range(1,19):
	    exp_number = '0' + str(i) if i < 10 else str(i)
	    file = pd.read_csv("experiment_{}.csv".format(exp_number))
	    row = main_df[main_df['No'] == i]
	    
	     #add experiment settings to features
	    file['feedrate']=row.iloc[0]['feedrate']
	    file['clamp_pressure']=row.iloc[0]['clamp_pressure']
	    
	    # Having label as 'tool_conidtion'
	    
	    file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
	    files.append(file)
	df = pd.concat(files, ignore_index = True)
	df.head()
	pro={'Layer 1 Up':1,'Repositioning':2,'Layer 2 Up':3,'Layer 2 Up':4,'Layer 1 Down':5,'End':6,'Layer 2 Down':7,'Layer 3 Down':8,'Prep':9,'end':10,'Starting':11}

	data=[df]

	for dataset in data:
	    dataset['Machining_Process']=dataset['Machining_Process'].map(pro)
	    df=df.drop(['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia'],axis=1)
	    
	X=df.drop(['label','Machining_Process'],axis=1)
	Y=df['label']
	print('The dimension of X table is: ',X.shape,'\n')
	print('The dimension of Y table is: ', Y.shape)
	from sklearn.model_selection import train_test_split

	#divided into testing and training
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
	sgd_model=SGDClassifier()
	sgd_model.fit(x_train,y_train)
	sgd_model_pred=sgd_model.predict(x_test)
	acc_sgd_model=round(sgd_model.score(x_train, y_train)*100,2)
	rmf_model=RandomForestClassifier()
	rmf_model.fit(x_train,y_train)
	rmf_model_pred=rmf_model.predict(x_test)
	acc_rmf_model=round(rmf_model.score(x_train, y_train)*100,2)
	from sklearn.model_selection import cross_val_score
	rmf_model = RandomForestClassifier(n_estimators=100)
	scores = cross_val_score(rmf_model, x_train, y_train, cv=10, scoring = "accuracy")
	print("Scores:", scores,'\n')
	print("Mean:", scores.mean(),'\n')
	print("Standard Deviation:", scores.std())
	rmf_model = RandomForestClassifier(n_estimators=100, oob_score = True)
	rmf_model.fit(x_train, y_train)
	y_prediction = rmf_model.predict(x_test)

	rmf_model.score(x_train, y_train)

	acc_rmf_model = round(rmf_model.score(x_train, y_train) * 100, 2)
	print(round(acc_rmf_model,2,), "%")
	print("oob score:", round(rmf_model.oob_score_, 4)*100, "%")




