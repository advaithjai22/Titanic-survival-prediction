#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
#importing the datasets
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
#inspecting the datasets
train.head()
train.info()
train.describe()
train.describe(include=['O'])
test.head()
test.info()
test.describe()
train.describe(include =['O'])
#using pivot tables
pd.pivot_table(train,index = 'Pclass',values = 'Survived')
pd.pivot_table(train,index = 'Sex',values = 'Survived')
pd.pivot_table(train,index = 'SibSp',values = 'Survived')
pd.pivot_table(train,index = 'Parch',values = 'Survived')
#visualizing the relation between age and survival
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#visualizing the relation between Pclass,age and sruvived 
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#visualizing the relation between embarked,Pclass,survived and sex 
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
#dropping unnecessary columns
train=train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test=test.drop(['Name','Ticket','Cabin'],axis=1)
#fillng in the missing values for the data
train.fillna(method='ffill',axis=0,inplace=True)
test.fillna(method='ffill',axis=0,inplace=True)
#creating new features
for n in train:
    train['FamilySize']=1+train['SibSp']+train['Parch']
for n in test:
    test['FamilySize']=1+test['SibSp']+test['Parch']
train=train.drop(['SibSp','Parch'],axis=1)
test=test.drop(['SibSp','Parch'],axis=1)    
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for n in train:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
for n in test:
    test['IsAlone'] = 0
    test.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
#encoding the categorical variables     
from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
train['Sex'] = lb.fit_transform(train['Sex'])
test['Sex'] = lb.fit_transform(test['Sex'])
train['Embarked'] = lb.fit_transform(train['Embarked'])
test['Embarked'] = lb.fit_transform(test['Embarked'])
#splitting the datasets
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
#SVC Classifier
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
#Decision Trees
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
#Random forest regressor
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
#writing the predictions to a file
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)



    


