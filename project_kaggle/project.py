# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 12:25:06 2017

@author: Amit
"""
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('C:/Users/Amit/Desktop/Kaggle/train.csv')
test = pd.read_csv('C:/Users/Amit/Desktop/Kaggle/test.csv')
combine = [train_df, test_df]

train["Survived"][train["Pclass"] == 3].value_counts(normalize=True)
train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)
train["Survived"][(train.Fare >=7.5) & (train.Fare<14)].value_counts(normalize=True)
train["Survived"][(train.Fare >=7.1) & (train.Fare<8)].value_counts()
train["Survived"][train["Sex"] == 'female'].mean()
#average prce of survived
train[['Fare', 'Survived']].groupby(['Survived']).median()

train[['Pclass', 'Survived']].groupby(['Pclass']).count()
train[['Pclass', 'Survived']].groupby(['Pclass']).mean()

train[['Pclass', 'Sex','Survived']].groupby(['Sex','Survived']).count()
train[['Pclass','Survived','Sex']].groupby(['Pclass','Survived']).count()

train[['Pclass','Survived','PassengerId','Sex']].groupby(['Pclass','Sex','Survived']).count()
train[['Pclass','Survived','Sex']].groupby(['Pclass','Sex']).mean()


train.tail()
train.info()
train.describe()
#Distribution of categorical features
train.describe(include=['O'])

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


g = sns.FacetGrid(train, col='Survived',row='Sex')
g.map(plt.hist, 'Age', bins=20)


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#  Wrangle data
train = train.drop(['Ticket', 'Cabin'], axis=1)
#test = test.drop(['Ticket', 'Cabin'], axis=1)
#combine = [train, test]


train = train.drop(['Name', 'PassengerId'], axis=1)
train = train.dropna()
df=train

# specifies the parameters of our graphs
fig = plt.figure(figsize=(18,6), dpi=1600) 
alpha=alpha_scatterplot = 0.2 
alpha_bar_chart = 0.55

# lets us plot many diffrent shaped graphs together 
ax1 = plt.subplot2grid((2,3),(0,0))
# plots a bar graph of those who surived vs those who did not.               
df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
ax1.set_xlim(-1, 2)
# puts a title on our graph
plt.title("Distribution of Survival, (1 = Survived)")    
plt.show()
plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survival by Age,  (1 = Survived)")

ax3 = plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
ax3.set_ylim(-1, len(df.Pclass.value_counts()))
plt.title("Class Distribution")

plt.subplot2grid((2,3),(1,0), colspan=2)
# plots a kernel density estimate of the subset of the 1st class passangers's age
df.Age[df.Pclass == 1].plot(kind='kde')    
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

ax5 = plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
ax5.set_xlim(-1, len(df.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")

plt.show()



# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1



train["Fare"][train.Fare <= 7.1] = 0
train["Fare"][(train.Fare <= 12) & (train.Fare >7.1)] = 1
train["Fare"][(train.Fare <= 31) & (train.Fare >12)] = 2
train["Fare"][train.Fare >= 31] = 3
    
    


y = train[["Survived"]]
X = train[["Pclass", "Sex", "Age", "Fare"]]



# Fit your first decision tree: my_tree_one
my_tree_one = DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_one = my_tree_one.fit(X,y)

print(my_tree_one.feature_importances_)
print(my_tree_one.score(X,y))

#random forest
random_forest = RandomForestClassifier(n_estimators=100,max_depth = 10, min_samples_split = 5)
random_forest.fit(X,y)
Y_pred = random_forest.predict(X)
random_forest.score(X,y)
acc_random_forest = round(random_forest.score(X, y) * 100, 2)
acc_random_forest

#test
test1 = test[["Pclass", "Sex", "Age", "Fare"]]
test1["Sex"][test1["Sex"] == "male"] = 0
test1["Sex"][test1["Sex"] == "female"] = 1
test1=test1.fillna(test1.Age.mean())

test1["Fare"][test1.Fare <= 7.1] = 0
test1["Fare"][(test1.Fare <= 12) & (test1.Fare >7.1)] = 1
test1["Fare"][(test1.Fare <= 31) & (test1.Fare >12)] = 2
test1["Fare"][test1.Fare >= 31] = 3
    
 

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test1)
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("C:/Users/Amit/Desktop/Kaggle/my_solution_six.csv", index_label = ["PassengerId"])

#random forest
my_prediction1 = random_forest.predict(test1)
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction1, PassengerId, columns = ["Survived"])
my_solution.to_csv("C:/Users/Amit/Desktop/Kaggle/my_solution_seven.csv", index_label = ["PassengerId"])

#knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)
Y_pred = knn.predict(test1)
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(Y_pred, PassengerId, columns = ["Survived"])
my_solution.to_csv("C:/Users/Amit/Desktop/Kaggle/my_solution_8.csv", index_label = ["PassengerId"])


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
  