#Import Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
PassengerId = test['PassengerId']
PassengerId.shape
train.head(5)
#Data Preprocessing and EDA**
train.shape
train.columns
train.describe()
train.info()
train['Survived'].value_counts()
(train["Survived"].value_counts(normalize=True)).plot(kind='bar')
plt.bar((train['Fare'].value_counts()).index, (train['Fare'].value_counts()).values,width=0.9)

sns.countplot(data=train,x='Sex',hue='Survived')

sns.catplot(x="Embarked", hue="Survived", data=train, kind="count", height=4, aspect=0.9)
train.groupby(['Pclass','Sex'])['Survived'].value_counts()
train['Age'].agg(['min', 'max','mean'])
train[train['Age']==train['Age'].min()]['Survived']
train["Fare"].max()
train[train["Fare"]==train['Fare'].max()]['Survived']
train[train["Fare"]==train['Fare'].max()]['Cabin']
sns.displot(data=train,x='Age',col="Sex",hue='Survived')
train['Family']=train['SibSp']+train['Parch']+1

plt.figure(figsize=(10,5))
plt.title("The Count of families on the ship");
sns.countplot(data=train,x='Family');

sns.countplot(data=train,x="Family",hue='Survived');
sns.displot(data=train,x='Family',col="Survived",hue='Pclass')
sns.catplot(x="Family", hue="Survived", col="Pclass",
                data=train, kind="count", height=4, aspect=0.9)
train[(train['Family']==1)& train['Survived']==1]['Sex'].value_counts(normalize=True)
train.isna().sum()
test.info()

X, y = train.drop('Survived', axis=1), train['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
for data in[X_train,X_test,test]:
    title_second_part=data['Name'].str.split(',').str[1]
    title_only=title_second_part.str.split('.').str[0]
    data['title']=title_only.str.strip()
    data['title'].value_counts()
    
X_train['title'].unique()
def age_fill(data):
    average_age_by_title = data.groupby('title')['Age'].mean()
    return data.apply(lambda row: average_age_by_title[row['title']] if pd.isna(row['Age']) else row['Age'], axis=1)

# Apply the age_fill function to fill missing ages based on titles
X_train['Age'] = age_fill(X_train)
X_test['Age'] = age_fill(X_test)
test['Age']=age_fill(test)
test['Age'].fillna(test['Age'].mean(),inplace=True)
average_age_by_title = X_train.groupby('title')['Age'].mean()

# Create a bar plot
plt.figure(figsize=(10, 6))
average_age_by_title.plot(kind='bar', color='skyblue')
for data in[X_train,X_test,test]:
    counts = data['title'].value_counts()

    # Update the DataFrame to replace counts with 'others'
    counts['others'] = counts[4:].sum()

    # Replace values directly in the 'title' column using loc
    data.loc[data['title'].isin(counts.index[4:]), 'title'] = 'others'
    data['title'].value_counts()
test.info()
for data in [X_train, X_test,test]:
    
    embarked_mode=data['Embarked'].mode()[0]
    data['Embarked'].fillna(embarked_mode,inplace=True)
    
## Remove Irrelevant Features 

for data in [X_train, X_test]:
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Family','title'],axis=1,inplace=True)
test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','title'],axis=1,inplace=True)
test.info()
X_train.info()
X_train.head(5)

for value in [X_train, X_test,test]:
    value['Sex'].replace({'female':1,'male':0},inplace=True)
    value['Embarked'].replace({'S':0, 'C':1, 'Q':2},inplace=True)
X_train.info()
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test.isna().sum()
X_train.info()
logistic_model = LogisticRegression(max_iter=1200, random_state=42)
logistic_model.fit(X_train, y_train)

y_pred=logistic_model.predict(X_test)


pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['Predict No','Predict Yes'],index=['Actual No','Actual Yes'])

print('Train accuracy is = ',accuracy_score(y_train,logistic_model.predict(X_train)))

Using SVC (non-linear)
svc_classifier=SVC(kernel='rbf',C=10, probability=True)
svc_classifier.fit(X_train,y_train)
print('Train accuracy:', accuracy_score(y_train, svc_classifier.predict(X_train)))

rndm_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
rndm_clf.fit(X_train,y_train)
print('Train accuracy by using RandomForest:', accuracy_score(y_train, rndm_clf.predict(X_train)))

X_test.info()


voting_clf = VotingClassifier(
    estimators=[('lr', logistic_model), ('rf', rndm_clf), ('svc', svc_classifier)],
    voting='hard')
voting_clf.fit(X_train, y_train)
print('Finding the accuracy ')
for clf in (logistic_model, rndm_clf, svc_classifier, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(" ",clf.__class__.__name__, accuracy_score(y_test, y_pred))
print('\n =======================')
voting_clf = VotingClassifier(
    estimators=[('lr', logistic_model), ('rf', rndm_clf), ('svc', svc_classifier)],
    voting='soft')
voting_clf.fit(X_train, y_train)

for clf in (logistic_model, rndm_clf, svc_classifier, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    

accuracy_score(y_test, rndm_clf.predict(X_test))
test.isna().sum()
Prediction = rndm_clf.predict(test)


Submission = pd.DataFrame({ 'PassengerId': PassengerId,'Survived': Prediction })

Submission.to_csv('Submission_file.csv',index=False)