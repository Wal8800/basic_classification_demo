import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def process_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    # white_list = ['Mr', 'Mrs', 'Miss', 'Master', 'Lady', 'Ms', 'Sir']

    return title

# reading train data
train = pd.read_csv('dataset/titanic/train.csv')

# reading test data
test = pd.read_csv('dataset/titanic/test.csv')

# extracting and then removing the targets from the training data
targets = train.Survived
train.drop('Survived', 1, inplace=True)

# merging train data and test data for future feature engineering
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop('index', inplace=True, axis=1)

# lets fill in the missing value for ages. a simple approach will be taking the median and put into the missing slot
combined['Age'].fillna(combined['Age'].median(), inplace=True)
combined['Fare'].fillna(combined['Fare'].median(), inplace=True)
combined["Embarked"].fillna("S", inplace=True)

# encode the non numerical value
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)

sex_dummies = pd.get_dummies(combined['Sex'], prefix='Sex')
combined = pd.concat([combined, sex_dummies], axis=1)
combined.drop('Sex', axis=1, inplace=True)

combined['Title'] = combined['Name'].map(lambda name: process_title(name))
title_dummies = pd.get_dummies(combined['Title'], prefix="Title")
combined = pd.concat([combined, title_dummies], axis=1)
combined.drop('Title', axis=1, inplace=True)

# dropping feature that aren't useful
combined = combined.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

print()
print("===resultant dataset===")
print(combined.head())

train0 = pd.read_csv('dataset/titanic/train.csv')

targets = train0.Survived
train = combined[0:891]
test = combined[891:]

# Search for the best parameters for this classifier with cross validation
clf = LogisticRegression()

parameter_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

kf = KFold(n_splits=6)

grid_search = GridSearchCV(clf,
                           param_grid=parameter_grid,
                           cv=kf)

grid_search.fit(train, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

# using the model to predict on the test set
output = grid_search.predict(test).astype(int)

test_df = pd.read_csv('dataset/titanic/test.csv')

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": output
    })

submission.to_csv('dataset/titanic/output.csv', index=False)







