import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


train = pd.read_csv('dataset/titanic/train.csv')

print(train.head(10))
print(train.describe())

def process_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title

train['Title'] = train['Name'].map(lambda name: process_title(name))
print(train['Title'].unique())

# for each Sex, sum up how many survive
survive_sex_group_by = train.groupby(["Sex", "Survived"])
plot1 = survive_sex_group_by.size().unstack(fill_value=0).plot.bar(title="Sex vs Survived")
plot1.set(ylabel="Number of person")
plt.tight_layout()
plt.show()

survive_age_group_by = train.groupby([pd.cut(train["Age"], np.arange(0, 100, 10)), "Survived"])
plot2 = survive_age_group_by.size().unstack(fill_value=0).plot.bar(title="Age vs Survived")
plot2.set(ylabel="Number of person")
plt.tight_layout()
plt.show()





