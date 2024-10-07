# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %%
url = 'https://raw.githubusercontent.com/drakenlander/TitanicCsv/refs/heads/main/titanic.csv'
df = pd.read_csv(url)

# %%
print(df.head())

# %%
print(df.select_dtypes('number').describe())

# %%
labels = ['Didn\'t Survive', 'Survived']

ax = df['Survived'].value_counts().plot.bar(y='Survived', rot=0)
ax.set_xlabel('Survivability')
ax.set_xticklabels(labels)
plt.show()

# %%
sns.boxplot(x = df['Age'])
plt.show()

# %%
conditions = [
    (df['Sex'] == 'male'),
    (df['Sex'] == 'female')
    ]

values = ['0', '1']

df['Sex'] = np.select(conditions, values, default=pd.NaT)

# %%
labels = ['Male', 'Female']

ax = df['Sex'].value_counts().plot.bar(y='Sex', rot=0)
ax.set_xticklabels(labels)
plt.show()

# %%
sns.boxplot(x = df['Fare'])
plt.show()

# %%
plt.scatter(df['Age'], df['Fare'], alpha = 0.5)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# %%
ax = sns.boxplot(x = df['Pclass'], y = df['Fare'])
ax.set_xlabel('Passenger Class')
plt.show()

# %%
labels = ['Didn\'t Survive', 'Survived']

ax = sns.boxplot(x = df['Survived'], y = df['Age'])
ax.set_xlabel('Survivability')
ax.set_xticklabels(labels)
plt.show()

# %%
df_surv = df.loc[df['Survived'] == 1]
df_dsurv = df.loc[df['Survived'] == 0]

labels = ['Male', 'Female']

ax = df_surv['Sex'].value_counts().plot.bar(y='Sex', rot=0)
ax.set_xlabel('Surviving Passengers')
ax.set_xticklabels(labels)
plt.show()

# %%
ax = df_dsurv['Sex'].value_counts().plot.bar(y='Sex', rot=0)
ax.set_xlabel('Casualties')
ax.set_xticklabels(labels)
plt.show()

# %%
labels = ['Didn\'t Survive', 'Survived']

ax = sns.boxplot(x = df['Survived'], y = df['Fare'])
ax.set_xlabel('Survivability')
ax.set_xticklabels(labels)
plt.show()

# %%
df_third = df.loc[df['Pclass'] == 3]
df_second = df.loc[df['Pclass'] == 2]
df_first = df.loc[df['Pclass'] == 1]

arr_class = [
    df_third['Survived'].value_counts(),
    df_second['Survived'].value_counts(),
    df_first['Survived'].value_counts()
    ]

cnt = 3

labels = ['Didn\'t Survive', 'Survived']

for i in arr_class:
    ax = i.plot.bar(rot = 0)
    ax.set_xlabel('Class ' + str(cnt) + ' Survivability')
    ax.set_xticklabels(labels)
    plt.show()

    cnt = cnt - 1


