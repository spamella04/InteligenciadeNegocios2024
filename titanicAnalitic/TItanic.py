# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INSURANCE_CARS = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'

tb = pd.read_csv(INSURANCE_CARS)

print(tb.head)

# %%
#exploracion inicial
print("exploracion inicial ")

print(f"informacion")
print(tb.info())

print(f"descripcion")
print(tb.describe())



# %%
#evaluar valores faltantes

valores_Faltantes = tb.isnull().sum()

print(valores_Faltantes[valores_Faltantes > 0])

# %%
age_by_class = tb.groupby('Pclass')['Age'].mean()

age_by_class.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Edad Promedio por Clase')
plt.xlabel('Pclass')
plt.ylabel('Edad Promedio')
plt.grid(axis='y')
plt.show()

# %%

survived_by_sex = tb.groupby(['Sex', 'Survived']).size().unstack()


plt.subplot(1, 2, 1) 
plt.pie(survived_by_sex.loc['female'], labels=['No Sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', colors=['salmon', 'lightgreen'], startangle=90)
plt.title('Supervivencia Mujeres')

plt.subplot(1, 2, 2)
plt.pie(survived_by_sex.loc['male'], labels=['No Sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', colors=['salmon', 'lightgreen'], startangle=90)
plt.title('Supervivencia Hombres')

plt.tight_layout()
plt.show()

# %%
survived_by_class = tb.groupby(['Pclass', 'Survived']).size().unstack()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.pie(survived_by_class.loc[1], labels=['No Sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', colors=['salmon', 'lightgreen'], startangle=90)
plt.title('Supervivencia Clase 1')
plt.subplot(1, 3, 2)
plt.pie(survived_by_class.loc[2], labels=['No Sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', colors=['salmon', 'lightgreen'], startangle=90)
plt.title('Supervivencia Clase 2')
plt.subplot(1, 3, 3)
plt.pie(survived_by_class.loc[3], labels=['No Sobrevivió', 'Sobrevivió'], autopct='%1.1f%%', colors=['salmon', 'lightgreen'], startangle=90)
plt.title('Supervivencia Clase 3')

plt.tight_layout()
plt.show()

# %%

bins = [0, 12, 18, 35, 60, 100]
labels = ['Niños', 'Jóvenes', 'Adultos', 'Mayores', 'Ancianos']
tb['EtapaEdad'] = pd.cut(tb['Age'], bins=bins, labels=labels)

survived_by_age = tb.groupby(['EtapaEdad', 'Survived']).size().unstack()
survived_by_age.plot(kind='bar', color=['salmon', 'lightgreen'], edgecolor='black')

plt.title('Supervivencia por Grupo de Edad')
plt.xlabel('Grupo de Edad')
plt.ylabel('Cantidad de Personas')
plt.legend(['No Sobrevivió', 'Sobrevivió'])
plt.grid(axis='y')
plt.tight_layout()
plt.show()


