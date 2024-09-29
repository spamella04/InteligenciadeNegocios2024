import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset 
url = 'https://raw.githubusercontent.com/cdtoruno/titanic/refs/heads/main/titanic.csv'
data = pd.read_csv(url)

# Exploracion inicial de los datos
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Estadistica basica de columnas numericas
print('\nEstadistica basica de columnas numericas')
print(data['Age'].describe())
print(data['Fare'].describe())

# Cantidad de hombres y mujeres
hombres = (data['Sex'] == 'male').sum()
mujeres = (data['Sex'] == 'female').sum()
print('\nCantidad de hombres y mujeres')
print('Hombres:', hombres)
print('Mujeres:', mujeres)

# Cantidad de pasajeros por boleto
primera_clase = (data['Pclass'] == 1).sum()
segunda_clase = (data['Pclass'] == 2).sum()
tercera_clase = (data['Pclass'] == 3).sum()
print('\nCantidad de pasajeros por boleto')
print('Primera Clase:', primera_clase)
print('Segunda Clase:', segunda_clase)
print('Tercera Clase:', tercera_clase)


# Distribucion de edades
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=20, color='green')
plt.title('Distribucion de las edades')
plt.xlabel('Edad')
plt.ylabel('Número de Pasajeros')
plt.show()

# Sobrevivientes por sexo
plt.figure(figsize=(8, 8))
survived_by_sex = data[data['Survived'] == 1]['Sex'].value_counts()
plt.pie(survived_by_sex, labels=survived_by_sex.index, autopct='%1.1f%%', startangle=90, colors=['pink','blue'])
plt.title('Sobrevivientes por sexo')
plt.axis('equal')
plt.show()

# Clase de los pasajes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=data)
plt.title('Tarifa para cada boleto')
plt.xlabel('Clase de Boleto')
plt.ylabel('Tarifa')
plt.show()

# Sobrevivientes por clase
plt.figure(figsize=(8, 8))
survived_by_class = data[data['Survived'] == 1]['Pclass'].value_counts()
plt.pie(survived_by_class, labels=survived_by_class.index, autopct='%1.1f%%', startangle=90, colors=['orange','lightblue','lightgreen'])
plt.title('Sobrevivientes por clase de boleto')
plt.axis('equal')
plt.show()

# Analisis de tasa de supervivencia por sexo
porcentaje_genero = data.groupby('Sex')['Survived'].mean() * 100
print(porcentaje_genero)

# Grafico de tasa por sexo
plt.figure(figsize=(8, 8))
sns.barplot(x=porcentaje_genero.index, y=porcentaje_genero.values)
plt.title('Tasa de supervivencia por sexo')
plt.ylabel('Porcentaje')
plt.show()

# Analisis de tasa de supervivencia por clase
porcentaje_clase = data.groupby('Pclass')['Survived'].mean() * 100
print(porcentaje_clase)

# Grafico de tasa por clase
plt.figure(figsize=(8, 8))
sns.barplot(x=porcentaje_clase.index, y=porcentaje_clase.values)
plt.title('Tasa de supervivencia por clase de boleto')
plt.ylabel('Porcentaje')
plt.show()

# Supervivencia por grupos de edad
# print('\nUniques')
# print(data['Age'].unique())
data['GrupoEdad'] = pd.cut(data['Age'], bins=[0, 12, 18, 65, 100], labels=['Niño', 'Adolescente', 'Adulto', 'Anciano'])
porcentaje_edad = data.groupby('GrupoEdad',observed=False)['Survived'].mean() * 100
print(porcentaje_edad)

# Grafico de tasa por edad
plt.figure(figsize=(8, 8))
sns.barplot(x=porcentaje_edad.index, y=porcentaje_edad.values)
plt.title('Tasa de supervivencia por edad')
plt.ylabel('Porcentaje')
plt.show()

