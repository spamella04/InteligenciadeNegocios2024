import pandas as pd

#CARGAR DATOS
#Importar el dataset
data = pd.read_csv('athlete_events.csv')

#Imprimir las primeras 5 lineas del dataset
print(data.head())

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#EXPLORACION INICIAL
#Ver el tipo de dato de cada columna
print("")
print(data.dtypes)

#Ver descripción de las variables numéricas
print("")
print(data.describe())

#Ver la distribución de valores faltantes en el dataset
print("")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#LIMPIEZA DATOS

#Obtener todas las columnas de tipo object y guardalas en una variable
columnas_string = ['Name', 'Sex', 'Team', 'NOC', 'Games', 'Season', 'City', 'Sport', 'Event', 'Medal']

#Remover los espacios en blanco iniciales y finales de las columnas de tipo object usando strip
for i in columnas_string:
    data[i] = data[i].str.strip()
    
#Reemplazar los valores Nan de las columnas "Age", "Height" y "Weight" con "No registro"
col = ['Age', 'Height', 'Weight']
for columna in col:
    data[columna] = data[columna].fillna("No registro")

#Reemplazar los valores Nan de la columna "Medal" con "No obtuvo"
col2 = ['Medal']

for columna in col2:
    data[columna] = data[columna].fillna("No obtuvo")
    
#Remover la columna "NOC" porque no es necesaria
data=data.drop('NOC', axis=1)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#EXPORTAR DATASET

# Verificar que las columnas se encuentran con los nuevos cambios
print(data.head())

# Exportar el Dataset al cual se le hizo la limpieza a Excel
data.to_csv('Athlete_Events_Limpio.csv', index=False)
