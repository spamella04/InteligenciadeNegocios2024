
#Hecho por Megan Bougle
#Dataset de Titanic recuperado de https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

#Se carga el data set desde un repositorio de GitHub 
url='https://raw.githubusercontent.com/meganbougle/TitanicDataSet/main/titanic.csv'
df= pd.read_csv(url)
#Se verifica que  se haya cargado correctamente mediante un print de las primeras columnas del data set
print(df.head())

#Tamanio del dataset y valores estadisticos de cada columna
print(df.shape)
print(df.describe())


print(df.info())


#Se revisa si hay valores faltantes en las columnas
#Vemos que no existe ninguna asi que no hay necesidad de llenar valores faltantes
df.isnull().sum()


#Estadisticas descriptivas para las columnas numericas: edad y tarifa del pasaje
media=df[['Age','Fare']].mean()
mediana=df[['Age','Fare']].median()
#el iloc es para que solo se extraiga la primera moda porque pueden haber mas de una aveces
moda=df[['Age','Fare']].mode().iloc[0]
std=df[['Age','Fare']].std()
EstadisticasDesc=pd.DataFrame({'Media':media,'Mediana':mediana,'Moda':moda,'Desviacion Estandar':std})
print(EstadisticasDesc)


contarSob=df['Survived'].value_counts()
tasaSup=df['Survived'].mean()
plt.figure(figsize=(10,5))
plt.pie(contarSob.values, labels=contarSob.index, autopct='%1.1f%%', startangle=90)
plt.title('Porcentaje de sobrevivientes')
plt.show()


#He notado que la columna edad tiene varios valores (es normal) pero, para facilitar el analisis decidi agruparlas en rangos de edades
bins = [0, 18, 35, 50, 80]
labels = ['0-18', '19-35', '36-50', '50+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Grafico de barras de la Edad utilizandi el grupo de edad previamente creado
figsize=(14, 5)
sns.countplot(x='AgeGroup', data=df)
plt.title('Distribución de Edad')
plt.xlabel('Grupo de Edad')
plt.tight_layout()
plt.show()

df['sobrevivientesLb'] = df['Survived'].replace({0: 'Muertos', 1: 'Sobrevivientes'})
contarSv=sns.countplot(x='sobrevivientesLb', data=df, color='pink')
for container in contarSv.containers:
    contarSv.bar_label(container)
plt.title('Cuántos sobrevivieron?')
contarSv.set_xlabel('Sobrevivientes')
plt.show()

#tasas de supervivencia por sexo, grupo de edad y clase. Para esto se calcula la mdia de la columna correspondiente y posteriormente se realiza un grafico de pastel para lograr visualizar los resultados.
porcentSexo=df.groupby('Sex')['Survived'].mean()*100
figsize=(14, 5)
plt.pie(porcentSexo.values, labels=porcentSexo.index, autopct='%1.1f%%', startangle=90, colors=['pink','lightblue'])
plt.title('Tasa de superviviencia por sexo')
plt.show()
porcentEdad=df.groupby('AgeGroup')['Survived'].mean()*100
plt.pie(porcentEdad.values, labels=porcentEdad.index, autopct='%1.1f%%', startangle=90, colors=['pink','lightblue','lightgreen','yellow'])
plt.title('Tasa de superviviencia por grupo de edad')
plt.show()
porcentClase=df.groupby('Pclass')['Survived'].mean()*100
plt.pie(porcentClase.values, labels=porcentClase.index, autopct='%1.1f%%', startangle=90, colors=['red','lightblue','orange'])
plt.title('Tasa de superviviencia por clase')
plt.show()

#Distribuciones para ver la cantidad de sobrevivientes y muertos por genero  y clase de pasaje, esto permite tener una mejor comprension de los datos
contarSexo=sns.countplot(x='Sex', hue='sobrevivientesLb', data=df)
plt.title('Distribución de Supervivencia por Género')
plt.xlabel('Género')
plt.ylabel('Cantidad de Pasajeros')
plt.legend(title='Supervivencia')
for container in contarSexo.containers:
    contarSexo.bar_label(container)
plt.show()

contarClase=sns.countplot(x='Pclass', hue='sobrevivientesLb', data=df)
plt.title('Distribución de Supervivencia por Clase de Pasaje')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Pasajeros')
plt.legend(title='Supervivencia')
for container in contarClase.containers:
    contarClase.bar_label(container)
plt.show()


sns.boxplot(x='sobrevivientesLb', y='Age', data=df)
plt.title('Distribuicion de la Edad y Supervivencia')
plt.xlabel('Supervivencia')
plt.ylabel('Edad')
plt.show()
#En este grafico de cajas se puede observar que la edad de los sobrevivientes es menor a la de los muertos y hay algunos outliers en ambos casos

#Intentado entender la relacion entre la edad y la tariafa del pasaje
sns.scatterplot(x='Age', y='Fare', hue='sobrevivientesLb', data=df)
plt.title('Edad vs. Tarifa')
plt.xlabel('Edad')
plt.ylabel('Tarifa')
plt.legend(title='Supervivencia')
plt.show()

#Decidi agrupar todas las columnas numericas relevantes para lograr analizar la relacion entre ellas mediante una matriz de correlacion
vnum=df[['Age','Fare','Pclass','Survived']]
correlation_matrix = vnum.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()
