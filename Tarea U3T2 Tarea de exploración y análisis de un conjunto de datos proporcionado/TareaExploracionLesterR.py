
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Load the data
url = 'https://media.githubusercontent.com/media/musanchez/DataSets/refs/heads/main/Pakistan%20Largest%20Ecommerce%20Dataset.csv'

df = pd.read_csv(url,low_memory=False)

# Eliminar las últimas 5 columnas (si están vacías o no son necesarias)
df = df.drop(df.columns[-5:], axis=1)


print(df.head())

print(df.describe())


# Analizar la cantidad de valores nulos y únicos
null_counts = df.isnull().sum()
unique_counts = df.nunique()

# Crear un dataframe de análisis
column_analysis = pd.DataFrame({
    'Null Count': null_counts,
    'Unique Count': unique_counts,
    'Data Type': df.dtypes
})

# Mostrar el análisis
print(column_analysis)


#seleccionamos las columnas con las que vamos a trabajar para el analisis
df_cleaned = df[['status','sku', 'price', 'category_name_1','qty_ordered', 'grand_total', 'discount_amount', 'BI Status', 'Year']].copy()


columns_to_impute = ['price', 'discount_amount']

def impute_mean(df_cleaned, columns_to_impute):
    for column in columns_to_impute:
        median_value = df_cleaned[column].mean()
        df_cleaned[column] = df_cleaned[column].fillna(median_value)
    return df_cleaned

# Aplicar la función para imputar valores nulos con la media
df_cleaned = impute_mean(df_cleaned, columns_to_impute)


# Condición para los registros donde 'grand_total' es nulo o 0
condition = (df_cleaned['grand_total'].isnull()) | (df_cleaned['grand_total'] == 0)

# Actualizar solo los registros que cumplen la condición
df_cleaned.loc[condition, 'grand_total'] = df_cleaned['price'] - df_cleaned['discount_amount']

# Visualizamos los datos limpios y seleccionados para el analisis
print(df_cleaned.head())

# Estadisticas descriptivas de las variables numericas
print(df_cleaned.describe())




# verificamos que no queden valores nulos en las columnas prince, discount_amount y grand_total
print(df_cleaned.isnull().sum())
print(' ')

columns_to_check = ['price', 'qty_ordered','discount_amount', 'grand_total']
negative_values = (df_cleaned[columns_to_check] < 0).sum()

print("Valores negativos por columna:")
print(negative_values)




# droppear los registros con valores negativos de la columna grand total y discount amount
df_cleaned = df_cleaned[df_cleaned['grand_total'] >= 0]
df_cleaned = df_cleaned[df_cleaned['discount_amount'] >= 0]

# Verificar que no queden valores negativos
negative_values = (df_cleaned[columns_to_check] < 0).sum()
print("Valores negativos por columna:")
print(negative_values)




#droppeamos los registros en los que todas las columnas son valores nulos
df_cleaned = df_cleaned[df_cleaned.isna().sum(axis=1) < 0.5 * len(df_cleaned.columns)]

# imputamos los valores nulos de la columna status con el valor mas frecuente

df_cleaned['status'] = df_cleaned['status'].fillna(df_cleaned['status'].mode()[0])
df_cleaned['category_name_1'] = df_cleaned['category_name_1'].fillna(df_cleaned['category_name_1'].mode()[0])

# Verificar que no queden valores nulos
print(df_cleaned.isnull().sum())


#graficamos los boxplot de las variables numericas para verificar si hay valores atipicos
def plot_boxplot(column):
    sns.boxplot(x=df_cleaned[column])
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

plot_boxplot('price')
plot_boxplot('qty_ordered')
plot_boxplot('discount_amount')
plot_boxplot('grand_total')



# Eliminar los registros con valores atípicos poniendo limite

df_cleaned = df_cleaned[(df_cleaned['price'] > 0) & (df_cleaned['price'] < 1780)]
df_cleaned = df_cleaned[(df_cleaned['qty_ordered'] > 0) & (df_cleaned['qty_ordered'] < 2)]
df_cleaned = df_cleaned[(df_cleaned['discount_amount'] >= 0) & (df_cleaned['discount_amount'] < 2)]
df_cleaned = df_cleaned[(df_cleaned['grand_total'] > 0) & (df_cleaned['grand_total'] < 2600)]




#graficamos los boxplot de las variables numericas para verificar que no queden valores atipicos
def plot_boxplot(column):
    sns.boxplot(x=df_cleaned[column], color='lightblue', 
            flierprops={'marker':'o', 'markerfacecolor':'red', 'markersize':8})
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

plot_boxplot('price')
plot_boxplot('qty_ordered')
plot_boxplot('discount_amount')
plot_boxplot('grand_total')

# hacemos un analisis de correlacion entre las variables numericas de precio y descuento
df_cleaned.plot.scatter(x='price', y='discount_amount')
plt.title('Precio vs Descuento')
plt.xlabel('Precio')
plt.ylabel('Descuento')
plt.show()

df_cleaned['status'].value_counts().plot(kind='bar', color = 'orange')
plt.title('Frecuencia de estados de transaccion')
plt.xlabel('Estado')
plt.ylabel('Frecuencia')
plt.show()

df_cleaned['category_name_1'].value_counts().plot(kind='bar', color='red')
plt.title('Frecuencia de categorias')
plt.xlabel('Categoria')
plt.ylabel('Frecuencia')
plt.show()


# Organizaremos los estados que significan lo mismo en grupos más generales para simplificar el análisis

state_mapping = {
    'complete': 'completed',
    'received': 'completed',
    'paid': 'completed',
    'cod': 'completed',
    'closed': 'completed',
    'canceled': 'cancelled',
    'order_refunded': 'refunded',
    'refund': 'refunded',
    'pending': 'pending',
    'pending_paypal': 'pending',
    'processing': 'pending',
    'holded': 'pending',
    '\\N' : 'completed'
}

# Reemplazar los estados con los valores nuevos
df_cleaned['status'] = df_cleaned['status'].replace(state_mapping)

# Verificar con el gráfico de barras
df_cleaned['status'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Frecuencia de Estados de Transacción Simplificados')
plt.xlabel('Estado')
plt.ylabel('Frecuencia')
plt.show()


state_mapping = {
    '\\N' : "Men's Fashion",
}

# Reemplazar los estados con los valores nuevos
df_cleaned['category_name_1'] = df_cleaned['category_name_1'].replace(state_mapping)

# Calcular el total gastado por cada categoría
category_totals = df_cleaned.groupby('category_name_1')['grand_total'].sum().reset_index()

# Verificar los datos
print(category_totals.head())

# Crear un escalador para normalizar los datos
scaler = MinMaxScaler()

# Normalizar el total gastado
category_totals['total_scaled'] = scaler.fit_transform(category_totals[['grand_total']])

# Aplicar K-means con un número de clusters
kmeans = KMeans(n_clusters=3, random_state=42)
category_totals['cluster'] = kmeans.fit_predict(category_totals[['total_scaled']])

# Ver los resultados
print(category_totals[['category_name_1', 'grand_total', 'cluster']])

# Graficar los clusters
plt.figure(figsize=(10,6))
plt.scatter(category_totals['category_name_1'], category_totals['grand_total'], c=category_totals['cluster'], cmap='viridis')
plt.title('Clusters de Categorías por Total Gastado')
plt.xlabel('Categoría de Producto')
plt.ylabel('Total Gastado')
plt.xticks(rotation=90)
plt.show()




# Calcular la frecuencia de cada estado en la columna 'status'
status_totals = df_cleaned['status'].value_counts().reset_index()
status_totals.columns = ['status', 'frequency']  # Renombrar columnas

# Verificar los datos
print(status_totals.head())

# Aplicar K-means clustering con un número de clusters (puedes ajustar este número)
kmeans = KMeans(n_clusters=3, random_state=42)
status_totals['cluster'] = kmeans.fit_predict(status_totals[['frequency']])

# Verificar los resultados
print(status_totals[['status', 'frequency', 'cluster']].head())

# Graficar los clusters
plt.figure(figsize=(10,6))
plt.scatter(status_totals['status'], status_totals['frequency'], c=status_totals['cluster'], cmap='viridis')
plt.title('Clusters de Estados por Frecuencia')
plt.xlabel('Estado')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)
plt.show()


