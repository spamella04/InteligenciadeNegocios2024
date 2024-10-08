# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
# URL del archivo CSV en su forma "raw"
url = 'https://raw.githubusercontent.com/CarlosG29/repos_csv/refs/heads/main/US_Stock_Data.csv'

# Cargar el archivo CSV directamente desde GitHub
df = pd.read_csv(url)

#  filas y columnas
filas, columnas = df.shape

print(f"El archivo  tiene {filas} filas y {columnas} columnas.")
df.head()

# %%
nombres_columnas = df.columns

print("Nombres de las columnas:")
print(nombres_columnas, "\n")


df.info()

# %%
#vemos medias, desviaciones estándar, y otros.
df.describe()

# %%
df.isna().mean()

# %%
# Rellenar valores faltantes en todas las columnas excepto 'Platinum_Vol.' Debido a que el propósito es visualizar los datos, los valores faltantes se completan con datos anteriores.
# Primero, elimina la columna 'Platinum_Vol.' del DataFrame.
# Luego, aplica ffill() (relleno hacia adelante) y bfill() (relleno hacia atrás) para llenar los valores faltantes.
data_pre = df.drop(columns=['Platinum_Vol.']).ffill().bfill()

# Ahora, volvemos a añadir la columna 'Platinum_Vol.' al DataFrame.
# La columna 'Platinum_Vol.' tiene sus valores faltantes (NaN) reemplazados con 0.
# Concatena las primeras 11 columnas, luego la columna 'Platinum_Vol.' con NaNs reemplazados por 0,
# y finalmente las columnas restantes después de la posición 11.
data_pre = pd.concat([data_pre.iloc[:, :11],  # Primeras 11 columnas sin 'Platinum_Vol.'
                      df['Platinum_Vol.'].fillna(0),  # Columna 'Platinum_Vol.' con NaNs reemplazados por 0
                      data_pre.iloc[:, 11:]],  # Resto de las columnas desde la posición 11 en adelante
                     axis=1)  # Concatena a lo largo del eje de las columnas

data_pre.isna().sum()

# %%
# Reemplaza las barras (/) por guiones (-) en la columna 'Date'
# se quiere estandarizar a DD-MM-YYYY
data_pre['Date'] = data_pre['Date'].str.replace('/', '-')

# Aquí se indica que el formato es día-mes-año (DD-MM-YYYY)
data_pre['Date'] = pd.to_datetime(data_pre['Date'], format='%d-%m-%Y')

# Muestra las primeras 5 filas del DataFrame para confirmar que los cambios en 'Date' son correctos
data_pre.head()


# %%
#procesar las columnas que tienen valores de tipo object y convertirlas en valores numéricos de tipo float.
for col in data_pre.columns:
    
    
    if data_pre[col].dtype == 'object':
        # Reemplaza las comas (,) por cadenas vacías y convierte los valores en tipo float
        data_pre[col] = data_pre[col].str.replace(',', '').astype(float)

data_pre.info()

# %%

col_price = []

for col in data_pre.columns.tolist():

    if 'Price' in col:
        col_price.append(col)


date = data_pre['Date']

# Crear una tabla dinámica (pivot table) con el promedio de precios por día
1 = pd.pivot_table(data_pre, index='Date', values=col_price, aggfunc='mean')

# Muestra la tabla dinámica resultante con los promedios de precios por fecha
pivot_price_1


# %%
# Crear una tabla dinámica (pivot table) para cada columna de precios (por día)

for col in col_price:
    pivot_price_2 = pd.pivot_table(data_pre, index='Date', values=col, aggfunc='mean')
    
    print(pivot_price_2)
    
    # Imprime una línea de separación para que los resultados de cada columna sean más fáciles de leer
    print('-' * 30)


# %%
# Agregar una nueva columna 'Month' que extrae el mes de la columna 'Date'
# pd.to_datetime(data_pre['Date']) convierte la columna 'Date' a un formato datetime si no lo está
# .dt.month extrae solo el mes de cada fecha
data_pre['Month'] = pd.to_datetime(data_pre['Date']).dt.month

# Crear una tabla dinámica (pivot table) que calcula el promedio mensual de los precios
pivot_table_month_1 = pd.pivot_table(data_pre, index='Month', values=col_price, aggfunc='mean')

# Mostrar la tabla dinámica resultante con los promedios de precios por mes
pivot_table_month_1


# %%
# Crear una pivot table(tabla dinamica) para calcular el promedio mensual de los precios

for col in col_price:
    pivot_table_month_2 = pd.pivot_table(data_pre, index='Month', values=col, aggfunc='mean')
    
    # Imprime la tabla dinámica generada para la columna actual
    print(pivot_table_month_2)
    
    # Imprime una línea de separación para hacer más legible la salida
    print('-' * 20)

# %%
# Extraer el año de la columna 'Date' y crear una nueva columna llamada 'Year'
# pd.to_datetime(data_pre['Date']) convierte la columna 'Date' a un formato datetime si no lo está
# .dt.year extrae solo el año de cada fecha
data_pre['Year'] = pd.to_datetime(data_pre['Date']).dt.year

# Crear una tabla dinámica (pivot table) que calcula el promedio anual de los precios
pivot_table_year_1 = pd.pivot_table(data_pre, index='Year', values=col_price, aggfunc='mean')

# Mostrar la tabla dinámica resultante con los promedios de precios por cada año
pivot_table_year_1


# %%
# Crear una tabla dinámica (pivot table) para calcular el promedio anual de cada columna de precios

for col in col_price:
    pivot_table_year_2 = pd.pivot_table(data_pre, index='Year', values=col, aggfunc='mean')
    
    print(pivot_table_year_2)
    
    # Imprime una línea de separación para hacer más legible la salida
    print('-' * 20)


# %%

# Convertir la columna 'Month' de números a nombres de meses para que muestre nombres en lugar de números
data_pre['Month_Name'] = pd.to_datetime(data_pre['Month'], format='%m').dt.strftime('%B')

# Graficar el promedio mensual de los precios para cada columna de precios
for col in col_price:

    pivot_table_month_3 = pd.pivot_table(data_pre, index='Month_Name', values=col, aggfunc='mean')

    # Ordenar los meses por su orden cronológico
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    
    pivot_table_month_3 = pivot_table_month_3.reindex(months_order)
    
    #tabla dinámica como gráfico de barras
    pivot_table_month_3.plot(kind='bar', color=np.random.rand(3,))
    
    #  etiquetas para los ejes X, Y
    plt.xlabel('Month')
    plt.xticks(rotation=55)  
    plt.ylabel('Price $$$')  
    
    # título del gráfico
    plt.title('Monthly Average Price Graph for ' + col)  # nombre de la columna actual al título
    
    # Mostrar la leyenda (aunque en este caso, solo habrá un conjunto de datos por gráfico)
    plt.legend([col])
    
    # Mostrar el gráfico
    plt.show()

# %%
# Graficar el promedio anual de los precios para cada columna de precios
for col in col_price:
    # Crear una tabla dinámica (pivot table) que agrupa los datos por 'Year' y calcula el promedio
    pivot_table_year_3 = pd.pivot_table(data_pre, index='Year', values=col, aggfunc='mean')
    
    # Graficar la tabla dinámica resultante como un gráfico de barras
    pivot_table_year_3.plot(kind='bar', color=np.random.rand(3,))
    
    # Etiquetas y título 
    plt.xlabel('Year')
    plt.xticks(rotation=0)
    plt.ylabel('Price')  
    plt.title(f'Yearly Average Price Graph for {col}') 
    
    # Mostrar la leyenda
    plt.legend([col])
    
    # Mostrar el gráfico
    plt.show()

# %%
# Cálculo del rendimiento diario para cada columna de precios
for col in col_price:
    # Calcular el cambio porcentual diario (rendimiento diario)
    data_pre[f'{col}_return'] = data_pre[col].pct_change()

# Mostrar las primeras filas para verificar el cálculo
print(data_pre[[f'{col}_return' for col in col_price]].head())


# %%
# Visualizar el rendimiento diario para cada columna de precios
for col in col_price:
    plt.figure(figsize=(10, 6))
    plt.plot(data_pre['Date'], data_pre[f'{col}_return'], label=f'{col} Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.title(f'Daily Returns of {col}')
    plt.legend()
    plt.show()


# %%
# Cálculo de la volatilidad (desviación estándar) de los rendimientos diarios para cada columna de precios
for col in col_price:
    data_pre[f'{col}_volatility'] = data_pre[f'{col}_return'].rolling(window=30).std()

# Visualizar la volatilidad
for col in col_price:
    plt.figure(figsize=(10, 6))
    plt.plot(data_pre['Date'], data_pre[f'{col}_volatility'], label=f'{col} Volatility')

    # Ajustar las etiquetas del eje X para mostrar meses de manera más clara
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Etiquetas cada 3 meses
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato Año-Mes

    # Añadir etiquetas y título
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title(f'Volatility of {col}')
    plt.legend()

    plt.xticks(rotation=60)

    # Mostrar el gráfico
    plt.show()

# %%
# Crear la matriz de correlación usando las columnas de precios
correlation_matrix = data_pre[col_price].corr()

# Visualizar la matriz de correlación usando un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Stock Prices')
plt.show()

# %%
# Parámetros de la media móvil
window_short = 50  # Media móvil a corto plazo ( 50 días)
window_long = 200  # Media móvil a largo plazo (200 días)

# Calcular la media móvil a corto y largo plazo para cada acción
for col in col_price:
    data_pre[f'{col}_MA_{window_short}'] = data_pre[col].rolling(window=window_short).mean()
    data_pre[f'{col}_MA_{window_long}'] = data_pre[col].rolling(window=window_long).mean()

    # Graficar el precio y las medias móviles
    plt.figure(figsize=(12, 6))
    plt.plot(data_pre['Date'], data_pre[col], label=f'{col} Price', color='blue')
    plt.plot(data_pre['Date'], data_pre[f'{col}_MA_{window_short}'], label=f'{window_short}-day MA', color='red', linestyle='--')
    plt.plot(data_pre['Date'], data_pre[f'{col}_MA_{window_long}'], label=f'{window_long}-day MA', color='green', linestyle='--')

    # Añadir etiquetas y título
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{col} Price and Moving Averages ({window_short}-day, {window_long}-day)')
    plt.legend()

    plt.xticks(rotation=45)
    
    # Mostrar el gráfico
    plt.show()



