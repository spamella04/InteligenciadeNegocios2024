# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats
import statsmodels.api as sm

# %%
url = 'https://media.githubusercontent.com/media/musanchez/DataSets/refs/heads/main/realtor-data.csv'
df = pd.read_csv(url)

df.head() # Verificamos que el dataset se haya cargado correctamente


# %%
# Estadísticas descriptivas
# Evita que aparezcan los exponenciales y poder apreciar mejor el dataset
def set_overview(df):
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print('Información general del dataset:')
    print(df.describe())
    print('Valores nulos en cantidad:')
    print(df.isna().sum()) # Verificamos si hay datos nulos
    print('Valores nulos en porcentaje:')
    print(df.isna().mean() * 100) # Observamos de manera porcentual los datos nulos

# Haremos un grafico de caja para cada variable numérica, le pasaremos el dataframe y todas las variables numéricas
def box_plot_vars(df):
    # Filtrar automáticamente solo las columnas numéricas
    numeric_vars = df.select_dtypes(include=['number']).columns
    
    # Definir el tamaño de la cuadrícula según el número de variables numéricas
    num_vars = len(numeric_vars)
    num_cols = 3  # Número de columnas en la cuadrícula
    num_rows = (num_vars + num_cols - 1) // num_cols  # Calcula las filas necesarias
    
    # Crear el grid de subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Aplana la cuadrícula de ejes
    
    for i, var in enumerate(numeric_vars):
        sns.boxplot(x=df[var], ax=axes[i], color='lightblue', 
                    flierprops={'marker':'o', 'markerfacecolor':'red', 'markersize':8})
        
        # Título y etiquetas
        axes[i].set_title(f'Box Plot de {var}', fontsize=16)  # Agregar título con el nombre de la variable
        axes[i].set_xlabel(var, fontsize=12)  # Etiqueta del eje X
        axes[i].set_ylabel('Valores', fontsize=12)  # Etiqueta del eje Y
        
        # Quitar notación científica en el eje X
        axes[i].ticklabel_format(style='plain', axis='x')
        axes[i].grid(True)  # Añadir rejilla

        # Rotar las etiquetas del eje X para evitar traslapes
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(45)  # Ajusta el ángulo de rotación a 45 grados
    
    # Eliminar cualquier gráfico vacío si es que sobran ejes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()  # Ajustar los gráficos para que no se solapen
    plt.show()

def remove_outliers_iqr(df, column_name):

    if isinstance(df[column_name], np.ma.MaskedArray):
        df[column_name] = df[column_name].data

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[column_name] >= (Q1 - 1.5 * IQR)) & (df[column_name] <= (Q3 + 1.5 * IQR))]
    return df

def plot_correlation_matrix(df):
    df_numeric = df.select_dtypes(include=['number'])
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de correlación', fontsize=16)
    plt.show()

def plot_histograms(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, color='skyblue')
        plt.title(f'Distribución de {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True)
        plt.show()

def plot_state_distribution(df, column, title):
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x=column, hue=column, order=df[column].value_counts().index, palette='viridis', dodge=False)
    plt.title(title, fontsize=16)
    plt.xlabel('Estado', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.legend([], [], frameon=False)  # Ocultar leyenda
    plt.show()

def plot_average_price_per_state(df):
    avg_price_per_state = df.groupby('state')['price'].mean().reset_index()
    avg_price_per_state = avg_price_per_state.sort_values(by='price', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=avg_price_per_state, x='state', y='price', hue='state', palette='viridis', dodge=False)
    plt.title('Precio medio de las propiedades por estado', fontsize=16)
    plt.xlabel('Estado', fontsize=12)
    plt.ylabel('Precio Medio', fontsize=12)
    plt.xticks(rotation=90)
    plt.ticklabel_format(style='plain', axis='y')  # Quitar notación científica en el eje Y
    plt.grid(True)
    plt.legend([], [], frameon=False)  # Ocultar leyenda
    plt.show()

def remove_outliers_zscore(df, column_name, threshold):
    z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
    df = df[(z_scores < threshold)]
    return df


def fit_ols_wls(X, y, weights):
    X_clean = X.astype(float).dropna()
    y_clean = y.loc[X_clean.index].dropna()
    
    # Agregar una constante a X
    X_sm = sm.add_constant(X_clean)

    # Ajustar el modelo WLS
    model_wls = sm.WLS(y_clean, X_sm, weights=weights.loc[X_clean.index]).fit()
    
    # Ajustar el modelo OLS
    model_ols = sm.OLS(y_clean, X_sm).fit()
    
    # Imprimir el resumen de los modelos
    print(f"Resumen WLS:\n", model_wls.summary())
    print(f"Resumen OLS:\n", model_ols.summary())

set_overview(df)
plot_state_distribution(df, 'state', 'Distribución total de observaciones por estado')
plot_average_price_per_state(df)
#Observamos de manera porcentual los datos nulos
# Las columnas brokered by, street, zip_code realmente no aportan información relevante para el análisis.
# Notamos datos extraños, por ejemplo, bed es el número de habitaciones y el valor máximo es de 473, sin sentido.
# Lo mismo podemos notar en el caso de bath, que es el número de baños y el valor máximo es de 830.

# %%
# Limpieza de datos

# Eliminamos registros de casas sin precio
df_cleaned = df.dropna(subset=['price']).copy()
df_cleaned.drop_duplicates(inplace=True)

df_cleaned['prev_sold_date'] = df_cleaned['prev_sold_date'].fillna('Not sold before')

df_cleaned = df_cleaned.drop(columns=['brokered_by', 'street', 'zip_code'])

# Eliminamos columnas que tengan demasiados valores nulos
df_cleaned = df_cleaned[df_cleaned.isna().sum(axis=1) < 0.5 * len(df_cleaned.columns)]

# Asigna la cantidad de baños promedios por cantidad de habitaciones
# Hacemos esto para la limpieza, así imputar valores faltantes.
bath_avg_bed = df.groupby('bed')['bath'].mean().round(0).reset_index()
bath_avg_bed.columns = ['bed', 'bath_avg']

# Vamos a llenar los valores faltantes de bath con el promedio de baños por cantidad de habitaciones
# Agregamos al cleaned set la columna con el promedio de baños por cantidad de habitaciones
df_cleaned = df_cleaned.merge(bath_avg_bed, on='bed', how='left')
df_cleaned['bath'] = df_cleaned['bath'].fillna(df_cleaned['bath_avg'])

# Eliminamos la columna de promedio de baños por cantidad de habitaciones, ya no se necesita
df_cleaned = df_cleaned.drop(columns='bath_avg')

# Imputamos valores faltantes en bed
df_cleaned = df_cleaned.fillna(df_cleaned['bed'].median())

set_overview(df_cleaned)


# %%
# Eliminación de outliers
box_plot_vars(df_cleaned)

# %%
# Eliminar valores que claramente son absurdos
# Segun san google, la casa mas pequeña del mundo es de 325 pies cuadrados, no puede ser que haya casas con 3 pies cuadrados
df_cleaned = df_cleaned[(df_cleaned['bed'] != 0) & (df_cleaned['bath'] != 0) & (df_cleaned['price'] > 0) & (df_cleaned['house_size'] > 300)]
# Eliminamos datos en los que el tamaño del lote es menor que el tamaño de la casa
df_cleaned = df_cleaned[df_cleaned['acre_lot'] * 43560 >= df_cleaned['house_size']]

df_cleaned = df_cleaned[df_cleaned['state'] != 3.0]
df_cleaned['state'] = df_cleaned['state'].astype(str)


df_cleaned['log_house_size'] = np.log(df_cleaned['house_size'])

df_cleaned = remove_outliers_iqr(df_cleaned, 'log_house_size')
df_cleaned = remove_outliers_zscore(df_cleaned, 'bath', 3)
df_cleaned = remove_outliers_zscore(df_cleaned, 'bed', 3)
df_cleaned = remove_outliers_iqr(df_cleaned, 'acre_lot')


# %%
df_cleaned['log_price'] = np.log(df_cleaned['price'])
df_cleaned['log_price'] = mstats.winsorize(df_cleaned['log_price'], limits=[0.05, 0.05])
#df_cleaned = df_cleaned.drop(columns=['price', 'house_size', 'acre_lot'])
box_plot_vars(df_cleaned)

# %%
plot_correlation_matrix(df_cleaned)
col = ['log_house_size', 'log_price']

plot_histograms(df_cleaned, col)

plt.figure(figsize=(10, 6))
sns.kdeplot(df_cleaned['log_price'], fill=True, color='skyblue')
plt.title('Density Plot para log_price', fontsize=16)
plt.xlabel('log_price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.show()

sm.qqplot(df_cleaned['log_price'], line='s')
plt.title('Q-Q Plot for log_price')
plt.grid(True)
plt.show()

plot_average_price_per_state(df_cleaned)

# %%
extreme_low = np.exp(11.6)  # Valor original correspondiente a log_price ~ 11.5
extreme_high = np.exp(13.9)  # Valor original correspondiente a log_price ~ 14.0
# Esto se estuvo variando para ver mas o menos que valores de extreme low y high me servian, pero no sirvio de mucho
# Mostrar las filas con esos valores extremos
df_extremes = df_cleaned[(df_cleaned['price'] <= extreme_low) | (df_cleaned['price'] >= extreme_high)]
print(df_extremes[['price', 'log_price', 'state']])
df_extremes.groupby('state').size().reset_index(name='frequency')

# Filtrar los datos extremos bajos y altos
df_extreme_low = df_cleaned[df_cleaned['price'] <= extreme_low]
df_extreme_high = df_cleaned[df_cleaned['price'] >= extreme_high]

# Muestra un grafico de barras para la distribucion por estado de precios extremos bajos
plot_state_distribution(df_extreme_low, 'state', 'Distribución de precios extremos bajos por estado')

# Muestra un grafico de barras para la distribucion por estado de precios extremos altos
plot_state_distribution(df_extreme_high, 'state', 'Distribución de precios extremos altos por estado')

# Muestra un grafico de barras para la distribucion por estado de todas las observaciones
plot_state_distribution(df_cleaned, 'state', 'Distribución total de observaciones por estado')


# %%
state_stats = df_cleaned.groupby('state')['log_price'].agg(['var', 'count']).reset_index()
print(state_stats)
#ponderación por varianza
state_stats['weight_var'] = 1 / state_stats['var']
state_stats['weight_geo'] = 1/ state_stats['count']
state_stats['weight'] = state_stats['weight_var'] * state_stats['weight_geo']
df_cleaned = df_cleaned.merge(state_stats[['state', 'weight']], on='state', how='left')

weights_by_state = df_cleaned.groupby('state')['weight'].max().sort_values(ascending=False).reset_index()

# Crear el gráfico de barras para los pesos por estado
plt.figure(figsize=(12, 8))
sns.barplot(data=weights_by_state, x='state', y='weight', hue='state', palette='viridis')
plt.title('Pesos por Estado', fontsize=16)
plt.xlabel('Estado', fontsize=12)
plt.ylabel('Peso', fontsize=12)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# %%

df_cleaned_dummies = pd.get_dummies(df_cleaned, columns=['state'], drop_first=True)
X_dum = df_cleaned_dummies.drop(columns=['price', 'log_price', 'status', 'prev_sold_date', 'weight', 'house_size', 'city'])
y_dum = df_cleaned_dummies['log_price']
weights = df_cleaned['weight']
fit_ols_wls(X_dum, y_dum, weights)

# %%
X_size = df_cleaned_dummies.drop(columns=['price', 'log_price', 'status', 'prev_sold_date', 'weight', 'house_size', 'city', 'bed', 'bath', 'acre_lot'])
y_size = df_cleaned_dummies['log_price'].loc[X_size.index]
fit_ols_wls(X_size, y_size, weights)

# %%
# Queria ver si mejoraba algo quitando los picos, pero no mejora nada
df_extremes_cleaned = df_cleaned[(df_cleaned['price'] > extreme_low) & (df_cleaned['price'] < extreme_high)]
#df_cleaned_no_california_no_ohio = df_cleaned[(df_cleaned['state'] != 'California') & (df_cleaned['state'] != 'Ohio')].copy()
df_extremes_cleaned_dum = pd.get_dummies(df_extremes_cleaned, columns=['state'], drop_first=True)
X_cs = df_extremes_cleaned_dum.drop(columns=['price', 'log_price', 'status', 'prev_sold_date', 'weight', 'house_size', 'city'])
y_cs = df_extremes_cleaned_dum['log_price']
weights_cd = df_extremes_cleaned_dum['weight']

plt.figure(figsize=(10, 6))
sns.kdeplot(df_extremes_cleaned_dum['log_price'], fill=True, color='skyblue')
plt.title('Density Plot for log_price', fontsize=16)
plt.xlabel('log_price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.show()

fit_ols_wls(X_cs, y_cs, weights_cd)

# %%
# Nota, aca me puse a mover los valores de los extremos para ver si se podia mejorar el modelo, se miraba raro y ahi lo deje
sm.qqplot(df_extremes_cleaned['log_price'] * df_extremes_cleaned['weight'], line='s')
plt.title('Q-Q Plot para log_price')
plt.grid(True)
plt.show()

