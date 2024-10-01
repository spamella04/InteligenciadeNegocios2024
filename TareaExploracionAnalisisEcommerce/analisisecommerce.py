import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', 25)

#Cargar el dataset
url='https://media.githubusercontent.com/media/Silvio258/ecommerce/refs/heads/master/Pakistan%20Largest%20Ecommerce%20Dataset.csv'
df=pd.read_csv(url)

print(df.head())

#Informacion general del dataset y algunas medidas estadisticas
print(df.info())
print(df.describe())

##Limpieza de datos
#Eliminar columnas que no aportaran informacion relevante en este analisis
df.drop(['sales_commission_code','BI Status', 'M-Y','Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25'],axis='columns', inplace=True)
print(df.head())

#Verificar cuantas filas tienen todas sus columnas con datos vacios
null_rows_count = df.isnull().all(axis=1).sum()
print(f"Filas con todas las columnas nulas: {null_rows_count}")

#Eliminar filas con todas las columnas vacias y las filas con categoria vacias
df.replace('', pd.NA, inplace=True)
df.replace('\\N', pd.NA, inplace=True)
df = df.dropna(how='all')
df = df.dropna(subset=['category_name_1'])
df = df.dropna(subset=['Customer Since'])

null_rows_count = df.isnull().all(axis=1).sum()
print(f"Filas con todas las columnas nulas: {null_rows_count}")

#Pone en 0 los valores incorrectos en la columna discount_amount que aparecen como negativos
df['discount_amount'] = df['discount_amount'].where(df['discount_amount'] >= 0, 0)

#Llenado de valores nulos
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

df.fillna('N/S', inplace=True)

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

#Tablas de frecuencias
status_freq = df['status'].value_counts()
print(status_freq)
print('\n')
category_freq = df['category_name_1'].value_counts()
print(category_freq)
print('\n')
payment_freq = df['payment_method'].value_counts()
print(payment_freq)
print('\n')
financialy_freq = df['FY'].value_counts()
print(financialy_freq)

##Corregir valores incoherentes en las columnas
#Cambiar el nombre de algunos valores con el mismo significado para mayor coherencia
df['status'] = df['status'].replace('order_refunded', 'refund')
df['status'] = df['status'].replace('cod', 'cashatdoorstep')
df['payment_method'] = df['payment_method'].replace('cod', 'cashatdoorstep')

status_freq = df['status'].value_counts()
print(status_freq)
print('\n')
payment_freq = df['payment_method'].value_counts()
print(payment_freq)
print('\n')

#Recalculo de la columna grand_total (incluye el descuento) y de mv, que sera renombrada a total (sin descuento) ,y otros renombrados para mayor orden
df.rename(columns={'Working Date': 'working_date'}, inplace=True)
df.rename(columns={'Year': 'transaction_year'}, inplace=True)
df.rename(columns={'Month': 'transaction_month'}, inplace=True)
df.rename(columns={'Customer Since': 'customer_since'}, inplace=True)
df.rename(columns={'Customer ID': 'customer_id'}, inplace=True)

df.columns = df.columns.str.strip()
df.rename(columns={'MV': 'total'}, inplace=True)

#Pasar a tipo fecha
df['created_at'] = pd.to_datetime(df['created_at'], format='%m/%d/%Y')
df['working_date'] = pd.to_datetime(df['working_date'], format='%m/%d/%Y')

df['transaction_year'] = pd.to_datetime(df['transaction_year'], format='%Y')
df['transaction_year'] = df['transaction_year'].dt.strftime('%Y')

df['transaction_month'] = pd.to_datetime(df['transaction_month'], format='%m')
df['transaction_month'] = df['transaction_month'].dt.strftime('%m')

df['customer_since'] = pd.to_datetime(df['customer_since'], format='%Y-%m')
df['customer_since'] = df['customer_since'].dt.strftime('%m-%Y')

print('\n')
print(df[['created_at','working_date','transaction_year','transaction_month', 'customer_since']].head())
print('\n')

df['total'] = df['price'] * df['qty_ordered']

#Eliminar filas donde el descuento es mayor que el total resultando en datos incoherentes
df = df[df['discount_amount'] < df['total']]

df['grand_total'] = df['total'] - df['discount_amount']

print(df.info())
print(df.describe())
print('\n')
print(df[['price','qty_ordered','total','discount_amount', 'grand_total']].head())

##Graficos
#Grafico de barras del total de ventas por categoria
total_sales_per_category = df.groupby('category_name_1')['grand_total'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(total_sales_per_category['category_name_1'], total_sales_per_category['grand_total'], color='skyblue')
plt.title('Total de ventas segun la categoria')
plt.xlabel('Category')
plt.ylabel('Total sales')
plt.xticks(rotation=45)  
plt.ticklabel_format(style='plain', axis='y')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

#Grafico de barras del total de ventas por categoria
qty_sales_per_category = df.groupby('category_name_1')['item_id'].count().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(qty_sales_per_category['category_name_1'], qty_sales_per_category['item_id'], color='skyblue')
plt.title('Cantidad de ventas segun la categoria')
plt.xlabel('Category')
plt.ylabel('Total qty of sales')
plt.xticks(rotation=45)  
plt.ticklabel_format(style='plain', axis='y')
plt.grid(axis='y')

plt.tight_layout()
plt.show()


#Ventas en cada año fiscal (FY17, FY18, FY19)
df_monthlysales_fy17  = df[df['FY'] == 'FY17']
print(df_monthlysales_fy17['created_at'].describe())
monthly_sales_fy17 = df_monthlysales_fy17.groupby('transaction_month')['grand_total'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(monthly_sales_fy17['transaction_month'], monthly_sales_fy17['grand_total'])
plt.title('Ventas totales por mes en el año fiscal 17')
plt.xlabel('Month')
plt.ylabel('Total sales')
plt.xticks(range(0, 12),['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])  
plt.grid(axis='y')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()


df_monthlysales_fy18  = df[df['FY'] == 'FY18']
print(df_monthlysales_fy18['created_at'].describe())
monthly_sales_fy18 = df_monthlysales_fy18.groupby('transaction_month')['grand_total'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(monthly_sales_fy18['transaction_month'], monthly_sales_fy18['grand_total'])
plt.title('Ventas totales por mes en el año fiscal 18')
plt.xlabel('Month')
plt.ylabel('Total sales')
plt.xticks(range(0, 12),['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])  
plt.grid(axis='y')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()


df_monthlysales_fy19  = df[df['FY'] == 'FY19']
print(df_monthlysales_fy19['created_at'].describe())
monthly_sales_fy19 = df_monthlysales_fy19.groupby('transaction_month')['grand_total'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(monthly_sales_fy19['transaction_month'], monthly_sales_fy19['grand_total'])
plt.title('Ventas totales por mes en el año fiscal 19')
plt.xlabel('Month')
plt.ylabel('Total sales')
plt.xticks(range(0, 12),['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])  
plt.grid(axis='y')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()


#Line graph del numero de transacciones por año 
monthly_transactions = df.groupby(['transaction_year', 'transaction_month']).size().reset_index(name='transaction_count')


pivot_table = monthly_transactions.pivot(index='transaction_month', columns='transaction_year', values='transaction_count')
pivot_table.index = pivot_table.index.astype(int)

plt.figure(figsize=(10, 6))

for year in pivot_table.columns:
    if year == 2016:
        months_to_plot = pivot_table.loc[(pivot_table.index >= 7), year]
    elif year == 2018:
        months_to_plot = pivot_table.loc[(pivot_table.index <= 8), year]
    else:
        months_to_plot = pivot_table.loc[pivot_table.index, year]

    plt.plot(months_to_plot.index, months_to_plot, marker='o', label=year)

plt.title('Num de transacciones cada mes por año')
plt.xlabel('Month')
plt.ylabel('No. of transactions')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.legend(title='Year')
plt.grid()
plt.tight_layout()
plt.show()


monthly_total_transactions = df.groupby('transaction_month').size().reset_index(name='total_transactions')
monthly_discounted_transactions = df[df['discount_amount'] > 0].groupby('transaction_month').size().reset_index(name='discounted_transactions')

monthly_data = pd.merge(monthly_total_transactions, monthly_discounted_transactions, on='transaction_month', how='outer').fillna(0)


plt.figure(figsize=(12, 6))
plt.plot(monthly_data['transaction_month'], monthly_data['total_transactions'], marker='o', label='Transactions')
plt.plot(monthly_data['transaction_month'], monthly_data['discounted_transactions'], marker='o', label='Transactions with discount')
plt.title('Trend de transacciones por mes')
plt.xlabel('Month')
plt.ylabel('No. of transactions')
plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.legend(title='Transaction type')
plt.grid()
plt.tight_layout()
plt.show()

#Grafico de pie con los tipos de metodos de pago 
payment_counts = df['payment_method'].value_counts()

plt.figure(figsize=(8, 8))
wedges, texts = plt.pie(payment_counts, startangle=90, counterclock=False)

plt.legend(wedges, [f"{label}: {pct:.3f}%" for label, pct in zip(payment_counts.index, 100 * payment_counts / payment_counts.sum())],
           title="Payment method", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Distribucion de metodos de pago')
plt.tight_layout()
plt.show()

##Dataframe que contenga informacion de caracteristicas de compra de los clientes
#Obtener cantidad total gastada en todas las compras del cliente
amount_spent= df.groupby('customer_id')['grand_total'].sum().reset_index()
amount_spent.columns= ['customer_id', 'total_spent']
print(amount_spent.head())

#Frecuencia de transacciones (cantidad) que ha hecho el cliente
transaction_frequency= df.groupby('customer_id')['item_id'].count()
transaction_frequency= transaction_frequency.reset_index()
transaction_frequency.columns= ['customer_id', 'frequency']
print(transaction_frequency.head())

df_customer_info = pd.merge(amount_spent, transaction_frequency, on='customer_id', how='inner')
print(df_customer_info.head())

#Que tan reciente fue la ultima compra DEL CLIENTE
df['created_at'] = pd.to_datetime(df['created_at'],format='%m-%d-%Y')
max_date = max(df['created_at'])
df['days_since_purchase'] = max_date - df['created_at']

purchase_recency = df.groupby('customer_id')['days_since_purchase'].min()
purchase_recency = purchase_recency.reset_index()
purchase_recency.columns= ['customer_id', 'recency']
purchase_recency['recency'] = purchase_recency['recency'].dt.days // 30
print(purchase_recency.head())

df_customer_info = pd.merge(df_customer_info, purchase_recency, on='customer_id', how='inner')
df_customer_info.columns = ['customer_id', 'total_spent', 'transactions', 'last_purchase_in_months']

print(df_customer_info.head())

#Edad del cliente desde la primera compra
df['customer_since'] = pd.to_datetime(df['customer_since'],format='%m-%Y')
customer_purchase_age= df.groupby('customer_id')['customer_since'].min()
customer_purchase_age= customer_purchase_age.reset_index()
customer_purchase_age.columns= ['customer_id', 'customer_purchase_age']

customer_purchase_age['customer_purchase_age'] = (max_date-customer_purchase_age['customer_purchase_age']).dt.days // 30
print(customer_purchase_age.head())

df_customer_info = pd.merge(df_customer_info, customer_purchase_age, on='customer_id', how='inner')
df_customer_info.columns = ['customer_id', 'total_spent', 'transactions', 'last_purchase_in_months', 'customer_purchase_age_months']

print(df_customer_info.head())

#Estandarizar los datos
scaler = StandardScaler()
df_customer_info['total_spent_scaled'] = scaler.fit_transform(df_customer_info[['total_spent']])
df_customer_info['transactions_scaled'] = scaler.fit_transform(df_customer_info[['transactions']])
df_customer_info['last_purchase_in_months_scaled'] = scaler.fit_transform(df_customer_info[['last_purchase_in_months']])
df_customer_info['customer_purchase_age_months_scaled'] = scaler.fit_transform(df_customer_info[['customer_purchase_age_months']])

#Clustering con kmeans
kmeans = KMeans(n_clusters=4, random_state=42)
df_customer_info['cluster']=kmeans.fit_predict(df_customer_info[['total_spent_scaled', 'transactions_scaled', 'last_purchase_in_months_scaled', 'customer_purchase_age_months_scaled']])

sns.scatterplot(x='last_purchase_in_months', y='transactions', hue='cluster', data=df_customer_info, palette=sns.color_palette('hls', 4))
plt.title('Cluster de datos de clientes')
plt.xlabel('Last purchase (Months)')
plt.ylabel('Transactions')
plt.show()

sns.scatterplot(x='customer_purchase_age_months', y='transactions', hue='cluster', data=df_customer_info, palette=sns.color_palette('hls', 4))
plt.title('Cluster de datos de clientes')
plt.xlabel('Purchase age (Months)')
plt.ylabel('Transactions')
plt.show()

sns.scatterplot(x='customer_purchase_age_months', y='total_spent', hue='cluster', data=df_customer_info, palette=sns.color_palette('hls', 4))
plt.title('Cluster de datos de clientes')
plt.xlabel('Purchase age (Months)')
plt.ylabel('Total spent')
plt.ticklabel_format(style='plain', axis='y')
plt.show()