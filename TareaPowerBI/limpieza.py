import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/meganbougle/LAX-Dataset/main/LAXTraffic.csv"
df= pd.read_csv(url)

#Exploracion inicial
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())
print('Duplicados: ',df.duplicated().sum())

#Esta terminal dejo de existir hace tiempo por lo que no es relevante para el analisis
df = df[~df['Terminal'].isin(["Imperial Terminal", "Miscellaneous Terminal"])]
#Esta fecha no importa porque es la fecha en la que se extrajeron los datos y realmente no contribuye al analisis
df.drop(['DataExtractDate'], inplace=True, axis=1)
#Se cambia el nombre de la columna para que sea mas facil de entender
df['Terminal']= df["Terminal"].str.replace('TBIT West Gates', 'TBIT West', regex=False)

df['ReportPeriod'] = df['ReportPeriod'].str.split(' ').str[0]
df['ReportPeriod'] = pd.to_datetime(df['ReportPeriod'], format='%m/%d/%Y')

df['Month'] = df['ReportPeriod'].dt.month 
df['Year'] = df['ReportPeriod'].dt.year

#Se dropea esta columna porque todas las filas son a inicios de mes, lo que indica que son datos extraidos a inicios de mes
df.drop(['ReportPeriod'], inplace=True, axis=1) 

#Se eliminan los valores en 0, ya que no contribuyen al analisis porque no hay viajes con 0 pasajeros y son pocos registros con ese valor
df = df[df['Passenger_Count'] != 0]

# Detectamos Outliers con un grafico de caja
plt.figure(figsize=(10, 6))
plt.boxplot(df['Passenger_Count'])
plt.title('Outliers en columna Passenger Count')
plt.ylabel('Passenger Count')
plt.grid()
plt.show()

#Se eliminan los outliers
Q1 = df['Passenger_Count'].quantile(0.25)
Q3 = df['Passenger_Count'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Passenger_Count'] >= Q1 - 1.5 * IQR) & (df['Passenger_Count'] <= Q3 + 1.5 * IQR)]

#Se vuelve a graficar para verificar que los outliers fueron eliminados
plt.figure(figsize=(10, 6))
plt.boxplot(df['Passenger_Count'])
plt.title('Columna Passenger Count sin Outliers')
plt.ylabel('Passenger Count')
plt.grid()
plt.show()

df.columns = ['Terminal', 'Arrival/Departure', 'Domestic/International', 'Passenger Count', 'Month', 'Year']


df.to_csv('LAXTrafficClean.csv', index=False)
