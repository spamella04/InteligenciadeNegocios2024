import pandas as pd
from datetime import time



pd.set_option('display.max_columns', 25)

#Cargar el dataset en un datagrame
url='https://media.githubusercontent.com/media/Silvio258/arrestDataLA/refs/heads/main/Arrest_Data_from_2020_to_Present.csv'
df=pd.read_csv(url)

print(df.head())

#Informacion general del dataset 
print(df.info())
print(df.describe())

##Limpieza de datos
#Eliminar aquellas columnas que no aportaran informacion relevante en el analisis
df.drop(['Area ID', 'Reporting District', 'Charge Group Code','Charge','Cross Street','Booking Location Code'],axis='columns', inplace=True)

#Verificar valores nulos
print('\n Filas con valores nulos:')
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print('\n')

#Eliminar columnas de las cuales no se puede descifrar su valor concreto, llenado de valores nulos y poner en mayus
df.replace('', pd.NA, inplace=True)
df = df.dropna(subset=['Charge Group Description'])
df = df.dropna(subset=['Time'])
df['Time'] = df['Time'].astype(int).astype(str).str.replace('2400', '0000')
df['Booking Date'].fillna('1/1/2000 12:00:00 AM', inplace=True)
df['Booking Time'].fillna(1120.0, inplace=True)
df['Booking Time'] = df['Booking Time'].astype(int).astype(str).str.replace('2400', '0000')
df['Disposition Description'].fillna('NOT ESPECIFIED', inplace=True)
df['Booking Location'].fillna('NOT ESPECIFIED', inplace=True)
df['Area Name']=df['Area Name'].str.upper()
df['Charge Group Description']=df['Charge Group Description'].str.upper()
df = df.drop(df[df['Age'] == 0].index)


print('\n Filas con valores nulos despues de reemplazo')
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print('\n')


#Renombramiento de columnas para mejor entendimiento y coherencia
df.rename(columns={'Time': 'Arrest Time'}, inplace=True)
df.rename(columns={'Sex Code': 'Sex'}, inplace=True)
df.rename(columns={'Descent Code': 'Ethnicity'}, inplace=True)
df.rename(columns={'Charge Group Description': 'Charge Group'}, inplace=True)
df.rename(columns={'Arrest Type Code': 'Arrest Type'}, inplace=True)

#Reemplazar codigos o abreviamientos con su valor correspondiente (Ejm M con Male) y otros cambios
df['Report Type'] = df['Report Type'].replace({'RFC': 'RELEASED FROM CUSTODY'})
df['Sex'] = df['Sex'].replace({'M': 'MALE', 'F': 'FEMALE'})
df['Ethnicity'] = df['Ethnicity'].replace({'A': 'OTHER ASIAN', 'B': 'BLACK', 'C': 'CHINESE', 'D': 'CAMBODIAN', 'F': 'FILIPINO', 'G': 'GUAMANIAN', 
                                     'H': 'HISPANIC', 'I': 'AMERICAN INDIAN/ALASKAN NATIVE', 'J': 'JAPANESE', 'K': 'KOREAN', 
                                     'L': 'LAOTIAN', 'O': 'OTHER', 'P': 'PACIFIC ISLANDER', 'S': 'SAMOAN', 'U': 'HAWAIIAN', 'V': 'VIETNAMESE', 
                                     'W': 'WHITE', 'X': 'UNKNOWN', 'Z': 'ASIAN INDIAN'})
df['Arrest Type'] = df['Arrest Type'].replace({'F': 'FELONY', 'M': 'MISDEMEANOR', 'I': 'INFRACTION', 'O': 'OTHER', 'D': 'OTHER'})
df['Charge Group'] = df['Charge Group'].replace({'WEAPON (CARRY/POSS)': 'WEAPON POSS','MISCELLANEOUS OTHER VIOLATIONS':'MISC. VIOLATIONS'})
df['Booking Location'] = df['Booking Location'].replace({'METRO - JAIL DIVISION':'METRO JAIL DIV','VALLEY - JAIL DIV':'VALLEY JAIL DIV'})


#Observacion de la distribucion de los datos
rt_freq = df['Report Type'].value_counts()
print(rt_freq)
print('\n')

s_freq = df['Sex'].value_counts()
print(s_freq)
print('\n')

e_freq = df['Ethnicity'].value_counts()
print(e_freq)
print('\n')

c_freq = df['Charge Group'].value_counts()
print(c_freq)
print('\n')

a_freq = df['Arrest Type'].value_counts()
print(a_freq)
print('\n')

d_freq = df['Disposition Description'].value_counts()
print(d_freq)
print('\n')



#Asginar tipos de datos correctos, como fecha 
df['Arrest Date'] = pd.to_datetime(df['Arrest Date'], format='%m/%d/%Y %I:%M:%S %p').dt.strftime('%m-%d-%Y')
df['Booking Date'] = pd.to_datetime(df['Booking Date'], format='%m/%d/%Y %I:%M:%S %p').dt.strftime('%m-%d-%Y')
print(df.iloc[1436])

df['Arrest Time'] = pd.to_datetime(df['Arrest Time'].str.zfill(4), format='%H%M').dt.time
print(df.iloc[1436])
df['Booking Time'] = pd.to_datetime(df['Booking Time'].str.zfill(4), format='%H%M').dt.time


#Nuevas Columnas
def age_group(age):
    if age < 13:
        return 'INFANT'
    elif 12 < age < 20:
        return 'ADOLESCENT'
    elif 19 < age < 26:
        return 'YOUNG ADULT'
    elif 25 < age < 35:
        return 'ADULT'
    elif 34 < age < 46:
        return 'MIDDLE-AGED ADULT'
    elif 45 < age < 56:
        return 'OLDER ADULT'
    else:
        return 'SENIOR'
df['Age Group'] = df['Age'].apply(age_group)

def categorize_time(hour):
    if hour == 0:
        return 'MIDNIGHT'
    elif 0 < hour < 6:
        return 'EARLY MORNING'
    elif 6 <= hour < 12:
        return 'MORNING'
    elif 12 <= hour < 18:
        return 'AFTERNOON'
    elif 18 <= hour < 21:
        return 'EVENING'
    elif 21 <= hour < 24:
        return 'NIGHT'

df['Arrest Time Category'] = df['Arrest Time'].apply(lambda x: categorize_time(x.hour))

print(df.head())
print(df.info())
print(df.describe())
print('\n')

df.to_csv('C:/Users/silvi/Desktop/Octavo Semestre/Inteligencia de Negocios/Segundo Corte/Dashboards/Arrest_Data_from_2020_to_Present_Cleaned.csv', index=False)