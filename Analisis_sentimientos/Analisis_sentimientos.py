#Importar librerías a utilizar
import gdown
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#1. Leer el dataframe

#Cargar el dataframe
url = 'https://drive.google.com/uc?id=1nONpng5xOXNr2kK-bdtLL4RsaingezVK'
output = 'Reviews.csv'
if not os.path.exists(output):
    print(f"El archivo {output} no existe. Descargando...")
    gdown.download(url, output, quiet=False)
else:
    print(f"El archivo {output} ya está descargado.")
data = pd.read_csv(output)

#Explorar el dataframe
print(data.head())
print("")
print(data.shape)
print("")
print(data.dtypes)

#Ver la cantidad de datos vacíos
data.isnull().sum()


#2. Análisis de datos

#Observar la variable Score para revisar si la mayoría de las calificaciones son positivas o negativas
#Utilizaremos un diagrama de barras
conteo_score = data['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Conteo de calificaciones por score',
          figsize=(10, 5))
conteo_score.set_xlabel('Score')
conteo_score.set_ylabel('Cantidad')
plt.xticks(rotation=0)
plt.show()

#Crear nube de palabras más utilizadas en las reseñas

# Descargar los stopwords
nltk.download('stopwords')

# Unir todas las reseñas en un solo string
text = ' '.join(review for review in data['Text'].astype(str))

# Definir las palabras comunes que no queremos en la nube de palabras
stop_words = set(stopwords.words('english'))

# Crear la nube de palabras
wordcloud = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Mostrar nube de palabras positivas