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
#Descargar stopwords de nltk
nltk.download('stopwords')

# Obtener stopwords en inglés
stop_words = set(stopwords.words('english'))

# Limpiar el texto eliminando "br", etiquetas HTML y caracteres no alfabéticos
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Eliminar "br" en sus diferentes formas
    text = re.sub(r'<.*?>', ' ', text)  # Eliminar cualquier otra etiqueta HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar cualquier carácter no alfabético
    text = text.lower()  # Convertir a minúsculas
    return text.strip()
 
# Limpiar la columna 'Text'
data['cleaned_text'] = data['Text'].apply(clean_text)

#Mostrar la nube de palabras
wordcloud = WordCloud().generate(' '.join(data['cleaned_text']))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#3. Clasificando lo tweets

#Clasificacion de reseñas como positivas y negativas

#Score de 3 es considerado una reseña neutra
#Remover las opniniones neutrales
data = data[data['Score'] != 3]

#Creación la nueva columna Sentiment
#Reseñas positivas serán clasificadas como +1, y reseñas negativas serán clasificadas como -1.
data['Sentiment'] = data['Score'].apply(lambda x: +1 if x > 3 else -1)
print(data.head())


#4. Más análisis de datos

#Construir nube de palabras para reseñas positivas y negativas

#Generamos el dataframe de reseñas positivas y también la nube de palabras
def positive_wordcloud(data, col, sentiment_col):
    df_positive = data[data[sentiment_col] == 1]

    positive_txt = ' '.join(df_positive[col].dropna().astype(str))
    positive_wordcloud = WordCloud().generate(positive_txt)

    plt.figure(figsize=(10, 8))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('WordCloud para Reseñas Positivas')
    plt.axis('off')

    plt.show()
    
positive_wordcloud(data,'cleaned_text','Sentiment')

#Generamos el dataframe de reseñas negativas y también la nube de palabras
def negative_wordcloud(data, col, sentiment_col):
    df_negative = data[data[sentiment_col] == -1]

    negative_txt = ' '.join(df_negative[col].dropna().astype(str))
    negative_wordcloud = WordCloud().generate(negative_txt)

    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('WordCloud para Reseñas Negativas')
    plt.axis('off')
    plt.show()

negative_wordcloud(data,'cleaned_text','Sentiment')

#Contar la cantidad de reseñas positivas y negativas
conteo_sentimiento = data['Sentiment'].value_counts()

#Crear el gráfico de barras para ver la distribución de reseñas
plt.figure(figsize=(6, 4))
plt.bar(conteo_sentimiento.index, conteo_sentimiento.values, color=['green', 'red'])
plt.title('Cantidad de Reseñas Positivas y Negativas')
plt.xlabel('Sentimiento')
plt.ylabel('Cantidad')
plt.xticks(ticks=[-1, 1], labels=['Negativas', 'Positivas'])
plt.show()


#5. Construir el modelo
# Limpieza del texto eliminando puntuación, caracteres especiales, etc.
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    text = text.lower()  # Convertir a minúsculas
    return text.strip()

# Aplicar la función de limpieza en la columna 'Summary'
data['cleaned_summary'] = data['Summary'].fillna('').apply(clean_text)

# Crear el nuevo dataframe con las columnas necesarias
data_filtrada = data[['cleaned_summary', 'Sentiment']]
print(data_filtrada.head())

# Dividir los datos en conjunto de entrenamiento y prueba
X = data_filtrada['cleaned_summary']  # Reseñas
y = data_filtrada['Sentiment']  # Sentimiento

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Usar CountVectorizer para crear una bolsa de palabras
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)  # Transformar el conjunto de entrenamiento
X_test_bow = vectorizer.transform(X_test)  # Transformar el conjunto de prueba

# Importar el modelo de regresión logística
model = LogisticRegression()

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train_bow, y_train)

# Hacer predicciones sobre los datos de prueba
y_pred = model.predict(X_test_bow)

# Evaluar el rendimiento del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))