# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk as nltk
import wordcloud as wc
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



# %%
url = 'https://media.githubusercontent.com/media/cdtoruno/review-csv/refs/heads/main/Reviews.csv'
data = pd.read_csv(url)

print(data.head())

# %%
fig = px.histogram(data, x='Score', title='Calificacion del Producto')
fig.show()

# %%
#Generamos wordcloud
if 'Text' in data.columns:
    text = ' '.join(review for review in data['Text'])
    stopwords_words = set(stopwords.words('english'))
    wordcloud = wc.WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# %%
#clasificar reseñas positivas y negativas
data['Sentiment'] = np.where(data['Score'] > 3, '1', '-1').astype(int)

# Eliminar registros donde 'Score' es igual a 3
data = data[data['Score'] != 3]
print(data.head())

# %%
#creamos un dataframe solo para los registros con sentimiento positivo
positive_reviews = data[data['Sentiment'] == 1]

#creamos un dataframe solo para los registros con sentimiento negativo
negative_reviews = data[data['Sentiment'] == -1]

#verificamos que se hayan creado correctamente los dataframes
print(positive_reviews.head())
print(negative_reviews.head())

# %%
#generamos wordcloud para reseñas positivas
if 'Text' in positive_reviews.columns:
    text = ' '.join(review for review in positive_reviews['Text'])
    stopwords_words = set(stopwords.words('english'))
    wordcloud = wc.WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# %%
#generamos wordcloud para reseñas negativas
if 'Text' in negative_reviews.columns:
    text = ' '.join(review for review in negative_reviews['Text'])
    stopwords_words = set(stopwords.words('english'))
    wordcloud = wc.WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# %%
# Crear el gráfico de barras usando seaborn
plt.figure(figsize=(10,6))
sns.countplot(x='Sentiment', data=data)
plt.title('Product Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')

# Mostrar el gráfico
plt.show()

# %%
data['Summary'] = data['Summary'].str.replace(r'[^\w\s]', '', regex=True)
data['Summary'] = data['Summary'].fillna('')
#crear un nuevo dataframe con solo las columnas 'Summary' y 'Sentiment'

clean_data = data[['Summary', 'Sentiment']]
print(clean_data.head())


# %%
# Dividir el dataframe en Summary y Sentiment en 80% y 20%
train_data, test_data = train_test_split(data, test_size=0.2, random_state=7)

# Creacion de bolsa de palabras
vectorizer = CountVectorizer()

x = vectorizer.fit_transform(train_data['Summary'])
x_test = vectorizer.transform(test_data['Summary'])

y = train_data['Sentiment']
y_test = test_data['Sentiment']
    

print(f'Tamaño de la bolsa de palabras: {len(vectorizer.get_feature_names_out())}')

# Pruebas para la precisión del modelo
model = LogisticRegression(max_iter=2500) # Establecemos un maximo para iterar, asi no colapsaba la terminal
model.fit(x, y)
y_pred = model.predict(x_test)

# Matriz 
print('Matriz de Confusión')
matriz = confusion_matrix(y_test, y_pred)
print(matriz)

# Reporte 
reporte = classification_report(y_test, y_pred)
print(reporte)



