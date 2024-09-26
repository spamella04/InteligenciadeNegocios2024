import pandas as pd 
import nltk as nl
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')  # Descargar stopwords

url = 'https://media.githubusercontent.com/media/cdtoruno/review-csv/refs/heads/main/Reviews.csv'
data = pd.read_csv(url)

print(data.head())


# Visualización de la variable Score con Plotly express
fig = px.histogram(data, x='Score')
fig.show()

# Creación de wordclouds en las reseñas
if 'Text' in data.columns:
    # Una sola cadena de texto de todas las reseñas 
    all_reviews = " ".join(review for review in data['Text'])

    # Remover stopwords
    stop_words = set(stopwords.words('english'))
    
    # Generar la nube de palabras
    wordcloud = WordCloud(stopwords=stop_words, background_color="black", colormap="viridis", width=800, height=400).generate(all_reviews)
    
    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Clasificación de las reseñas por Score, positivas = +1, negativas = -1
    # Score > 3 = +1, Score < 3 = -1 y Score = 3 = Neutro (Se elimina)
    data['Sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0)

    data = data[data['Sentiment'] != 0]  # Estas son las reseñas neutrales
    print(data.head())

    # Crear una nube de palabras para las reseñas positivas
    positive_reviews = " ".join(review for review in data[data['Sentiment'] == 1]['Text'])
    wordcloud = WordCloud(stopwords=stop_words, background_color="black", colormap="viridis", width=800, height=400).generate(positive_reviews)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Crear una nube de palabras para las reseñas negativas
    negative_reviews = " ".join(review for review in data[data['Sentiment'] == -1]['Text'])
    wordcloud = WordCloud(stopwords=stop_words, background_color="black", colormap="viridis", width=800, height=400).generate(negative_reviews)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Distribución de las reseñas del dataset
    fig = px.histogram(data, 
                       x=data['Sentiment'].replace({1: 'positive', -1: 'negative'}), 
                       title='Product Sentiment', 
                       color=data['Sentiment'].replace({1: 'positive', -1: 'negative'}), 
                       labels={'x': 'Sentiment'},
                       category_orders={'x': ['positive', 'negative']})
    fig.show()

    # Construir el modelo con los pasos para las predicciones 
    # Remover todas las puntuaciones de Summary
    data['Summary'] = data['Summary'].str.replace(r'[^\w\s]', '', regex=True)
    data['Summary'] = data['Summary'].fillna('')

    # Nuevo dataset
    data = data[['Summary', 'Sentiment']]
    print(data.head())

    # Dividir el dataframe en Summary y Sentiment en 80% y 20%
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

    # Creacion de bolsa de palabras
    vectorizer = CountVectorizer()

    x = vectorizer.fit_transform(train_data['Summary'])
    x_test = vectorizer.transform(test_data['Summary'])

    y = train_data['Sentiment']
    y_test = test_data['Sentiment']
    

    print(f'Tamaño de la bolsa de palabras: {len(vectorizer.get_feature_names_out())}')

    # Pruebas para la precisión del modelo
    model = LogisticRegression(max_iter=5000) # Si no se establecia un maximo para iterar, colapsaba la terminal
    model.fit(x, y)
    y_pred = model.predict(x_test)

    # Matriz 
    matriz = confusion_matrix(y_test, y_pred)
    print('\n Matriz de confusión \n')
    print(matriz)

    # Reporte 
    reporte = classification_report(y_test, y_pred)
    print(reporte)
