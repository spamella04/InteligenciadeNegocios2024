import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import gdown

nltk.download('stopwords')
pd.set_option('display.max_columns', 10)


#Cargar el archivo y explorar los datos
url = 'https://drive.google.com/uc?id=1nONpng5xOXNr2kK-bdtLL4RsaingezVK'  
dataset = 'Reviews.csv'
gdown.download(url, dataset, quiet=False)
df = pd.read_csv(dataset)

print(df.head())
print(df.info())
print(df.describe())

#Grafico con la cantidad de scores que tiene el dataset
score_count = df['Score'].value_counts().sort_index()
score_count.plot(kind='bar', color='skyblue', legend=False, width=1) 

plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Cantidad de puntajes')
plt.xticks(rotation=0)
plt.show()


#Representacion de la frecuencia de las palabras en la columna Summary de la reseña de los usuarios, a traves de un wordcloud
df['Summary'] = df['Summary'].astype(str)
text = ' '.join(df['Summary'])
stop_words = set(stopwords.words('english'))
text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.show()

#Crear una nueva columna Sentiment que segun el valor de la columna Score sera positivo o negativo 
#(Si es <3 es -1, si es >3 es 1 y de ser 3 se elimina por neutro)
score_freq = df['Score'].value_counts()
print("\nTabla de frecuencia de score antes de eliminar los neutros:")
print(score_freq)

df = df[df['Score'] != 3]
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else -1)

score_freq = df['Score'].value_counts()
print("\nTabla de frecuencia de score despues de eliminar los neutros:")
print(score_freq)

print("\nEstado actual de los valores en el dataframe:")
print(df[['Score','Summary','Sentiment']].head())

#Crear dos datasets, uno para las reseñas positivas y uno para las negativas
df_positive = df[df['Sentiment'] == 1]
df_negative = df[df['Sentiment'] == -1]

#Worldcloud positivo y negativo

text_positve = ' '.join(df_positive['Summary'])
stop_words = set(stopwords.words('english'))
text_positve = ' '.join([word for word in text_positve.split() if word.lower() not in stop_words])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_positve)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')  
plt.show()

text_negative= ' '.join(df_negative['Summary'])
stop_words = set(stopwords.words('english'))
text_negative = ' '.join([word for word in text_negative.split() if word.lower() not in stop_words])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_negative)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')  
plt.show()

#Grafico del conteo de reseñas positivas y negativas
sentiment_count = df['Sentiment'].value_counts().sort_index()
sentiment_count.plot(kind='bar', color='skyblue', legend=False, width=0.9) 

plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Cantidad de sentiment en el dataset')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)
plt.show()

#Limpieza y division del dataframe
df_cleaned= df[['Summary','Sentiment']]
df_cleaned['Summary'] = df_cleaned['Summary'].str.replace(r'[^\w\s]', '', regex=True)
print("\nDataframe limpio:")
print(df_cleaned.head())


#Creasion del modelo de bolsa de palabras
#Toma un poco de tiempo en cargar
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(train_df['Summary'])
x_test = vectorizer.transform(test_df['Summary'])

y_train = train_df['Sentiment']
y_test = test_df['Sentiment']

model = LogisticRegression(max_iter=300, solver='saga')
model.fit(x_train, y_train)

predictions = model.predict(x_test)

conf_matrix = confusion_matrix(y_test, predictions)

print("\nMatriz de confusion:")
print(conf_matrix)

print("\nInforme de clasificacion:")
print(classification_report(y_test, predictions))