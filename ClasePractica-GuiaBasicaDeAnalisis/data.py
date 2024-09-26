#Hecho por Megan Bougle
#pip install gdown para que le funcione la leida del csv
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import gdown

url = 'https://drive.google.com/uc?id=1nONpng5xOXNr2kK-bdtLL4RsaingezVK'
Reviews = 'Reviews.csv'
gdown.download(url, Reviews, quiet=False)
df = pd.read_csv(Reviews)

print(df.head())

#Grafico 1 sobre el Score o Calificacion del producto
plt.figure(figsize=(10, 6))
plt.hist(df['Score'], bins=5, color='pink', alpha=0.7)
plt.title("Calificación del Producto", fontsize=16)
plt.xlabel("Score", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()

#configurando el bolson de stopwords/palabras que no aportan 
nltk.download('stopwords')
bolson = set(stopwords.words('english'))
#aqui se esta excluyendo estas palabras porque sino, la nube de palabras negativas se llan de palabras que se consideran "positivas" 
bolson.update(['good', 'great'])

#profe esta nube tarda en cargar pero si carga
reviews=" ".join(resenia for resenia in df['Summary'].astype(str))
nube = WordCloud(stopwords=bolson, background_color="black", max_words=100).generate(reviews)
plt.figure(figsize=(10, 6))
plt.imshow(nube, interpolation='bilinear')
plt.axis("off")
plt.show()

#Ahora se procede a clasificar los tweets en positivos, negativos y neutros, los neutros se eliminan
df=df[df['Score']!=3]
def clasificacion(tweet):
    if tweet>3:
        return +1
    else:
        return -1
df['sentimiento']=df['Score'].apply(clasificacion)
print(df.head())

#Nubes para la palabras de los tweets poitivos y negativos previamente clasificados
positivos=" ".join(resenia for resenia in df[df['sentimiento']==1]['Summary'].astype(str))
negativos=" ".join(resenia for resenia in df[df['sentimiento']==-1]['Summary' ].astype(str))
nubePositiva=WordCloud(stopwords=bolson, background_color="black", max_words=100 ).generate(positivos)
nubeNegativa=WordCloud(stopwords=bolson, background_color="black", max_words=100).generate(negativos)
plt.figure(figsize=(10,6))
plt.title("Palabras en Resenias Positivas", fontsize=16)
plt.imshow(nubePositiva, interpolation='bilinear')
plt.axis("off")

plt.figure(figsize=(10,6))
plt.title("Palabras en Resenias Negativas", fontsize=16) 
plt.imshow(nubeNegativa, interpolation='bilinear')
plt.axis("off")
plt.show()

#Grafico 2
plt.figure(figsize=(10, 6))
plt.hist(df['sentimiento'].replace({1: 'positive', -1: 'negative'}), bins=5, color='red', alpha=0.7)
plt.title("Sentimiento del Producto", fontsize=16)
plt.xlabel("Sentiment", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()

print(df.head())

#Procedemos a limpiar las puncticiones de la columna summary
df['Summary'] = df['Summary'].str.replace('[^\w\s]', '', regex=True)
print(df['Summary'])
#Nuevo df
newdf = df[['Summary', 'sentimiento']]
print(newdf.head())

newdf=newdf.dropna(subset=['Summary'])

#Aqui se hace una divison del dataset, teniendo encuenta que el 80 porciento sera para entrenar y el restante para las pruebas
train_df, test_df = train_test_split(newdf, test_size=0.2, random_state=42)

#Profe todo esto tambien tarda en cargar un rato
#Se procede a vectorizar los datos
vectorizador = CountVectorizer()
x_train = vectorizador.fit_transform(train_df['Summary'])
x_test = vectorizador.transform(test_df['Summary'])

y_train = train_df['sentimiento']
y_test = test_df['sentimiento']

model = LogisticRegression(max_iter=500, solver='saga')
model.fit(x_train, y_train)
predicciones = model.predict(x_test )
matriz = confusion_matrix(y_test, predicciones)

print("\nMatriz de confusion:")
print(matriz)

print("\nInforme de clasificacion:")
print(classification_report(y_test, predicciones))