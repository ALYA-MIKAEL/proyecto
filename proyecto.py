import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

dataset = pd.read_csv("smogon.csv")
vec = TfidfVectorizer(ngram_range=(1,1))
x = vec.fit_transform(dataset["Pokemon"])

print("La cantidad de tokens es:")
print(len(vec.vocabulary_))
print()

print("El vocabulario es:")
print(vec.vocabulary_.keys())
print()

print("La matriz de frecuencias es:")
print(x.toarray())

cabeceras = sorted(vec.vocabulary_.keys())
TablaFrecuencias = pd.DataFrame(data=x.toarray(), columns=cabeceras)

#Agrupamiento
km = KMeans(3)
cluster = km.fit_predict(TablaFrecuencias)
TablaFrecuencias['categoria'] = cluster

print(TablaFrecuencias)

#generar un csv:
print("CSV generado: smogongenerado.csv")
TablaFrecuencias.to_csv("smogongenerado.csv")