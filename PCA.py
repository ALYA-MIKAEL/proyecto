import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

dataset = pd.read_csv("smogongenerado.csv")
print(dataset)

#eliminar la columna categoría
dataset.drop(['categoria'],1, inplace=True)
print(dataset)
print()

#eliminar el indice repetido
dataset.drop(dataset.columns[0],axis=1, inplace=True)
print(dataset)
print()
#PCA
pca=PCA(n_components=5)
pca.fit(dataset)
print("ya se alimentó el pca!")

x=pca.transform(dataset)
print(x)
cabeceras = ['PCA1','PCA2','PCA3','PCA4','PCA5']
tablaPCA= pd.DataFrame(data=x, columns=cabeceras)

print(tablaPCA)

#agrupamiento
km=KMeans(2)
cluster=km.fit_predict(tablaPCA)
tablaPCA['categoria'] = cluster
print(tablaPCA)
#generar un csv:
print("CSV generado: smogongeneradoPCA.csv")
tablaPCA.to_csv("smogongeneradoPCA.csv")