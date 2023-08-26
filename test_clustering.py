#codigo para la prueba de salidas

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#metricas y metodos de validacion
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#cargar csv
df_original = pd.read_csv('machine-readable-business-employment-data-dec-2021-quarter.csv')
df = df_original
#quitamos las dos ultimas
size_data = df.shape[1]
df = df.drop(df.columns[[size_data-1,size_data-2]], axis=1) 
df = df.drop(df.columns[[0]], axis=1) 



#convertimos de cat a num, las columas 
df = pd.get_dummies(df, columns=['Suppressed'])

cats = df["STATUS"].unique()
print(cats)
df['STATUS'].replace([cats[0], cats[1], cats[2]],
                        [0, 1, 2], inplace=True)
cats = df["UNITS"].unique()
print(cats)
df['UNITS'].replace([cats[0], cats[1]],
                        [0, 1], inplace=True)
cats = df["Subject"].unique()
print(cats)
df['Subject'].replace([cats[0]],
                        [0], inplace=True)
cats = df["Group"].unique()
print(cats)
df['Group'].replace([cats[0], cats[1], cats[2], cats[3], cats[4]],
                        [0, 1, 2, 3, 4], inplace=True)
cats = df["Series_title_1"].unique()
print(cats)
df['Series_title_1'].replace([cats[0], cats[1], cats[2]],
                        [0, 1, 2], inplace=True)
cats = df["Series_title_3"].unique()
print(cats)
df['Series_title_3'].replace([cats[0], cats[1], cats[2]],
                        [0, 1, 2], inplace=True)

#dos rasgos que podrian no tener informacion
label_encoder = LabelEncoder()
cats = df["Series_title_2"].unique()
y = label_encoder.fit_transform(cats)
#df['Series_title_2'] = y
#ver cuantos periodos diferentes hay
cat_p = df["Period"].unique()

#creamos el dataset con las caracteristicas que usaremos para el kmeans
df_cluster = df[["Suppressed_Y","STATUS", "UNITS", "Magnitude", "Subject", "Group", "Series_title_1", "Series_title_3"]]
#lo que sea nan sera 0, 
df_cluster = df_cluster.fillna(0)
#calcular promedio de Data_value y sustituir los val perdidos
#data_value = df_cluster['Data_value']
#data_value.replace(to_replace = 0, value = data_value.mean(), inplace=True)

#vemos si ya no hay mas val perdidos
df_cluster.isnull().sum() 

#cremamos la matriz de datos
X = np.array(df_cluster)

#aplicar PCA, feature extraction o algo asi
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
pca.score(X)

#escalando los datos, para ver si hay algun cambio
scaler = StandardScaler()
X_pca_std = scaler.fit(X_pca).transform(X_pca)

#X_pca = X_pca_std

#====== buscando el mejor k del cluster
#el metodo Elbow Curve
#For each of the K values, 
#we calculate average distances to the centroid across all data points

k = range(1, 11)
Sum_of_distances = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_pca)
    Sum_of_distances.append(kmeans.inertia_)
    
plt.plot(k, Sum_of_distances)
plt.xlabel("Values of K") 
plt.ylabel("Sum of squared distances/Inertia") 
plt.title("Elbow Method For Optimal k")
plt.show()

#=== Silhouette analysis
# range_n_clusters = range(2, 11)
# silhouette_avg = []
# for num_clusters in range_n_clusters: 
#     # initialise kmeans
#     kmeans = KMeans(n_clusters=num_clusters)
#     kmeans.fit(X_pca)
#     cluster_labels = kmeans.labels_
 
#     #silhouette score
#     silhouette_avg.append(silhouette_score(X_pca, cluster_labels))
    
# plt.plot(range_n_clusters,silhouette_avg)
# plt.xlabel("‘Values of K’") 
# plt.ylabel("‘Silhouette score’") 
# plt.title("‘Silhouette analysis For Optimal k’")
# plt.show()

kmeans = KMeans(n_clusters=3).fit(X_pca)
#los centroides
centroids = kmeans.cluster_centers_

#recuperamos las etiquetas calculadas por el kmeans
targets = kmeans.predict(X_pca)
df_cluster['target'] = targets
print(df_cluster.target.value_counts())

colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in targets:
    asignar.append(colores[row])

fig = plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=asignar, s=2)
plt.show()


#=========== APLICAMOS CLASIFICACION
full_data = df_cluster.to_numpy()
data = full_data[:,0:-1]
target = full_data[:,-1]

#metodo de validacion, de hold-out
data_train, data_test, t_train, t_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=1)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data_train, t_train)
prediction = model.predict(data_test)
print("\nBA: ", balanced_accuracy_score(t_test, prediction))
print("Accuracy: ", metrics.accuracy_score(t_test, prediction))
print("F1: ", metrics.f1_score(t_test, prediction, average='macro'))
    

#ahora hacemos regresion con el Data_value y el tiempo
df = df.fillna(0)
data_value = df['Data_value']
data_value.replace(to_replace = 0, value = data_value.mean(), inplace=True)

x = np.array(df[["Data_value"]])
y = np.array(df[["Period"]])

plt.scatter(x,y)
plt.xlabel("Data value")
plt.ylabel("Period")
plt.show()

data_train, data_test, t_train, t_test = train_test_split(x, y, test_size=0.3, stratify=target, random_state=1)
lr = LinearRegression()
lr.fit(data_train, t_train)
y_pred = lr.predict(data_test)

#graficamos la predicion de la regresion
plt.scatter(data_test, t_test)
plt.plot(data_test, y_pred, color='red', linewidth=2)
plt.title("Regresión")
plt.xlabel("Data value")
plt.ylabel("Period")
plt.show()








#========================
# #con k fold estratificado
# kf = KFold(n_splits=10,shuffle=True)
# predictions = []
# true_test_targets = []
# fold_accu = []
# fold_BA = []
# fold_f1score = []
    
# for train_index, test_index in kf.split(data):
#     X_train, X_test = data[train_index], data[test_index]
#     y_train, y_test = target[train_index], target[test_index]
#     true_test_targets.append(y_test)
        
#     #FASE DE APRENDIZAJE
#     #MLP
#     #model = MLPClassifier(max_iter=500)
        
#     #SVM, mejores: sigmoid=95.5 y rbf=82
#     #model = svm.SVC(kernel='sigmoid')
        
#     #naive bayes
#     #model = GaussianNB()
        
#     #tree, el mejor es el random forest
#     #model = tree.DecisionTreeClassifier()
#     #model = RandomForestClassifier()
        
#     #LogisticRegression
#     #model = LogisticRegression(max_iter=1000)
        
#     #KNN, con k=5
#     model = KNeighborsClassifier(n_neighbors=1)
    
#     #para acelerar via GPU
#     #with config_context(target_offload="gpu:0"):
#     model.fit(X_train, y_train)
        
#     #FASE DE CLASIFICACION
#     prediction = model.predict(X_test)
#     predictions.append(prediction)
#     #calculamos el accuracy por cada fold
#     #fold_accu.append(sum(prediction==y_test)/len(y_test))
#     fold_accu.append(metrics.accuracy_score(y_test,prediction))
#     fold_BA.append(balanced_accuracy_score(y_test, prediction))
#     fold_f1score.append(metrics.f1_score(y_test,prediction,average='macro'))
    
# #aplicar metricas de desempeño. Accuracy
# accuracy_model = round(sum(fold_accu) / len(fold_accu), 4)
# BA_model = round(sum(fold_BA) / len(fold_BA), 4)
# f1_score_model = round(sum(fold_f1score) / len(fold_f1score), 4)
# print("\nAccuracy: ", accuracy_model)
# print("BA: ", BA_model)
# print("F1 Score: ", f1_score_model)




