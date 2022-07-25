"""
https://www.kaggle.com/code/kanncaa1/machine-learning-tutorial-for-beginners/notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt


data = pd.read_csv("column_2C_weka.csv")
#print(plt.style.available)
plt.style.use("ggplot")

#print(data.head())
#print(data.info())
#print(data.describe())


color_list = ["red" if i=="Abnormal" else "green" for i in data.loc[:, "class"]]
pd.plotting.scatter_matrix(data.loc[:,  data.columns != "class"],
                                        c = color_list,
                                        figsize= [15,15],
                                        diagonal="hist",
                                        alpha=0.5,
                                        s = 200,
                                        marker = "-",
                                        edgecolor = "black"
                                        )
plt.show()

import seaborn as sns
sns.countplot(x="class", data=data)
data.loc[:, "class"].value_counts()
plt.show()

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
X, y = data.loc[:, data.columns != "class"], data.loc[:, "class"]
knn.fit(X, y)

prediction = knn.predict(X)
print("Prediction: {}".format(prediction))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
X, y = data.loc[:, data.columns != 'class'], data.loc[:,'class']
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print('With KNN (K=3) accuracy is: ',knn.score(X_test, y_test))

# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(X_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(X_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(X_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

#Rergression
# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data["class"] == "Abnormal"]
X = np.array(data1.loc[:, "pelvic_incidence"]).reshape(-1, 1)
y = np.array(data1.loc[:, "sacral_slope"]).reshape(-1, 1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=X, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()

# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
predict_space = np.linspace(min(X), max(X)).reshape(-1,1)
reg.fit(X, y)
predicted = reg.predict(predict_space)
print('R^2 score: ',reg.score(X, y))
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=X, y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()

#CV
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg, X, y, cv=k)
print("CV Scores: ", cv_result)
print("CV Scores average:", np.sum(cv_result)/k)

#UNSUPERVISED LEARNING

#KMEANS
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
# KMeans Clustering
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()

#EVALUATING OF CLUSTERING¶
# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()