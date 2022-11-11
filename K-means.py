import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('Mall_Customers.csv')

# first 5 rows in the dataframe
customer_data.head()

# finding the number of rows and columns
customer_data.shape

# getting some informations about the dataset
customer_data.info()

# checking for missing values
customer_data.isnull().sum()

# Choosing the Annual Income Column & Spending Score column
X = customer_data.iloc[:,[3,4]].values
a = customer_data.iloc[:,[3]].values
b = customer_data.iloc[:,[4]].values

print(X)

plt.scatter(a, b, color='black')  

plt.xlabel('anual income')  
plt.ylabel('spending score')  
plt.title('Data set')  
plt.legend()  
plt.show()  

# Choosing the number of clusters
# finding wcss value for different number of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)
  
  
 # plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Training the k-Means Clustering Model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)





# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='blue', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='green', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='orange', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='purple', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

#calculate the number of data points in each cluster
cat = [0,0,0,0,0]
assignationArray = kmeans.labels_
for i in range(200) :
    if assignationArray[i]==0 :
        cat[0]= cat[0] + 1
    elif assignationArray[i]==1 :
        cat[1] = cat[1] + 1
    elif assignationArray[i]==2 :
        cat[2] = cat[2] + 1
    elif assignationArray[i]==3 :
        cat[3] = cat[3] + 1
    else:
        cat[4] = cat[4] + 1
print(assignationArray)
print(cat)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:  
customers = 'Economical', 'Medium spenders', 'Luxurious', 'Risk of churn','Low spenders'  
explode = (0, 0, 0, 0, 0)  # it "explode" the 1st slice   
  
fig1, ax1 = plt.subplots()  
ax1.pie(cat, explode=explode, labels=customers, autopct='%1.1f%%',  
        shadow=True, startangle=90)  
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.  
  
plt.show()  
