
import numpy as np
import pandas as pd
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

np.seterr(divide='ignore', invalid='ignore')

'''
random centroids:
    receives an array nArr and a constant k.
    returns an array of k arrays formed by floats in  the range of the given array. 
    the amount of floats in each internal array is euqal to the number of columns in nArr
'''
def randCenter(nArr, k):
    coordArr = []
    for i in range(k):
        coordArr.append(np.random.uniform(np.amin(nArr), np.amax(nArr), size = nArr.shape[1]))
    coordArr = np.array(coordArr)
    return coordArr





'''formatting the data'''
cols = list(range(72))
data = pd.read_csv('C:\\Users\\VictorHugo\\Desktop\\leukemia_new.csv', names = cols)

data = pd.DataFrame.transpose(data)

'''saving the labels'''
y = data[0]
'''cutting them off the training data'''
data = data.loc[:,1:]

'''number of clusters and iterations desired'''
#k = int(input("\nInsira o K desejado, entre 2 e 3: \t"))
k = 0
while ((k != 2) and (k != 3)):
    k = int(input('insira 2 ou 3 para escolher o número de clusters:\t'))
maxIterations = 1


'''initializes variables'''
actualMaxSize = 0
iterations = 0
protoClusterDict = {}
clusLengths = []
clustersOld = np.array([[] for i in range(k)])

while iterations < maxIterations:
    '''initial centroids'''
    #centersOld = randCenter(data.values, k)
    if k == 2:
        centersOld = np.array([data.loc[0,:], data.loc[21,:]])
    elif k == 3:
        centersOld = np.array([data.loc[0,:], data.loc[20,:],data.loc[30,:]])
    centers = centersOld
    while True:
        '''clears clusters'''
        clusters = [[] for i in range(k)]
        '''for all samples'''
        for i in range(len(y)):
            '''pick one sample'''
            amostraAtual = data.iloc[i,:]
            distances = []
            closest = []
            '''chooses the closest cluster to the sample'''
            for j in range(len(centers)):
                '''finds the distance'''
                dist_ponto_centroide = distance.euclidean(amostraAtual, centers[j])
                '''adds it to the list of distances'''
                distances.append([dist_ponto_centroide])
                '''keeps the smallest'''
                if distances[j] == min(distances):
                    closest = centers[j] 
                    closest_index = j
                    
            '''adds current sample to corresponding cluster'''
            clusters[closest_index].append(amostraAtual)
        
       
        
        '''transform actual clusters in array for centroid calculations'''
        showClusters = clusters
        for i in range(len(clusters)):
            clusters[i] = np.array(clusters[i])
            for j in range(len(clusters[i])):
                clusters[i][j] = np.array(clusters[i][j])
        clusters = np.array(clusters)
        '''saves old centroids'''
        centersOld = centers
        
        '''create updated ones'''
        centers = []
        for i in range(len(clusters)):
            if clusters[i].size == 0:
                centers.append(centersOld[i])
            else:
                centers.append(clusters[i].mean(0))
                ##print(clusters[i].mean(0))
        centers = np.array(centers)
        '''if it converged, breaks'''
        equals = 1
        if (clusters.shape == clustersOld.shape):
            for i in range(len(clusters)):
                if not np.array_equal(clusters[i], clustersOld[i]):
                    equals = 0
        else:
            equals = 0
        if equals == 1:
            for i in range(len(clusters)):
                clusLengths.append(len(clusters[i]))
            #print('max interno: ', max(clusLengths))
            actualMaxSize = max(clusLengths)
            protoClusterDict[actualMaxSize] = showClusters
            break
        '''saves old clusters and iterates until local convergence'''
        clustersOld = clusters 
        
    '''goes to the next iteration of k-means'''
    iterations = iterations + 1
chosenClustersKey = min(list(protoClusterDict.keys()))
showClusters = protoClusterDict.get(chosenClustersKey)
'''
Searches for the labels of each clustered sample and pairs them with the clusters' indexes

mapping = []
for i in range(len(clusters)):
    for j in range(len(clusters[i])):
        amostraAtual = clusters[i][j]
        for k in range(len(y)):
            if np.array_equal(data.values[k], amostraAtual):
                mapping.append((int(y[k]), i))


print('\n\nmapping:\n')
print(mapping)
'''

'''
Formats clusters for plotting
'''
clusLengths = []
for i in range(len(showClusters)):
    clusLengths.append(len(showClusters[i]))

clusList = []
for i in range(len(clusLengths)):
    if clusLengths[i] != 0:
        aux = np.full((clusLengths[i],1), i)
        showClusters[i] = np.hstack((aux, showClusters[i]))
        clusList.append(showClusters[i])

clusList = tuple(clusList)

showClusters = np.vstack(clusList)     

'''
separates labels and data of clusterized data
'''
showClusters = pd.DataFrame(showClusters)
t = showClusters[0]
X = showClusters.loc[:,1:]


#showClusters = showClusters.loc[:,1:]
'''plotting clustered data
   the actual plotting part should be updated to increase flexibility
   in case of different numbers of clusters
'''


X_norm = (X - X.min())/(X.max() - X.min())
pca = PCA(n_components=k)
transformed = pd.DataFrame(pca.fit_transform(X_norm))
plt.scatter(transformed[t==0][0], transformed[t==0][1], label=('Class 0', clusLengths[0]), c='red')
plt.scatter(transformed[t==1][0], transformed[t==1][1], label=('Class 1', clusLengths[1]), c='blue')
'''if the uses chooses k = 3'''
if k == 3:
    plt.scatter(transformed[t==2][0], transformed[t==2][1], label=('Class 2: ', clusLengths[2]), c='lightgreen')
plt.legend()
plt.show()

'''plotting original data, with only 2 groups, since thats what happens in the actual csv
    this should be changed for more flexibility'''
Z = data

Z_norm = (Z - Z.min())/(Z.max() - Z.min())
pca = PCA(n_components=2)
transformed_og = pd.DataFrame(pca.fit_transform(Z_norm))
plt.scatter(transformed_og[y==1][0], transformed_og[y==1][1], label='ALL', c='red')
plt.scatter(transformed_og[y==2][0], transformed_og[y==2][1], label='AML', c='blue')
plt.legend()
plt.show()