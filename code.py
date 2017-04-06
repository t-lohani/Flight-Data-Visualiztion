from __future__ import division

import random
import sys

import numpy as np
import pandas
import pylab as plt
from flask import Flask, render_template
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

input_file = pandas.read_csv('Flight_Data_2008.csv', low_memory=False)
orig_file = pandas.read_csv('Flight_Data_2008.csv', low_memory=False)
input_file = input_file.fillna(0)
orig_file = orig_file.fillna(0)

del input_file['UniqueCarrier']
del input_file['TailNum']
del input_file['AirTime']
del input_file['Origin']
del input_file['Dest']
del input_file['TaxiIn']
del input_file['TaxiOut']
del input_file['Cancelled']
del input_file['CancellationCode']
del input_file['Diverted']
del input_file['CarrierDelay']
del input_file['WeatherDelay']
del input_file['NASDelay']
del input_file['SecurityDelay']
del input_file['LateAircraftDelay']

data = []
labels = []
random_samples = []
adaptive_samples = []
loadingVector = {}
sample_size = 1000

columns = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime']
imp_fetures = []

scaler = StandardScaler()
minmaxscaler = MinMaxScaler()
input_file[columns]=scaler.fit_transform(input_file[columns])
# input_file[columns]=minmaxscaler.fit_transform(input_file[columns])

eigen_values = []
eigen_vectors = []

def random_sampling():
    # Randomized sampling
    print("Getting random samples");
    global data
    global input_file
    global random_samples
    global sample_size
    features = input_file[columns]
    data = np.array(features)
    rnd = random.sample(range(len(input_file)), sample_size)
    for j in rnd:
        random_samples.append(data[j])

def clustering():
    # Clustering the data
    print("Clustering data with K = 4");
    global input_file
    features = input_file[columns]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(features)
    labels = kmeans.labels_
    input_file['kcluster'] = pandas.Series(labels)

def adaptive_sampling():
    # Adaptive sampling
    print("Getting adaptive samples");
    global input_file
    global adaptive_samples
    global sample_size

    kcluster_0 = input_file[input_file['kcluster'] == 0]
    kcluster_1 = input_file[input_file['kcluster'] == 1]
    kcluster_2 = input_file[input_file['kcluster'] == 2]
    kcluster_3 = input_file[input_file['kcluster'] == 3]

    size_kcluster_0 = len(kcluster_0) * sample_size / len(input_file)
    size_kcluster_1 = len(kcluster_1) * sample_size / len(input_file)
    size_kcluster_2 = len(kcluster_2) * sample_size / len(input_file)
    size_kcluster_3 = len(kcluster_3) * sample_size / len(input_file)

    sample_cluster0 = kcluster_0.ix[random.sample(list(kcluster_0.index), int(size_kcluster_0))]
    sample_cluster1 = kcluster_1.ix[random.sample(list(kcluster_1.index), int(size_kcluster_1))]
    sample_cluster2 = kcluster_2.ix[random.sample(list(kcluster_2.index), int(size_kcluster_2))]
    sample_cluster3 = kcluster_3.ix[random.sample(list(kcluster_3.index), int(size_kcluster_3))]

    adaptive_samples = pandas.concat([sample_cluster0, sample_cluster1, sample_cluster2, sample_cluster3])

def plotElbow():
    print("Plotting Elbow plot");
    global orig_file
    features = orig_file[columns]

    k = range(1, 11)

    clusters = [KMeans(n_clusters=c, init='k-means++').fit(features) for c in k]
    centr_lst = [cc.cluster_centers_ for cc in clusters]

    k_distance = [cdist(features, cent, 'euclidean') for cent in centr_lst]
    distances = [np.min(kd, axis=1) for kd in k_distance]
    avg_within = [np.sum(dist) / features.shape[0] for dist in distances]

    kidx = 3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k, avg_within, 'g*-')
    ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()

def generate_eig_values(data):
    centered_matrix = data - np.mean(data, axis=0)
    cov = np.dot(centered_matrix.T, centered_matrix)
    eig_values, eig_vectors = np.linalg.eig(cov)

    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    return eig_values/1000, eig_vectors

def plot_intrinsic_dimensionality_pca(data, k):
    # print("Inside plot_intrinsic_dimensionality_pca")
    global loadingVector
    [eigenValues, eigenVectors] = generate_eig_values(data)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    squaredLoadings = []
    ftrCount = len(eigenVectors)
    for ftrId in range(0,ftrCount):
        loadings = 0
        temp = []
        for compId in range(0, k):
            loadings = loadings + eigenVectors[compId][ftrId] * eigenVectors[compId][ftrId]
        loadingVector[columns[ftrId]] = loadings
        squaredLoadings.append(loadings)

    # print("Return Squareloadings")
    print(loadingVector)
    return squaredLoadings

plotElbow()
clustering()
random_sampling()
adaptive_sampling()

squared_loadings = plot_intrinsic_dimensionality_pca(data, 3)
imp_fetures = sorted(range(len(squared_loadings)), key=lambda k: squared_loadings[k], reverse=True)

print(imp_fetures)

@app.route("/")
def index():
    return render_template('task2b.html')

@app.route("/task2c")
def task2():
    return render_template('task2c.html')

@app.route("/task3")
def task3():
    return render_template('task3.html')

@app.route("/get_squareloadings")
def getSquareLoadings():
    global loadingVector
    return pandas.json.dumps(loadingVector)

@app.route("/scree_plot_random")
def scree_plot_random():
    print("Inside scree plot random")
    # Plotting scree plot
    global random_samples
    global eigen_values
    global eigen_vectors
    try:
        [eigen_values, eigen_vectors] = generate_eig_values(random_samples)
    except:
        e = sys.exc_info()[0]
        print(e)

    return pandas.json.dumps(eigen_values)


@app.route("/scree_plot_adaptive")
def scree_plot_adaptive():
    print("Inside scree plot adaptive")
    # Plotting scree plot
    global adaptive_samples
    global eigen_values
    global eigen_vectors
    try:
        [eigen_values, eigen_vectors] = generate_eig_values(adaptive_samples[columns])
    except:
        e = sys.exc_info()[0]
        print(e)

    return pandas.json.dumps(eigen_values)

@app.route("/pca_random")
def pca_random():
    print("Inside PCA Random");
    # PCA reduction with random sampling
    data_col = []
    try:
        global random_samples
        global imp_fetures
        pca_data = PCA(n_components=2)
        X = random_samples
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_col = pandas.DataFrame(X)

        for i in range(0, 2):
            data_col[columns[imp_fetures[i]]] = orig_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = input_file['kcluster'][:sample_size]

    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route("/pca_adaptive")
def pca_adaptive():
    print("Inside PCA Adaptive");
    # PCA reduction with adaptive sampling
    data_col = []

    try:
        global adaptive_samples
        global imp_fetures
        # print(adaptive_samples)

        pca_data = PCA(n_components=2)
        X = adaptive_samples[columns]
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_col = pandas.DataFrame(X)

        for i in range(0, 2):
            data_col[columns[imp_fetures[i]]] = orig_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = np.nan
        x = 0

        for index, row in adaptive_samples.iterrows():
            data_col['clusterid'][x] = row['kcluster']
            x = x + 1

    except:
        e = sys.exc_info()[0]
        print(e)

    return pandas.json.dumps(data_col)


@app.route("/mds_euclidean_random")
def mds_euclidean_random():
    print("Inside MDS Random using Euclidean Distance")
    # MSD reduction with random sampling and using euclidean distance
    data_col = []
    try:
        global random_samples
        global imp_fetures
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)

        for i in range(0, 2):
            data_col[columns[imp_fetures[i]]] = orig_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route("/mds_euclidean_adaptive")
def mds_euclidean_adaptive():
    print("Inside MDS Adaptive using Euclidean Distance")
    # MSD reduction with adaptive sampling and using euclidean distance
    data_col = []
    try:
        global adaptive_samples
        global imp_fetures
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[columns]
        similarity = pairwise_distances(X, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)

        for i in range(0, 2):
            data_col[columns[imp_fetures[i]]] = orig_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = np.nan
        x = 0

        for index, row in adaptive_samples.iterrows():
            data_col['clusterid'][x] = row['kcluster']
            x = x + 1
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route("/mds_correlation_random")
def mds_correlation_random():
    print("Inside MDS Random using Correlation")
    # MSD reduction with random sampling and using Correlation
    data_col = []
    try:
        global random_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)

        for i in range(0, 2):
            data_col[columns[imp_fetures[i]]] = orig_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = input_file['kcluster'][:sample_size]

    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route("/mds_correlation_adaptive")
def mds_correlation_adaptive():
    print("Inside MDS Adaptive using Correlation")
    # MSD reduction with adaptive sampling and using correlation
    data_col = []
    try:
        global adaptive_samples
        global imp_fetures
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[columns]
        similarity = pairwise_distances(X, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)

        for i in range(0, 2):
            data_col[columns[imp_fetures[i]]] = orig_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = np.nan
        x = 0

        for index, row in adaptive_samples.iterrows():
            data_col['clusterid'][x] = row['kcluster']
            x = x + 1

    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route("/scatterplot_matrix_random")
def scatter_matrix_random():
    print("Inside Scatterplot matrix random")
    data_col = pandas.DataFrame()
    try:
        global random_samples
        global imp_fetures

        for i in range(0, 3):
            data_col[columns[imp_fetures[i]]] = input_file[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)

    return pandas.json.dumps(data_col)

@app.route("/scatterplot_matrix_adaptive")
def scatter_matrix_adaptive():
    print("Inside Scatterplot matrix adaptive")
    data_col = pandas.DataFrame()
    try:
        global adaptive_samples
        global imp_fetures

        for i in range(0, 3):
            data_col[columns[imp_fetures[i]]] = adaptive_samples[columns[imp_fetures[i]]][:sample_size]

        data_col['clusterid'] = np.nan

        for index, row in adaptive_samples.iterrows():
            data_col['clusterid'][index] = row['kcluster']
        data_col = data_col.reset_index(drop=True)
    except:
        e = sys.exc_info()[0]
        print(e)

    return pandas.json.dumps(data_col)

if __name__ == "__main__":
    app.run('localhost', '5050')