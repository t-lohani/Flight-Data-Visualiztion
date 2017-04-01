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
from scipy.spatial.distance import cdist, pdist

app = Flask(__name__)

input_file = pandas.read_csv('Flight_Data_2008.csv', low_memory=False)
input_file = input_file.fillna(0)

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

labels = []
random_samples = []
adaptive_samples = []

sample_size = 200
n_components = 200
n_features = 1000
lsa_clusters = 4

def clustering():
    # Clustering the data
    print("Inside clustering");
    global data
    global input_file
    features = input_file[['DepTime', 'CRSDepTime', 'ArrTime']]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features)
    labels = kmeans.labels_
    input_file['kcluster'] = pandas.Series(labels)

def random_sampling():
    # Randomized sampling
    print("Inside randomized sampling");
    global data
    global input_file
    global random_samples
    global sample_size
    features = input_file[['DepTime', 'CRSDepTime', 'ArrTime']]
    data = np.array(features)
    rnd = random.sample(range(len(input_file)), sample_size)
    for j in rnd:
        random_samples.append(data[j])

def adaptive_sampling():
    # Adaptive sampling
    print("Inside adaptive sampling");
    global input_file
    global adaptive_samples
    size_sample = sample_size

    kcluster_0 = input_file[input_file['kcluster'] == 0]
    kcluster_1 = input_file[input_file['kcluster'] == 1]
    kcluster_2 = input_file[input_file['kcluster'] == 2]

    size_kcluster_0 = len(kcluster_0) * size_sample / len(input_file)
    size_kcluster_1 = len(kcluster_1) * size_sample / len(input_file)
    size_kcluster_2 = len(kcluster_2) * size_sample / len(input_file)

    sample_cluster0 = kcluster_0.ix[random.sample(list(kcluster_0.index), int(size_kcluster_0))]
    sample_cluster1 = kcluster_1.ix[random.sample(list(kcluster_1.index), int(size_kcluster_1))]
    sample_cluster2 = kcluster_2.ix[random.sample(list(kcluster_2.index), int(size_kcluster_2))]

    adaptive_samples = pandas.concat([sample_cluster0, sample_cluster1, sample_cluster2])


def plotElbow():
    print("Inside Plot elbow");
    global input_file
    features = input_file[['DepTime', 'CRSDepTime', 'ArrTime']]

    k = range(1, 11)

    clusters = [KMeans(n_clusters=c, init='k-means++').fit(features) for c in k]
    centr_lst = [cc.cluster_centers_ for cc in clusters]

    k_distance = [cdist(features, cent, 'euclidean') for cent in centr_lst]
    # clust_indx = [np.argmin(kd, axis=1) for kd in k_distance]
    distances = [np.min(kd, axis=1) for kd in k_distance]
    avg_within = [np.sum(dist) / features.shape[0] for dist in distances]

    # with_in_sum_square = [np.sum(dist ** 2) for dist in distances]
    # to_sum_square = np.sum(pdist(features) ** 2) / features.shape[0]
    # bet_sum_square = to_sum_square - with_in_sum_square

    kidx = 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k, avg_within, 'g*-')
    ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    print("End of plotElbow")
    plt.show()
    print("End of program");

clustering()
random_sampling()
adaptive_sampling()
plotElbow()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/pca_random")
def pca_random():
    print("Inside PCA Random");
    # PCA reduction with random sampling
    data_col = []
    try:
        global random_samples
        pca_data = PCA(n_components=2)
        X = random_samples
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
        pca_variance = pca_data.explained_variance_ratio_
        data_col['variance'] = pandas.DataFrame(pca_variance)[0]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/pca_adaptive')
def pca_adaptive():
    print("Inside PCA Adaptive");
    # PCA reduction with adaptive sampling
    data_col = []
    try:
        global adaptive_samples
        pca_data = PCA(n_components=2)
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        pca_data.fit(X)
        X = pca_data.transform(X)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
        pca_variance = pca_data.explained_variance_ratio_
        data_col['variance'] = pandas.DataFrame(pca_variance)[0]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/mds_euclidean_random')
def mds_euclidean_random():
    print("Inside MDS Random using Euclidean Distance")
    # MSD reduction with random sampling and using euclidean distance
    data_col = []
    try:
        global random_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/mds_euclidean_adaptive')
def mds_euclidean_adaptive():
    print("Inside MDS Adaptive using Euclidean Distance")
    # MSD reduction with adaptive sampling and using euclidean distance
    data_col = []
    try:
        global adaptive_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        similarity = pairwise_distances(X, metric='euclidean')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/mds_correlation_random')
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
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/mds_correlation_adaptive')
def mds_correlation_adaptive():
    print("Inside MDS Adaptive using Correlation")
    # MSD reduction with adaptive sampling and using correlation
    data_col = []
    try:
        global adaptive_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        similarity = pairwise_distances(X, metric='correlation')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

if __name__ == "__main__":
    app.run('localhost', '5050')