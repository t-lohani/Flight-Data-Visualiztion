from __future__ import division
import random
import sys
import numpy as np
import pandas
from sklearn.decomposition import TruncatedSVD
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from flask import Flask, render_template

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

clustering()
random_sampling()
adaptive_sampling()

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

@app.route('/isomap_random')
def isomap_random():
    print("Inside Isomap Random");
    # Isomap reduction with random sampling
    data_col = []
    try:
        global random_samples
        isomap_data = manifold.Isomap(n_components=2)
        X = isomap_data.fit_transform(random_samples)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/isomap_adaptive')
def isomap_adaptive():
    print("Inside Isomap adaptive")
    # Isomap reduction with adaptive sampling
    data_col = []
    try:
        global adaptive_samples
        isomap_data = manifold.Isomap(n_components=2)
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        isomap_data.fit(X)
        X = isomap_data.transform(X)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
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

@app.route('/mds_cosine_random')
def mds_cosine_random():
    print("Inside MDS Random using Cosine Distance")
    # MSD reduction with random sampling and using cosine distance
    data_col = []
    try:
        global random_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        similarity = pairwise_distances(random_samples, metric='cosine')
        X = mds_data.fit_transform(similarity)
        data_col = pandas.DataFrame(X)
        data_col['departure'] = input_file['DepTime'][:sample_size]
        data_col['arrival'] = input_file['ArrTime'][:sample_size]
        data_col['clusterid'] = input_file['kcluster'][:sample_size]
    except:
        e = sys.exc_info()[0]
        print(e)
    return pandas.json.dumps(data_col)

@app.route('/mds_cosine_adaptive')
def mds_cosine_adaptive():
    print("Inside MDS Adaptive using Cosine Distance")
    # MSD reduction with adaptive sampling and using cosine distance
    data_col = []
    try:
        global adaptive_samples
        mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
        X = adaptive_samples[['DepTime', 'CRSDepTime', 'ArrTime']]
        similarity = pairwise_distances(X, metric='cosine')
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

@app.route('/lsa')
def lsa():
    global n_components
    global lsa_clusters
    global n_features
    svd = TruncatedSVD(n_components)
    svd_normalizer = Normalizer(copy=False)
    svd_lsa = make_pipeline(svd, svd_normalizer)
    data_categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    data = fetch_20newsgroups(subset='all', categories=data_categories, shuffle=True, random_state=42)
    svd_vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, stop_words='english', use_idf=True)
    X = svd_vectorizer.fit_transform(data.data)
    X = svd_lsa.fit_transform(X)
    kmeans = KMeans(n_clusters=lsa_clusters, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(X)
    doc_original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_)
    doc_order_centroids = doc_original_space_centroids.argsort()[:, ::-1]

    lsa_data = []
    terms = svd_vectorizer.get_feature_names()
    for i in range(lsa_clusters):
        data = []
        for ind in doc_order_centroids[i, :10]:
            print(terms[ind])
            data.append(terms[ind])
        lsa_data.append(data)
    return pandas.json.dumps(lsa_data)

if __name__ == "__main__":
    app.run('localhost', '5050')