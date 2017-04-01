import numpy as np
import pandas
from sklearn.cluster import KMeans
import pylab as plt
from scipy.spatial.distance import cdist, pdist

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

def plotElbow():
    print("Inside Plot elbow");
    global input_file
    features = input_file[['DepTime', 'CRSDepTime', 'ArrTime']]

    k = range(1, 51)

    clusters = [KMeans(n_clusters=c, init='k-means++').fit(features) for c in k]
    centr_lst = [cc.cluster_centers_ for cc in clusters]

    k_distance = [cdist(features, cent, 'euclidean') for cent in centr_lst]
    clust_indx = [np.argmin(kd, axis=1) for kd in k_distance]
    distances = [np.min(kd, axis=1) for kd in k_distance]
    avg_within = [np.sum(dist) / features.shape[0] for dist in distances]

    with_in_sum_square = [np.sum(dist ** 2) for dist in distances]
    to_sum_square = np.sum(pdist(features) ** 2) / features.shape[0]
    bet_sum_square = to_sum_square - with_in_sum_square

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

plotElbow()