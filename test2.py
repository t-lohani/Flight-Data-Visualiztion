import numpy as np

data = [[1,2,-3,4], [5,16,7,8], [29,10,11,12], [13,4,-15,16]]

def generate_eigenValues(data):
    centered_matrix = data - np.mean(data, axis=0)
    cov = np.dot(centered_matrix.T, centered_matrix)
    eig_values, eig_vectors = np.linalg.eig(cov)
    print(eig_values)
    print(eig_vectors)
    return eig_values, eig_vectors

generate_eigenValues(data)