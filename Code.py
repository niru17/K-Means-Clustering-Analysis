#Niranjana Subramanian <UTA ID:1002046305>

import random
import os
import matplotlib.pyplot as plt
import numpy as np


# Range of number of clusters
norange_cluster = range(2, 11)
# Total number of iterations for k-means
total_iteration = 20

# Function to load data from a file
def file_load(file_path):
    try:
        with open(file_path, 'r') as file_open:
            return file_open.readlines()
    except FileNotFoundError as e:
        print(f"Data load error: {e}")
        return None

def main():
    try:
        random.seed(0)

        dfile = input("Enter the file: ")

        # Check if the file exists
        if not os.path.exists(dfile):
            print(f"Error: File '{dfile}' is not found.")
            return

        # Read data from the file
        l_file = file_load(dfile)
        if l_file is None:
            return

        r = []

        # Iterate over different numbers of clusters
        for cluster_no in norange_cluster:
            d, clusters, datap = initial_cluster(cluster_no, l_file)
            centroids = cluster_centroids(cluster_no, d, clusters)
            centroids = update(cluster_no, d, clusters)

            # Perform k-means iterations
            for i in range(1, total_iteration + 1):
                clusters, error = k_means(cluster_no, datap, centroids)

                # Print the error for the 20th iteration
                if i == total_iteration:
                    print(f"For k = {cluster_no} After {total_iteration} iterations: Error = {error:.4f}")
                    r.append(error)

                centroids = cluster_centroids(cluster_no, d, clusters)
                centroids = update(cluster_no, d, clusters)

        # Plotting the results
        plt.plot(list(norange_cluster), r, marker='o', color='blue', linestyle='-')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title(f"K-Means Clustering: Sum of Squared Errors after {total_iteration} Iterations for various number of clusters")
        plt.grid(True)
        plt.show()

    except Exception as c:
        print(f"Error in occurred in main: {c}")



# Function to initialize clusters with distinct data points as centroids
def initial_cluster(cluster_no, l_file):
    try:
        # Extracting data points from the file
        datap = [list(map(float, line.strip().split()[:-1])) for line in l_file]

        # Choose K distinct data points as initial centroids
        centroids = random.sample(datap, cluster_no)

        # Assign each data point to its nearest initial centroid
        clusters = [[] for _ in range(cluster_no)]

        # Assign each data point to its nearest initial centroid
        for p in datap:
            dist = [euclid_dist(p, centroid) for centroid in centroids]
            i = np.argmin(dist)
            clusters[i].append(p)

        return len(datap[0]), clusters, datap
    except Exception as c:
        print(f"Cluster initialization Error: {c}")
        return None, None, None

# Function to calculate Euclidean distance between two points
def euclid_dist(pt1, p2):
    diff = [p1 - p2 for p1, p2 in zip(pt1, p2)]
    sq_diff = [diff ** 2 for diff in diff]
    return sum(sq_diff) ** 0.5

# Function to calculate cluster centroids
def cluster_centroids(cluster_no, d, clusters):
    centroids = [[0] * d for _ in range(cluster_no)]

    try:
        for c_index in range(cluster_no):
            for p_index, point in enumerate(clusters[c_index]):
                for idimension in range(d):
                    centroids[c_index][idimension] += point[idimension]
    except Exception as c:
        print(f"Couldn't calculate Cluster Centroids: {c}")

    return centroids

# K-means iteration
def k_means(cluster_no, datap, centroids):
    clusters = [[] for _ in range(cluster_no)]
    toterror = 0

    try:
        for point in datap:
            dist = [euclid_dist(point, centroid) for centroid in centroids]
            i = np.argmin(dist)
            clusters[i].append(point)
            toterror += min(dist)
    except Exception as c:
        print(f"Error found in k-means iteration: {c}")

    return clusters, toterror

# Function to update cluster centroids
def update(numcluster, d, clusters):
    try:
        new = [[0] * d for _ in range(numcluster)]

        for c_i in range(numcluster):
            if len(clusters[c_i]) > 0:
                for p_index in range(len(clusters[c_i])):
                    for dim_i in range(d):
                        new[c_i][dim_i] += clusters[c_i][p_index][dim_i]

        for c_i in range(numcluster):
            if len(clusters[c_i]) > 0:
                for dim_i in range(d):
                    new[c_i][dim_i] /= len(clusters[c_i])
    except Exception as c:
        print(f"Error while updating centroids: {c}")

    return new

if __name__ == "__main__":
    main()
