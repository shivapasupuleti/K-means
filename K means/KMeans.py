import numpy as np

def load_dataset(file_name):
    """
    Load the dataset from the provided file.

    Args:
        file_name (str): Name of the file containing the dataset.

    Returns:
        data (np.ndarray): NumPy array containing the dataset.
        labels (np.ndarray): NumPy array containing the labels (if available).
    """
    try:
        data = np.loadtxt(file_name, delimiter=' ', usecols=range(1, 257))
        labels = np.loadtxt(file_name, delimiter=' ', usecols=[0], dtype=str)
        # print(data)
        # print(labels)
        return data, labels
    except FileNotFoundError:
        print("File not found. Please provide the correct file name.")
    except Exception as e:
        print(f"An error occurred: {e}")

import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score

def compute_distance(point1, point2):
    """
    Compute the Euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point.
        point2 (np.ndarray): Second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)

def initialize_centroids(data, k):
    """
    Initialize centroids randomly from the dataset.

    Args:
        data (np.ndarray): Dataset.
        k (int): Number of clusters.

    Returns:
        np.ndarray: Array of initial centroids.
    """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def  assign_clusters(data, centroids):
    """
    Assign data points to the closest centroid.

    Args:
        data (np.ndarray): Dataset.
        centroids (np.ndarray): Array of centroids.

    Returns:
        np.ndarray: Array of cluster assignments for each data point.
    """
    distances = np.array([[compute_distance(point, centroid) for centroid in centroids] for point in data])
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

def update_centroids(data, cluster_assignments, k):
    """
    Update the centroids based on the current cluster assignments.

    Args:
        data (np.ndarray): Dataset.
        cluster_assignments (np.ndarray): Array of cluster assignments for each data point.
        k (int): Number of clusters.

    Returns:
        np.ndarray: Array of updated centroids.
    # """
    # centroids = np.zeros((k, data.shape[1]))
    # for cluster_id in range(k):
    #     cluster_points = data[cluster_assignments == cluster_id]
    #     if cluster_points.size > 0:
    #         centroids[cluster_id] = np.mean(cluster_points, axis=0)
    # # print(centroids)
    # return centroids
    centroids = np.zeros((k, data.shape[1]))
    for cluster_id in range(k):
        cluster_points = data[cluster_assignments == cluster_id]
        if cluster_points.size > 0:  # Check if there are points in the cluster
            centroids[cluster_id] = np.mean(cluster_points, axis=0)
    return centroids


def kmeans(data, k, max_iterations=100):
    """
    Perform k-means clustering.

    Args:
        data (np.ndarray): Dataset.
        k (int): Number of clusters.
        max_iterations (int, optional): Maximum number of iterations. Default is 100.

    Returns:
        np.ndarray: Array of final centroids.
        np.ndarray: Array of final cluster assignments for each data point.
    """
    centroids = initialize_centroids(data, k)
    previous_centroids = np.zeros(centroids.shape)

    for _ in range(max_iterations):
        cluster_assignments = assign_clusters(data, centroids)
        centroids = update_centroids(data, cluster_assignments, k)

        if np.array_equal(centroids, previous_centroids):
            break

        previous_centroids = centroids.copy()

    return centroids, cluster_assignments

def compute_silhouette_coefficient(data, cluster_assignments, centroids):
    """
    Compute the Silhouette coefficient for a given set of clusters.

    Args:
        data (np.ndarray): Dataset.
        cluster_assignments (np.ndarray): Array of cluster assignments for each data point.
        centroids (np.ndarray): Array of centroids.

    Returns:
        float: Silhouette coefficient.
    """
    n_clusters = len(np.unique(cluster_assignments))
    silhouette_coefficients = []

    for i, point in enumerate(data):
        cluster_id = cluster_assignments[i]
        same_cluster_points = data[cluster_assignments == cluster_id]
        a = np.mean([compute_distance(point, other_point) for other_point in same_cluster_points])

        b_values = []
        for other_cluster_id in range(n_clusters):
            if other_cluster_id != cluster_id:
                other_cluster_points = data[cluster_assignments == other_cluster_id]
                b_values.append(np.mean([compute_distance(point, other_point) for other_point in other_cluster_points]))

        if len(b_values) > 0:
            b = min(b_values)
            if a < b:
                silhouette_coefficients.append(1.0 - a / b)
            elif b < a:
                silhouette_coefficients.append(b / a - 1.0)
            else:
                silhouette_coefficients.append(0.0)

    return np.mean(silhouette_coefficients)


def plot_silhouette(data, labels=None):
    """
    Plot the Silhouette coefficient for different values of k.

    Args:
        data (np.ndarray): Dataset.
        labels (np.ndarray, optional): Labels for the dataset.
    """
    k_values = range(1, 10)
    silhouette_coefficients = []

    for k in k_values:
        print(k)
        centroids, cluster_assignments = kmeans(data, k)
        # print(centroids)
        # print(cluster_assignments)
        if k == 1:
            silhouette_coefficient = 0
        else:
           silhouette_coefficient = compute_silhouette_coefficient(data, cluster_assignments, centroids)
        silhouette_coefficients.append(silhouette_coefficient)

    # print(silhouette_coefficients)
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_coefficients, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Coefficient vs Number of Clusters')
    plt.show()
# Load the dataset
data, labels = load_dataset('dataset')

# Plot Silhouette coefficient for k-means clustering
plot_silhouette(data, labels)