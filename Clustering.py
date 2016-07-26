from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import pickle
import json


def kmeans_clustering(n_clusters=3, reduce_dimensionality=True, n_components=2):
    """
    Load the tf-idf matrix. If reduce dimensionality is true, reduce the dimensionality of the problem using PCA.
    Build and use a K-Means Clustering model to cluster the data. Return the clusters assignment for each tweet.

    :param n_clusters: number of clusters of the model
    :type n_clusters: int
    :param reduce_dimensionality: If a PCA should be performed to reduce the dimensionality of the clustering
    :type reduce_dimensionality: bool
    :param n_components: Number of components in the PCA used to reduce the dimensionality of the clustering
    :type n_components: int

    :return: clusters assignment for each tweet
    :rtype: list
    """

    # If reduce_dimensionality, reduce the dimensionality of the clustering problem. Else, load the files
    if reduce_dimensionality:
        tfidf_matrix = reduce_data_dimensionality(n_components)
    else:
        tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Obtain the K-Means Clustering clusters
    start = timer()
    km = KMeans(n_clusters=n_clusters).fit(tfidf_matrix)
    clusters = km.labels_
    end = timer()
    print('Build K-Means Clustering model in %.2f seconds' % (end - start))
    txt_file.write('Build K-Means Clustering model in %.2f seconds. \n' % (end - start))

    return clusters


def spectral_clustering(n_clusters=3, reduce_dimensionality=True, n_components=2):
    """
    Load the tf-idf matrix. If reduce dimensionality is true, reduce the dimensionality of the problem using PCA.
    Build and use a Spectral Clustering model to cluster the data. Return the clusters assignment for each tweet.

    :param n_clusters: number of clusters of the model
    :type n_clusters: int
    :param reduce_dimensionality: If a PCA should be performed to reduce the dimensionality of the clustering
    :type reduce_dimensionality: bool
    :param n_components: Number of components in the PCA used to reduce the dimensionality of the clustering
    :type n_components: int

    :return: clusters assignment for each tweet
    :rtype: list
    """

    # If reduce_dimensionality, reduce the dimensionality of the clustering problem. Else, load the files
    if reduce_dimensionality:
        tfidf_matrix = reduce_data_dimensionality(n_components)
    else:
        tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Obtain the Spectral Clustering clusters
    start = timer()
    spectral = SpectralClustering(n_clusters=n_clusters).fit(tfidf_matrix)
    clusters = spectral.labels_
    end = timer()
    print('Build Spectral Clustering model in %.2f seconds' % (end - start))
    txt_file.write('Build Spectral Clustering model in %.2f seconds. \n' % (end - start))

    return clusters


def agglomerative_clustering(n_clusters=3, reduce_dimensionality=True, n_components=2):
    """
    Load the tf-idf matrix. If reduce dimensionality is true, reduce the dimensionality of the problem using PCA.
    Build and use an Agglomerative Clustering model to cluster the data. Return the clusters assignment for each tweet.

    :param n_clusters: number of clusters of the model
    :type n_clusters: int
    :param reduce_dimensionality: If a PCA should be performed to reduce the dimensionality of the clustering
    :type reduce_dimensionality: bool
    :param n_components: Number of components in the PCA used to reduce the dimensionality of the clustering
    :type n_components: int

    :return: clusters assignment for each tweet
    :rtype: list
    """

    # If reduce_dimensionality, reduce the dimensionality of the clustering problem. Else, load the files
    if reduce_dimensionality:
        tfidf_matrix = reduce_data_dimensionality(n_components)
    else:
        tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Obtain the Agglomerative Clustering clusters
    start = timer()
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters).fit(tfidf_matrix.todense())
    clusters = agglomerative.labels_
    end = timer()
    print('Build Agglomerative Clustering model in %.2f seconds' % (end - start))
    txt_file.write('Build Agglomerative Clustering model in %.2f seconds. \n' % (end - start))

    return clusters


def reduce_data_dimensionality(n_components=2):
    """
    Loads the tf-idf matrix. Reduce the dimensionality of the vectors through PCA. Returns the tf-idf matrix with
    the reduced number of components.

    :param n_components: Number of components in the PCA used to reduce the dimensionality of the clustering
    :type n_components: int

    :return: data with reduced dimensionality
    :rtype: csr_matrix
    """

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Reduce the data dimensionality
    start = timer()
    reduced_tfidf_matrix = csr_matrix(PCA(n_components=n_components).fit_transform(tfidf_matrix.toarray()))
    end = timer()
    print('Reduced data dimensionality in %.2f seconds' % (end - start))
    txt_file.write('Reduced data dimensionality in %.2f seconds. \n' % (end - start))

    return reduced_tfidf_matrix


def plot_clustering(clusters):
    """
    Plot the clustering model in a two dimensional plane. For each cluster, obtain the tweets that belong to it,
    plot them through their principal components and set their color to the same value.

    :param clusters: clusters labels for each tweet
    :type clusters: list

    :return:
    """

    # Define a set of the clusters and colors
    start = timer()
    unique_clusters = set(clusters)
    n_clusters = len(unique_clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

    # Obtain the data in a two dimensional space
    pc = obtain_pc()

    # Plot the clustering model
    plt.figure(figsize=(17,9))
    for cluster, color in zip(unique_clusters, colors):
        # Build a boolean array mapping which samples are from the cluster
        cluster_samples_mask = (clusters == cluster)

        # Plot the samples which are from the cluster
        samples = pc[cluster_samples_mask]
        plt.plot(samples[:,0], samples[:,1], 'o',
                 markerfacecolor=color,
                 markeredgecolor='k')

    plt.title(clustering + ' Clustering: Number of Tweets = %d' % len(pc))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(clustering + ' Clustering - ' + str(n_data) + ' Tweets and ' + str(n_clusters) + ' Clusters.png',
                dpi=200)
    plt.close()
    end = timer()
    print('Build the plot in %.2f seconds' % (end - start))
    txt_file.write('Build the plot in %.2f seconds. \n' % (end - start))


def obtain_pc():
    """
    Loads the tf-idf matrix. Use Principal Components Analysis to obtain a representation of the tf-idf vectors, i.e,
    the tweets in a two dimensional plane. Return the principal components (x,y) of each tweets.

    :return: principal components (x,y) of each tweet
    :rtype: list
    """

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Find the 2 PC of the tweets text in order to plot them in a two dimensional plane
    start = timer()
    pc = PCA(n_components=2).fit_transform(tfidf_matrix.toarray())
    end = timer()
    print('Build PCA model in %.2f seconds' % (end - start))
    txt_file.write('Build PCA model in %.2f seconds. \n' % (end - start))

    return pc


def compute_silhouette_score(clusters):
    """
    Compute the euclidean silhouette score and the cosine silhouette score. Return the scores.

    :param clusters: clusters assignment for each tweet
    :type clusters: list

    :return: the silhouette scores
    :rtype: tuple
    """

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Compute the Silhouette Score
    start = timer()
    distance = 1 - cosine_similarity(tfidf_matrix)
    euclidean_silhouette_score = silhouette_score(tfidf_matrix, clusters, metric='euclidean')
    cosine_silhouette_score = silhouette_score(distance, clusters, metric='precomputed')
    end = timer()
    print('Silhouette Score (Euclidean): %.4f' % euclidean_silhouette_score)
    print('Silhouette Score (Cosine):  %.4f' % cosine_silhouette_score)
    print('Obtained the Silhouette Score in %.2f seconds' % (end - start))
    txt_file.write('Silhouette Score (Euclidean): %.4f. \n' % euclidean_silhouette_score)
    txt_file.write('Silhouette Score (Cosine):  %.4f. \n' % cosine_silhouette_score)
    txt_file.write('Obtained the Silhouette Score in %.2f seconds. \n' % (end - start))

    return euclidean_silhouette_score, cosine_silhouette_score


def obtain_clusters_centers(clusters):
    """
    Load the tf-idf matrix. Group the vectors by their cluster. Compute and return the mean value of each cluster, i.e.,
    its center.

    :param clusters: clusters assignment for each tweet
    :type clusters: list

    :return: center coordinates
    :rtype: nd array
    """

    start = timer()

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # For each cluster center, obtain its coordinates
    clusters_centers = []
    unique_clusters = set(clusters)
    for cluster in unique_clusters:
        cluster_mask = (clusters == cluster)
        cluster_samples_coordinates = np.asarray(tfidf_matrix.todense())[cluster_mask]
        center_coordinate = np.mean(cluster_samples_coordinates, 0)
        clusters_centers.append(center_coordinate)

    end = timer()
    print('Obtained the clusters centers coordinates in %.2f seconds' % (end-start))
    txt_file.write('Obtained the clusters centers coordinates in %.2f seconds. \n\n' % (end-start))

    return clusters_centers


def obtain_relevant_words(clusters_centers, n_words):
    """
    Obtain the most relevant words in each cluster. Receive the clusters centers coordinates. The most relevant
    words are the ones which the related center coordinate has the highest values.

    :param clusters_centers: centers coordinates
    :type clusters_centers: nd array
    :param n_words: number of words to be printed out
    :type n_words: int

    :return: the most relevant words
    :rtype: list
    """

    start = timer()

    # Load the files
    vocabulary = pickle.load(open('Vocabulary - ' + str(n_data) + ' Tweets.p', 'rb'))
    if len(vocabulary) < n_words:
        n_words = len(vocabulary)

    # Print the most relevant words for each cluster
    relevant_words = []
    for cluster_index, cluster_center in enumerate(clusters_centers):
        # Print the most relevant words and save them in relevant_words
        ordered_center_coordinates = cluster_center.argsort()[::-1]
        print('Cluster %d most relevant words:' % cluster_index, end='')
        txt_file.write('Cluster %d most relevant words:' % cluster_index)
        words = []
        for index in ordered_center_coordinates[:n_words]:
            print(' %s' % vocabulary[index], end=',')
            txt_file.write(' %s,' % vocabulary[index])
            words.append(vocabulary[index])
        relevant_words.append(words)
        print()
        print()
        txt_file.write('\n')
    txt_file.write('\n')

    pickle.dump(relevant_words, open(str(clustering) + ' Relevant Words - ' + str(n_data) + ' Tweets.p', 'wb'))

    end = timer()
    print('Obtain the most relevant words in %.2f seconds' % (end-start))
    txt_file.write('Obtain the most relevant words in %.2f seconds. \n' % (end-start))

    return relevant_words


# Start the timer
all_start = timer()

# Clustering Algorithm Parameters
n_data = 1000
n_clusters = 3
reduce_dimensionality = False
n_components = 2
n_words = 10
clustering = 'KMeans'        # KMeans, Spectral or Agglomerative

# Open a text file to store information
txt_file = open(clustering + ' Clustering Output - ' + str(n_data) + ' Tweets and ' + str(n_clusters) + ' Clusters.txt',
                "w")

# Perform the clustering algorithm and plot the results in a two dimensional space

# Obtain the cluster labels for each tweet
if clustering == 'KMeans':
    clusters = kmeans_clustering(n_clusters=n_clusters, reduce_dimensionality=reduce_dimensionality,
                                   n_components=n_components)
elif clustering == 'Spectral':
    clusters = spectral_clustering(n_clusters=n_clusters, reduce_dimensionality=reduce_dimensionality,
                                   n_components=n_components)
elif clustering == 'Agglomerative':
    clusters = agglomerative_clustering(n_clusters=n_clusters, reduce_dimensionality=reduce_dimensionality,
                                   n_components=n_components)

# Plot the Spectral Clustering
plot_clustering(clusters=clusters)

# Obtain the cluster most relevant words and tweets
clusters_centers = obtain_clusters_centers(clusters=clusters)
relevant_words = obtain_relevant_words(clusters_centers=clusters_centers, n_words=n_words)

# Compute the Silhouette Score
if n_data <= 10000:
    e_s_score, c_s_score = compute_silhouette_score(clusters=clusters)

# Save the clusters labels to the tweets data
pickle.dump(clusters, open(clustering + ' Clusters Assignments - ' + str(n_data) + ' Tweets.p', 'wb'))

# Print the code total execution time
all_end = timer()
print(clustering + ' Clustering Total Execution Time: %.2f' % (all_end - all_start))
txt_file.write(clustering + ' Clustering Total Execution Time: %.2f.' % (all_end - all_start))

# Close the text file
txt_file.close()