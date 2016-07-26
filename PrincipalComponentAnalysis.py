from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pickle


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


def obtain_spc(alpha=0.1):
    """
    Loads the tf-idf matrix. Use Sparse Principal Components Analysis to obtain a representation of the tf-idf vectors,
    i.e, the tweets in a two dimensional plane.

    The principal components of SPCA are not a linear combination of all the components in the model. In fact, it uses
    only a few of the model's components to build the principal components. The parameter alpha regulates how strong
    the filtering of model's components will be.

    Return the principal components (x,y) of each tweets.

    :param alpha: it regulates how strong the filtering of model's components will be
    :type alpha: int

    :return: principal components (x,y) of each tweet
    :rtype: list
    """

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Find the 2 PC of the tweets text in order to plot them in a two dimensional plane
    start = timer()
    spc = SparsePCA(n_components=2, alpha=alpha).fit_transform(tfidf_matrix.toarray())
    end = timer()
    print('Build SPCA model in %.2f seconds' % (end - start))
    txt_file.write('Build SPCA model in %.2f seconds. \n' % (end - start))

    return spc


def plot_data(pc, spc):
    """
    Receives the principal components (x,y) from PCA and SPCA for each tweet. Plot each tweet in both of these two-
    dimensional plane.

    :param pc: principal components from PCA
    :type pc: list
    :param spc: principal components from SPCA
    :type spc: list

    :return:
    """

    # Plot the tweets text vectors in the 2 PC for each method of PCA
    start = timer()
    fig = plt.figure(figsize=(17,9))

    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(pc[:,0], pc[:,1], 'bo')
    plt.ylabel('Principal Component 2')
    plt.title('PCA and SPCA: Number of Tweets = %d' % len(pc))

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(spc[:,0], spc[:,1], 'ro')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.savefig('PCA Plot - ' + str(n_data) + ' Tweets.png', dpi=200)
    plt.close()

    end = timer()
    print('Build the plots in %.2f seconds' % (end-start))
    txt_file.write('Build the plots in %.2f seconds. \n' % (end-start))

# Start the timer
all_start = timer()

# Parameters definition
n_data = 1000000
alpha = 0.3

# Open a text file to store information
txt_file = open("PCA Output - " + str(n_data) + " Tweets.txt", "w")

# Perform the PCA and SPCA and plot the results in a two dimensional space

# Obtain the 2 PC of the tweets text using PCA
pc = obtain_pc()

# Obtain the 2 PC of the tweets text using PCA
spc = obtain_spc()

# Plot the tweets in the PC and SPC
plot_data(pc=pc, spc=spc)

# Print the code total execution time
all_end = timer()
print('PCA Total Execution Time: %.2f' %(all_end - all_start))
txt_file.write('PCA Total Execution Time: %.2f.' %(all_end - all_start))

# Close the text file
txt_file.close()