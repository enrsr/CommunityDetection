from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.sparse import csr_matrix
import numpy as np
import mpmath as math
from timeit import default_timer as timer
import pickle


def documents_partition (time_range=3):
    """
    First step of the Temporal Text Mining algorithm. Partition the tweets text based on the day they were posted.
    Returns the partition of the tweets and corresponding vectors of the tf-idf matrix.

    :param time_range: How many days should be used to form one set in the partition
    :type time_range: int

    :return: tweets and tf-idf matrix partition
    :rtype: tuple
    """

    start = timer()

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))
    tweets = pickle.load(open('Tweets Data - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Partition the data
    # Obtain all the days in which a tweet was posted
    days = [tweet['date'] for tweet in tweets]

    # Count the number of days in the data
    number_days = len(set(days))

    # Build the partition
    tweets_partition = []
    dense_tfidf_matrix_partition = []

    partition_size = int(math.ceil(number_days/time_range))
    for index in range(partition_size):
        tweets_partition.append([])
        dense_tfidf_matrix_partition.append([])

    # Obtain the days in form of index and allocate the tweets and vectors to its partition
    minimum_day = min(days).toordinal()
    dense_tfidf_matrix = tfidf_matrix.todense().tolist()
    for tweet, vector in zip(tweets, dense_tfidf_matrix):
        day_index = tweet['date'].toordinal() - minimum_day + 1
        partition_index = int(math.ceil(day_index/time_range)) - 1
        tweets_partition[partition_index].append(tweet)
        dense_tfidf_matrix_partition[partition_index].append(vector)

    # Convert the dense tf-idf matrix in the parttion to sparse tf-idf matrix
    tfidf_matrix_partition = []
    for dense_tfidf_matrix in dense_tfidf_matrix_partition:
        tfidf_matrix = csr_matrix(dense_tfidf_matrix)
        tfidf_matrix_partition.append(tfidf_matrix)

    end = timer()
    print('Obtained the partition in %.2f seconds' % (end - start))
    txt_file.write('Obtained the partition in %.2f seconds. \n' % (end - start))

    # Return the tweets and tf-idf matrix partition
    return tweets_partition, tfidf_matrix_partition


def print_partitions(tweets_partition):
    """
    Print the dates and the number of tweets in each partition

    :param tweets_partition: the partition of tweets based on the day they were posted
    :type tweets_partition: list

    :return:
    """

    for partition_index, partition in enumerate(tweets_partition):

        # Print the number of tweets in the partition
        print('Partition %d has %d tweets.' % (partition_index, len(partition)))
        txt_file.write('Partition %d has %d tweets. \n' % (partition_index, len(partition)))

        # Obtain the set of unique dates in each partition
        partition_dates = [tweet['date'] for tweet in partition]
        unique_partition_dates = set(partition_dates)

        # Print the unique dates in each partition
        print('Partition %d dates: ' % partition_index, end='')
        txt_file.write('Partition %d dates: ' % partition_index)
        for date in unique_partition_dates:
            print(date, end=', ')
            txt_file.write('%s, ' % date)
        print()
        txt_file.write('\n')
    print()
    txt_file.write('\n')


def themes_extraction(tfidf_matrix_partition, n_themes = 3):
    """
    Second step of the Temporal Text Mining algorithm. Receives a partition of the tweets. For each partition, extract
    its themes. Return the themes of each partition.

    :param tfidf_matrix_partition: Nested list with the partition of the tf-idf matrix
    :type tfidf_matrix_partition: list
    :param n_themes: Number of themes to be extracted from each partition
    :type n_themes: int

    :return: A list containing the distribution of each of the themes of the partition
    :rtype: list
    """

    start = timer()

    # Defines the list to be returned
    all_partitions_themes = []                     # List to store the words of the themes of each partition

    # For each partition, obtain the word distribution for each theme
    for tfidf_matrix in tfidf_matrix_partition:

        # Obtain the themes word distribution in this partition through LDA
        partition_themes = LDA(n_topics=n_themes).fit(tfidf_matrix).components_

        # Normalize the themes to obtain probabilities
        normalized_partition_themes = []
        for theme in partition_themes:
            normalize_factor = sum(theme)
            theme = np.asarray([word/normalize_factor for word in theme])
            normalized_partition_themes.append(theme)

        # Store the partition's themes word distribution
        all_partitions_themes.append(normalized_partition_themes)

    end = timer()
    print('Extracted the themes in %.2f seconds' % (end - start))
    txt_file.write('Extracted the themes in %.2f seconds. \n' % (end - start))

    # Return the themes word distribution in each partition
    return all_partitions_themes


def print_themes(all_partitions_themes, n_words):
    """
    Receives the themes of each partition. Print the most relevant words in each theme.

    :param all_partitions_themes: themes word distribution in each partition
    :type all_partitions_themes: list
    :param n_words: number of relevant words to be printed for each theme
    :type n_words: int

    :return:
    """

    # Load the files
    vocabulary = pickle.load(open('Vocabulary - ' + str(n_data) + ' Tweets.p', 'rb'))

    # For each partition, print its most relevant words in the themes
    for partition_index, partition in enumerate(all_partitions_themes):
        print('Partition %d most relevant themes.' % partition_index)
        txt_file.write('Partition %d most relevant themes. \n' % partition_index)

        # For each theme, print its most relevant words
        for theme_index, theme in enumerate(partition):
            print('Theme %d most relevant words: ' % theme_index, end='')
            txt_file.write('Theme %d most relevant words: ' % theme_index)
            for index in theme.argsort()[:-n_words - 1:-1]:
                print('%s' % vocabulary[index], end=', ')
                txt_file.write('%s, ' % vocabulary[index])
            print()
            txt_file.write('\n')
        print()
        txt_file.write('\n')


def compute_similarity(partitions_themes, similarity='KLDivergence'):
    """
    Third step of the Temporal Text Mining algorithm. Receives the distribution of all themes in two partitions.
    Computes and returns the similarity between all the themes.

    :param partitions_themes: Contains the distribution of all themes in the partitions.
    :type list
    :param similarity: Specifies which similarity measure to use
    :type str

    :return: Returns the computed similarities between all the themes.
    :rtype: list
    """

    # Declare the similarity matrix to store the computed similarities
    similarity_matrix = []

    # Compute the similarities between all the themes
    for theme1_index, theme1 in enumerate(partitions_themes[0]):

        # Creates a new line in the matrix
        similarity_matrix.append([])

        for theme2_index, theme2 in enumerate(partitions_themes[1]):

            # If the similarity is KL-Divergence
            if similarity == 'KLDivergence':
                vocabulary_size = len(theme1)
                klDivergence = 0
                for word_index in range(vocabulary_size):
                    klDivergence += theme2[word_index]*math.log(theme2[word_index]/theme1[word_index])
                similarity_matrix[theme1_index].append(1/klDivergence)

            # If the similarity is Jensen-Shanon Divergence (JSD)
            if similarity == 'JSDivergence':
                vocabulary_size = len(theme1)
                jsDivergence = 0
                for word_index in range(vocabulary_size):
                    jsDivergence += (0.5*theme1[word_index]*math.log(theme1[word_index]/theme2[word_index]) +
                                     0.5*theme2[word_index]*math.log(theme2[word_index]/theme1[word_index]))
                similarity_matrix[theme1_index].append(1/jsDivergence)

            # If the similarity is the support of the distributions
            if similarity == 'support':
                # Compute the support of each theme distribution
                threshold = 0.001
                theme1_support = (theme1 > threshold)
                theme2_support = (theme2 > threshold)

                # Compute and store the similarity between the distributions' support
                support_similarity = sum((theme1_support & theme2_support))
                similarity_matrix[theme1_index].append(support_similarity)

    # Returns the computed similarities between all the themes.
    return similarity_matrix


def print_similarity_matrix(similarity_matrix, index):
    """
    Print the similarity matrix.

    :param similarity_matrix: matrix with the similarity between themes from 2 consecutive partitions.
    :type similarity_matrix: list
    :param index: index of the first partition
    :type index: int

    :return:
    """

    # Print the similarity matrix
    print('Similarity matrix for themes in partition %d and %d:' % (index, index+1))
    txt_file.write('Similarity matrix for themes in partition %d and %d: \n' % (index, index+1))
    for line_index, line in enumerate(similarity_matrix):
        print('Partition %d - Theme %d: ' % (index, line_index), end='')
        txt_file.write('Partition %d - Theme %d: ' % (index, line_index))
        for element_index, element in enumerate(line):
            print('%.4f' % float(element), end=', ')
            txt_file.write('%.4f, ' % float(element))
        print()
        txt_file.write('\n')
    print()
    txt_file.write('\n')


def print_related_themes(similarity_matrix, threshold):
    """
    Print the related themes from two consecutive partitions.

    :param similarity_matrix: matrix with the similarity between themes from 2 consecutive partitions.
    :type similarity_matrix: list
    :param threshold: minimum value of similarity for two themes to be considered related
    :type threshold: float

    :return:
    """

    # Print the related themes
    print('Related themes:')
    txt_file.write('Related themes: \n')
    for line_index, line in enumerate(similarity_matrix):
        for element_index, element in enumerate(line):
            if float(element) > threshold:
                print('Theme %d from partition %d and theme %d from partition %d.'
                          % (line_index, index, element_index, index+1))
                txt_file.write('Theme %d from partition %d and theme %d from partition %d. \n'
                          % (line_index, index, element_index, index+1))
    print()
    txt_file.write('\n')

# Start the timer
all_start = timer()

# Parameters
n_data = 1000000
time_range = 3
n_themes = 10
n_words = 10
similarity = 'KLDivergence'

# Open a text file to store information
txt_file = open('TTM Output - ' + str(n_data) + ' Tweets and ' + str(n_themes) + ' Themes.txt', "w")

# Step 1: Obtain the tweets and tf-idf partition

# Obtain the tweets and tf-idf matrix partition
tweets_partition, tfidf_matrix_partition = documents_partition(time_range=time_range)

# For each partition, print the tweet dates
print_partitions(tweets_partition=tweets_partition)

# Step 2: Obtain the most relevant themes of each partition

# Obtain the theme distribution of all the partitions
all_partitions_themes = themes_extraction(tfidf_matrix_partition, n_themes=n_themes)

# Print the most relevant words of each theme in the partitions
print_themes(all_partitions_themes=all_partitions_themes, n_words=n_words)

# Step 3: Find the similarities between the themes among different time sets.
start = timer()
all_partitions_similarity_matrix = []
for index in range(len(all_partitions_themes)-1):

    # Create a list with the two partitions to be compared
    partitions_themes = []
    partitions_themes.append(all_partitions_themes[index])
    partitions_themes.append(all_partitions_themes[index+1])

    # Compute the similarity matrix
    similarity_matrix = compute_similarity(partitions_themes, similarity=similarity)
    all_partitions_similarity_matrix.append(similarity_matrix)

    # Print the similarity matrix
    print_similarity_matrix(similarity_matrix=similarity_matrix, index=index)

    # Define the threshold
    threshold = 1/3

    # Print the related themes
    print_related_themes(similarity_matrix=similarity_matrix, threshold=threshold)

# Save the relevant information from the partitions
pickle.dump(all_partitions_themes, open(str(n_data) + 'all_partitions_themes.p', 'wb'))
pickle.dump(all_partitions_similarity_matrix, open(str(n_data) + 'all_partitions_similarity_matrix.p', 'wb'))

end = timer()
print('Computed the similarities in %.2f seconds' % (end - start))
txt_file.write('Computed the similarities in %.2f seconds. \n' % (end - start))

all_end = timer()
print('Temporal Text Mining Execution time: %.2f' % (all_end-all_start))
txt_file.write('Temporal Text Mining Execution time: %.2f. \n' % (all_end-all_start))

# Close the text file
txt_file.close()