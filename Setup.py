from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
from timeit import default_timer as timer
import pickle


def import_tweets_from_mongoDB(n_tweets):
    """
    Connects to a Mongo DB instance and retrieve the main information for each tweet. The information retrieved are:
        The unique tweet ID from Mongo DB index
        The text from the tweet
        The artist related to the tweet (handled by the tweet API)
        The user from twitter which posted the tweet
        The date in which the tweet was posted
        The language from the tweet

    Each tweet is stored in a dict. All tweets are stored in a list

    :param n_tweets: Number of tweets to be imported from Mongo DB
    :type n_tweets: int

    :return:
    """

    start = timer()

    # Import our tweet collection through Mongo DB and PyMongo
    client = MongoClient()
    db = client.ds4dems
    collection = db.tweets

    # Store the important tweets information in a list
    tweets = []  # List containing all the tweets important information

    for document in collection.find().limit(n_tweets):
        tweet = {}
        tweet['id'] = str(document['_id'])
        tweet['text'] = document['text']
        tweet['artist'] = document['handle']
        tweet['user'] = str(document['author_id'])
        tweet['date'] = document['created_at_datetime'].date()
        tweet['language'] = document['lang']

        # Append the tweet to the tweets list
        tweets.append(tweet)

    end = timer()
    print('Imported the %d tweets in %.2f seconds' % (len(tweets), (end - start)))
    txt_file.write('Imported the %d tweets in %.2f seconds. \n' % (len(tweets), (end - start)))

    return tweets


def build_tfidf_matrix(tweets, min_df, ngram_range):
    """
    Build the tf-idf matrix. Creates an object from the TfidfVectorizer class with the parameters for building the
    tf-idf matrix. The tokenize_and_stem method tokenize and stem each tweet text and filter some of the words.

    The tf-idf matrix and the vocabulary (model features) are obtained after fitting and transforming the data.

    :param tweets: list with tweets' information
    :type tweets: list
    :param min_df: Terms with document frequency smaller than min_df will be filtered
    :type min_df: int or float
    :param ngram_range: Combination of n consecutive words will be added to the vocabulary, where n in ngram_range
    :type ngram_range: tuple

    :return: tf-idf matrix and vocabulary list
    :rtype: tuple
    """

    # Define the vectorizers parameters
    start = timer()
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df,
                                       max_features=200000,
                                       stop_words='english', use_idf=True,
                                       tokenizer=tokenize_and_stem,
                                       ngram_range=ngram_range)


    # Fit the vectorizers to the tweets text
    tweets_text = [tweet['text'] for tweet in tweets]
    tfidf_matrix = tfidf_vectorizer.fit_transform(tweets_text)  # Tf-idf matrix
    vocabulary = tfidf_vectorizer.get_feature_names()  # Vocabulary list

    end = timer()
    print('Obtained the tf-idf matrix and vocabulary list in %.2f seconds' % (end - start))
    txt_file.write('Obtained the tf-idf matrix and vocabulary list in %.2f seconds. \n' % (end - start))

    return tfidf_matrix, vocabulary


def tokenize_and_stem(text, filtering=True, filter_words=["rt", "'s", "n't", "amp"], language='english'):
    """
    Receives a text and return only the stem of the tokens of a text. Some of the tokens are filtered if:
        it does not contain any letter
        it is an english stop word
        it is one of the filter words
        it starts with http or /, i.e., it is a website url

    :param text: Text to be analyzed
    :type text: String
    :param filtering: If tokens not containing letter should be filtered
    :type filtering: Boolean
    :param filter_words: List of words to be filtered
    :type filter_words: List
    :param language: Defines the language of the stemmer
    :type language: String

    :return: A list with the stems
    :rtype: List
    """

    # Load nltk's SnowballStemmer as variable 'stemmer'
    stemmer = SnowballStemmer(language)

    # First tokenize by sentence and then by word to ensure that punctuation is caught as it' own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    if filtering:
        # Filter out any tokens not containing letter (e.g., numbers and raw punctuation), in the stop words list
        # in the filter words list or containing a website link
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                if token not in stopwords.words(language):
                    if token not in filter_words:
                        if not token.startswith('http'):
                            if not token.startswith('/'):
                                filtered_tokens.append(token)
        stems = [stemmer.stem(token) for token in filtered_tokens]
        return stems
    else:
        stems = [stemmer.stem(token) for token in tokens]
        return stems


def build_term_document_matrix(tweets, min_df, ngram_range):
    """
    Build the term document matrix. Creates an object from the CountVectorizer class with the parameters for building
    the term document matrix. The tokenize_and_stem method tokenize and stem each tweet text and filter some of the
    words.

    :param tweets: list with tweets' information
    :type tweets: list
    :param min_df: Terms with document frequency smaller than min_df will be filtered
    :type min_df: int or float
    :param ngram_range: Combination of n consecutive words will be added to the vocabulary, where n in ngram_range
    :type ngram_range: tuple

    :return: term document matrix
    :rtype: list
    """

    # Define the vectorizers parameters
    start = timer()
    count_vectorizer = CountVectorizer(min_df=min_df,
                                       max_features=200000,
                                       stop_words='english',
                                       tokenizer=tokenize_and_stem,
                                       ngram_range=ngram_range)

    # Fit the vectorizers to the tweets text
    tweets_text = [tweet['text'] for tweet in tweets]
    term_document_matrix = count_vectorizer.fit_transform(tweets_text)      # Term-document matrix

    end = timer()
    print('Obtained the term-document matrix in %.2f seconds' % (end-start))
    txt_file.write('Obtained the term-document matrix in %.2f seconds. \n' % (end-start))

    return term_document_matrix


def filter_tweets(tweets, tfidf_matrix, term_document_matrix):
    """
    Receive all the tweets obtained from the Mongo DB instance. Filter some tweets and its related vectors in the
    tf-idf and term document matrix. Return the new tweets list, tf-idf and term document matrix, without the filtered
    tweets. A tweet is filtered if:
        it is not written in english
        it does not have any of tokens in the model (zero vector)

    :param tweets: list with tweets' information
    :type tweets: list
    :param tfidf_matrix: tf-idf matrix
    :type tfidf_matrix: csr_matrix
    :param term_document_matrix: term-document matrix
    :type term_document_matrix: csr_matrix

    :return: filtered tweets list, tf-idf and term-document matrix
    :rtype: tuple
    """

    start = timer()

    # Build the list to store the filtered tweets, tf-idf and term-document matrix
    relevant_tweets = []                    # List containing only the relevant tweets
    relevant_term_document_matrix = []      # List containing only the term-document matrix relevant components
    relevant_tfidf_matrix = []              # List containing only the tf-idf matrix relevant components

    # Store the old number of tweets
    old_number_of_tweets = len(tweets)

    # Obtain only the relevant tweets and correspondent document-term and tf-idf matrix vectors
    for tweet, vector1, vector2 in zip(tweets, term_document_matrix.todense().tolist(),
                                       tfidf_matrix.todense().tolist()):
        if not np.linalg.norm(vector2) == 0:
            if tweet['language'] == 'en':
                relevant_tweets.append(tweet)
                relevant_term_document_matrix.append(vector1)
                relevant_tfidf_matrix.append(vector2)

    # Substitute the tweets list, tf-idf and term-document matrix with the relevant ones
    tweets = relevant_tweets
    tfidf_matrix = csr_matrix(relevant_tfidf_matrix)
    term_document_matrix = csr_matrix(relevant_term_document_matrix)

    # Print the number of filtered documents
    print('Number of filtered documents: %d.' % (old_number_of_tweets - len(tweets)))
    txt_file.write('Number of filtered documents: %d. \n' % (old_number_of_tweets - len(tweets)))

    end = timer()
    print('Filtered the tweets in %.2f seconds.' % (end-start))
    txt_file.write('Filtered the tweets in %.2f seconds. \n' % (end-start))

    return tweets, tfidf_matrix, term_document_matrix

# Start the timer
all_start = timer()

# Define setup parameters
n_tweets = 1000000
min_df = 0.01
ngram_range = (1,1)

# Open a text file to store information
txt_file = open("Setup Output - " + str(n_tweets) + " Tweets.txt", "w")

# Import and store the tweet collection
tweets = import_tweets_from_mongoDB(n_tweets=n_tweets)

# Obtain the tf-idf matrix and the vocabulary list
tfidf_matrix, vocabulary = build_tfidf_matrix(tweets=tweets, min_df=min_df, ngram_range=ngram_range)

# Obtain the term-document matrix
term_document_matrix = build_term_document_matrix(tweets=tweets, min_df=min_df, ngram_range=ngram_range)

# Filter the not relevant tweets
tweets, tfidf_matrix, term_document_matrix = filter_tweets(tweets=tweets, tfidf_matrix=tfidf_matrix,
                                                            term_document_matrix=term_document_matrix)
# Print the relevant information
print('Number of documents: %d.' % tfidf_matrix.shape[0])
print('Number of tokens in vocabulary: %d.' % tfidf_matrix.shape[1])
print('Term-document matrix shape: %d x %d' % (term_document_matrix.shape[0], term_document_matrix.shape[1]))
print('Tf-idf matrix shape: %d x %d' % (tfidf_matrix.shape[0], tfidf_matrix.shape[1]))
print('Vocabulary list size: %d' % len(vocabulary))

txt_file.write('Number of documents: %d. \n' % tfidf_matrix.shape[0])
txt_file.write('Number of tokens in vocabulary: %d. \n' % tfidf_matrix.shape[1])
txt_file.write('Term-document matrix shape: %d x %d. \n' % (term_document_matrix.shape[0],
                                                            term_document_matrix.shape[1]))
txt_file.write('Tf-idf matrix shape: %d x %d. \n' % (tfidf_matrix.shape[0], tfidf_matrix.shape[1]))
txt_file.write('Vocabulary list size: %d. \n' % len(vocabulary))

# Save the tf-idf matrix, the vocabulary list and the tweets
pickle.dump(tweets, open('Tweets Data - ' + str(n_tweets) + ' Tweets.p', 'wb'))
pickle.dump(tfidf_matrix, open('TF-IDF Matrix - ' + str(n_tweets) + ' Tweets.p', 'wb'))
pickle.dump(vocabulary, open('Vocabulary - ' + str(n_tweets) + ' Tweets.p', 'wb'))
pickle.dump(term_document_matrix, open('Term Document Matrix - ' + str(n_tweets) + ' Tweets.p', 'wb'))

# Print the code total execution time
all_end = timer()
print('Setup Total Execution Time: %.2f' % (all_end-all_start))
txt_file.write('Setup Total Execution Time: %.2f seconds. \n' % (all_end-all_start))

# Close the text file
txt_file.close()