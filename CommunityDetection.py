from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
import numpy as np
import mpmath as math
from timeit import default_timer as timer
import pickle

def import_tweets_information():
    """
    Import the tweets information that will be used in the algorithm. The information are:
        Author of the tweet (user)
        Artist to which the tweet is related

    :return: the tweets' important information
    :rtype: tuple
    """

    start = timer()

    # Load the files
    tweets = pickle.load(open('Tweets Data - ' + str(n_data) + ' Tweets.p', 'rb'))
    tweets_user = []
    tweets_artist = []

    # Store the important information
    for tweet in tweets:
        # Store the tweets' user
        tweets_user.append(tweet['user'])
        tweets_artist.append(tweet['artist'])

    end = timer()
    print('Imported the tweets information in %.2f seconds.' % (end-start))
    txt_file.write('Imported the tweets information in %.2f seconds. \n' % (end-start))

    # Return the important information
    return tweets_user, tweets_artist


def map_user_to_int(tweets_user):
    """
    Map the users' values to integer values. Return the tweets' users as integers

    :param tweets_user: User of each tweet
    :type tweets_user: list

    :return: tweets' users as integers
    :rtype: list
    """

    start = timer()
    # Create a dictionary to map the users to integers
    mapper = {}
    index = 0
    for user in set(tweets_user):
        mapper.update({user:index})
        index += 1

    # Substitute the user value to its mapped integer
    tweets_user_as_int = [mapper[user] for user in tweets_user]

    end = timer()
    print('Mapped the users to integers in %.2f seconds' % (end-start))
    txt_file.write('Mapped the users to integers in %.2f seconds. \n' % (end-start))

    return tweets_user_as_int


def number_words_in_vocabulary():
    """
    Obtain the number of words in the vocabulary.

    :return: number of words in the vocabulary
    :rtype: int
    """

    # Load the file
    vocabulary = pickle.load(open('Vocabulary - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Obtain and return the number of words
    n_words = len(vocabulary)
    return n_words


def assign_topic_and_community(n_topics, n_communities):
    """
    Assigns a topic and a community for each tweet. Build a LDA model and obtain the document topic distribution. Sample
    a topic from each document topic distribution. Uses the document topic distribution to cluster the tweets. Each
    cluster is related to a community.

    :return: the tweets' topics and communities
    :rtype: tuple
    """

    start = timer()

    # Load the files
    tfidf_matrix = pickle.load(open('TF-IDF Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    # Obtain the tweets' document topic distribution
    tweets_topic_distribution = LDA(n_topics=n_topics,max_iter=10,batch_size=128).fit_transform(tfidf_matrix)

    # Sample a topic for each tweet from its document topic distribution
    tweets_topic = []
    for tweet in range(n_tweets):
        normalize_factor = sum(tweets_topic_distribution[tweet])
        topic_distribution = tweets_topic_distribution[tweet]/normalize_factor
        topic_assignment = np.random.multinomial(1, topic_distribution).argsort()[::-1][0]
        tweets_topic.append(topic_assignment)

    # Cluster the tweets by its document topic distribution and assign the cluster label as community
    km = KMeans(n_clusters=n_communities, max_iter=300, n_init=10).fit(tweets_topic_distribution)
    tweets_community = km.labels_

    end = timer()
    print('Assigned a initial topic and community in %.2f seconds' %(end-start))
    txt_file.write('Assigned a initial topic and community in %.2f seconds. \n' %(end-start))

    # Return the tweets' topics and communities
    return tweets_topic, tweets_community


def markov_chain_convergence(n_iterations=1):
    """
    Performs a Markov Chain Convergence to obtain a topic and community assignment for each tweet.

    :param n_iterations: number of iterations of the Markov Chain Convergence
    :type n_iterations: int

    :return:
    """

    start = timer()

    # Load the files
    term_document_matrix = pickle.load(open('Term Document Matrix - ' + str(n_data) + ' Tweets.p', 'rb'))

    s = timer()
    # Build and compute the user-topic and user-topic-communities count
    user_topics_count = np.zeros((n_users, n_topics), dtype=int)
    user_topic_communities_count = np.zeros((n_users, n_topics, n_communities), dtype=int)
    users_count = np.zeros(n_users, dtype=int)

    for index in range(n_tweets):
        user = tweets_user[index]
        topic = tweets_topic[index]
        community = tweets_community[index]

        user_topics_count[user][topic] += 1
        user_topic_communities_count[user][topic][community] += 1
        users_count[user] += 1

    e = timer()
    print('Built the user counts in %.2f seconds' % (e - s))
    txt_file.write('Built the user counts in %.2f seconds. \n' % (e - s))

    for i in range(n_iterations):
        # For each tweet, obtain a topic and community assignment

        # Obtain the conditional probability distributions
        for tweet, words_count in zip(range(n_tweets), term_document_matrix.todense().tolist()):
            user = tweets_user[tweet]
            p_z = []    # Store p(z_p=z|u_p=u) for all z
            p_c = []    # Store p(c_p=c|u_p=u, z_p=z) for all z and c
            for topic in range(n_topics):
                # Obtain the conditional probability p(z_p=z|u_p=u)
                n_z = user_topics_count[user][topic] - 1
                N_z = users_count[user] - 1
                p_z.append((n_z + n_users)/(N_z + n_users*n_topics))

                p_c.append([])
                for community in range(n_communities):
                    # Obtain the conditional probability p(c_p=c|u_o=u, z_p=z) for all z and c
                    n_c = user_topic_communities_count[user][topic][community] - 1
                    N_c = user_topics_count[user][topic] - 1
                    p_c[topic].append((n_c + n_users*n_topics)/(N_c + n_users*n_topics*n_communities))

            # Obtain the joint probability distribution
            p = []  # Store p(z_p=z, c_p=c, W_p=W)
            for topic in range(n_topics):
                for community in range(n_communities):
                    p.append(p_z[topic]*p_c[topic][community])

            # Sample a topic and community assignment and update the user-topic-communities count
            old_topic = tweets_topic[tweet]
            old_community = tweets_community[tweet]
            user_topics_count[user][old_topic] -= 1
            user_topic_communities_count[user][old_topic][old_community] -= 1

            assignment = np.random.multinomial(1, p).argsort()[::-1][0]
            topic_assignment = int(math.floor(assignment/n_communities))
            community_assignment = assignment - topic_assignment*n_communities
            user_topics_count[user][topic_assignment] += 1
            user_topic_communities_count[user][topic_assignment][community_assignment] += 1

            tweets_topic[tweet] = topic_assignment
            tweets_community[tweet] = community_assignment

    # Save the result of the tweets' topic and community assignment
    pickle.dump(tweets_topic, open('Topics Assignment - ' + str(n_data) + ' Tweets.p', 'wb'))
    pickle.dump(tweets_community, open('Communities Assignment - ' + str(n_data) + ' Tweets.p', 'wb'))

    end = timer()
    print('Performed the Markov Chain Convergence in %.2f seconds' % (end-start))
    print('Number of iterations: %d.' % n_iterations)
    txt_file.write('Performed the Markov Chain Convergence in %.2f seconds. \n' % (end-start))
    txt_file.write('Number of iterations: %d. \n\n' % n_iterations)


def obtain_community_distributions():
    """
    Obtain the community topic distribution and the community user distribution. Use the user, topic and community
    assignments to estimate the conditional probabilities. Return the distributions.

    :return: community topic and user distribution
    :rtype: tuple
    """

    start = timer()

    # Build the community-topics count and community-users count
    s = timer()
    community_topics_count = np.zeros((n_communities, n_topics), dtype=int)
    community_users_count = np.zeros((n_communities, n_users), dtype=int)
    communities_count = np.zeros(n_communities, dtype=int)

    for tweet in range(n_tweets):
        user = tweets_user[tweet]
        topic = tweets_topic[tweet]
        community = tweets_community[tweet]

        community_topics_count[community][topic] += 1
        community_users_count[community][user] += 1
        communities_count[community] += 1
    e = timer()
    print('Built the community counts in %.2f seconds' % (e-s))
    txt_file.write('Built the community counts in %.2f seconds. \n' % (e-s))

    # Obtain and return the community topic and user distribution
    community_topic_distribution = []
    community_user_distribution = []
    for community in range(n_communities):
        s = timer()
        # Obtain the community topic distribution p(z_p=z|c_p=c)
        community_topic_distribution.append([])
        for topic in range(n_topics):
            n_z = max(community_topics_count[community][topic] - 1, 0)
            N_z = max(communities_count[community] - 1, 0)
            community_topic_distribution[community].append((n_z + n_users)/(N_z + n_users*n_topics))

        # Obtain the community user distribution p(u_p=u|c_p=c)
        community_user_distribution.append([])
        for user in set(tweets_user):
            n_u = max(community_users_count[community][user] - 1, 0)
            N_u = max(communities_count[community] - 1, 0)
            community_user_distribution[community].append(n_u/N_u)
        e = timer()
    end = timer()
    print('Obtained the community distributions in %.2f seconds' % (end-start))
    txt_file.write('Obtained the community distributions in %.2f seconds. \n' % (end-start))

    return community_topic_distribution, community_user_distribution


def print_community_distributions(community_topic_distribution, community_user_distribution):
    """
    Print the most relevant topics and most relevant users in each community

    :param community_topic_distribution: topic distribution in each community
    :type community_topic_distribution: list
    :param community_user_distribution: user distribution in each community
    :type community_user_distribution: list

    :return:
    """

    for community in range(n_communities):
        # Print the most relevant topics in each community
        print('Community %d most relevant topics: ' % community, end='')
        txt_file.write('Community %d most relevant topics: ' % community)
        ordered_topic_distribution = np.asarray(community_topic_distribution[community]).argsort()[::-1]
        for topic in ordered_topic_distribution[:n_relevant_topics]:
            print(topic, end=', ')
            txt_file.write('%d, ' % topic)
        print()
        txt_file.write('\n')

        # Print the most relevant users in each community
        print('Community %d most relevant users: ' % community, end='')
        txt_file.write('Community %d most relevant users: ' % community)
        ordered_user_distribution = np.asarray(community_user_distribution[community]).argsort()[::-1]
        for user in ordered_user_distribution[:n_relevant_users]:
            print(user, end=', ')
            txt_file.write('%d, ' % user)
        print()
        print()
        txt_file.write('\n\n')


def obtain_user_distributions():
    """
    Obtain the user topic distribution and the user community distribution. Use the user, topic and community
    assignments to estimate the conditional probabilities. Return the distributions.

    :return: user topic and community distribution
    :rtype: tuple
    """

    start = timer()

    # Build the user-topics count and user-communities count
    s = timer()
    users_count = np.zeros(n_users, dtype=int)
    user_topics_count = np.zeros((n_users, n_topics), dtype=int)
    user_communities_count = np.zeros((n_users, n_communities), dtype=int)

    for tweet in range(n_tweets):
        user = tweets_user[tweet]
        topic = tweets_topic[tweet]
        community = tweets_community[tweet]

        user_topics_count[user][topic] += 1
        user_communities_count[user][community] += 1
        users_count[user] += 1
    e = timer()
    print('Built the user counts in %.2f seconds' % (e - s))
    txt_file.write('Built the user counts in %.2f seconds. \n' % (e - s))

    # Obtain and return the user topic and community distribution

    user_topic_distribution = []
    user_community_distribution = []
    for user in range(n_users):
        # Obtain the user topic distribution p(z_p=z|u_p=u)
        user_topic_distribution.append([])
        for topic in range(n_topics):
            n_z = max(user_topics_count[user][topic] - 1, 0)
            N_z = max(users_count[user] - 1, 0)
            user_topic_distribution[user].append((n_z + n_users)/(N_z + n_users*n_topics))

        # Obtain the user community distribution p(c_p=c|u_p=u)
        user_community_distribution.append([])
        for community in range(n_communities):
            n_c = max(user_communities_count[user][community] - 1, 0)
            N_c = max(users_count[user] - 1, 0)
            user_community_distribution[user].append((n_c + n_users*n_topics)/(N_c + n_users*n_topics*n_communities))

    end = timer()
    print('Obtained the user distributions in %.2f seconds' % (end - start))
    txt_file.write('Obtained the user distributions in %.2f seconds. \n' % (end - start))

    return user_topic_distribution, user_community_distribution


def print_user_distributions(user_topic_distribution, user_community_distribution):
    """
    Print the most relevant topics and most relevant communities for each user.

    :param user_topic_distribution: topic distribution for each user
    :type user_topic_distribution: list
    :param user_community_distribution: community distribution for each user
    :type user_community_distribution: list

    :return:
    """

    for user in range(n_users):
        # Print the most relevant topics for each user
        print('User %d most relevant topics: ' % user, end='')
        txt_file.write('User %d most relevant topics: ' % user)
        ordered_topic_distribution = np.asarray(user_topic_distribution[user]).argsort()[::-1]
        for topic in ordered_topic_distribution[:n_relevant_topics]:
            print(topic, end=', ')
            txt_file.write('%d, ' % topic)
        print()
        txt_file.write('\n')

        # Print the most relevant communities for each user
        print('User %d most relevant communities: ' % user, end='')
        txt_file.write('User %d most relevant communities: ' % user)
        ordered_community_distribution = np.asarray(user_community_distribution[user]).argsort()[::-1]
        for community in ordered_community_distribution[:n_relevant_communities]:
            print(community, end=', ')
            txt_file.write('%d, ' % community)
        print()
        print()
        txt_file.write('\n\n')


# Start the timer
all_start = timer()

# Define the parameters
n_data = 1000
n_topics = 4
n_communities = 5
n_iterations = 50
n_relevant_topics = min(10, n_topics)
n_relevant_users = 10
n_relevant_communities = min(10, n_communities)

# Open a text file to store information
txt_file = open('Community Detection Output - ' + str(n_data) + ' Tweets, ' + str(n_topics) + ' Topics and ' +
    str(n_communities) + ' Communities.txt', "w")

# Step 0: Pre Processing - Obtaining useful values

# Import the tweets important information
tweets_user, tweets_artist = import_tweets_information()

# Map the tweets' user to integers
tweets_user = map_user_to_int(tweets_user=tweets_user)

# Obtain the number of tweets, the number of users and the number of words in the vocabulary
n_tweets = len(tweets_user)
n_users = len(set(tweets_user))
if n_users < n_relevant_users:
    n_relevant_users = n_users
n_words = number_words_in_vocabulary()

# Step 1: Initialization - assign a topic and a community for each tweet
tweets_topic, tweets_community = assign_topic_and_community(n_topics=n_topics, n_communities=n_communities)

# Step 2: Markov Chain Convergence - assigns a community and a topic for each post through iteration
markov_chain_convergence(n_iterations=n_iterations)

# Step 3: Inference - obtain the desired distributions

# Obtain the community distributions
community_topic_distribution, community_user_distribution = obtain_community_distributions()
print_community_distributions(community_topic_distribution=community_topic_distribution,
                              community_user_distribution=community_user_distribution)

# Obtain the user distributions
user_topic_distribution, user_community_distribution = obtain_user_distributions()
print_user_distributions(user_topic_distribution=user_topic_distribution,
                         user_community_distribution=user_community_distribution)

# Save the user distributions
pickle.dump(user_topic_distribution, open('User Topic Distribution - ' + str(n_data) + ' Tweets.p', 'wb'))
pickle.dump(user_community_distribution, open('User Community Distribution - ' + str(n_data) + ' Tweets.p', 'wb'))

# Print the total execution time
all_end = timer()
print('Community Detection Total Execution Time: %.2f seconds' % (all_end-all_start))
txt_file.write('Community Detection Total Execution Time: %.2f seconds. \n' % (all_end-all_start))

# Close the text file
txt_file.close()