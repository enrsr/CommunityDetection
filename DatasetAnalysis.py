import numpy as np
from timeit import default_timer as timer   # Import default_timer to measure processing time of parts of code
import pickle                               # Import pickle to store and restore the data


def obtain_key_count(key):
    """
    Receives a key from the tweet dictionary. Count the number of occurrence for each value of the key.

    :param key: one of the keys from the tweet dictionary
    :type key: str

    :return: dictionary with the number of occurrences of each unique value
    :rtype: dict
    """

    # Count the number of occurrence of each value in the key
    key_count = {}
    for tweet in tweets:
        key_value = tweet[key]
        if key_value in key_count:
            current_count = key_count[key_value]
            key_count.update({key_value: current_count + 1})
        else:
            key_count.update({key_value: 1})

    return key_count


def print_key_statistics(key, key_count):
    """
    Receives a key from the tweet dictionary. Uses the count of number of occurrences of each value to compute some
    simple statistics. Print the computed statistics.

    :param key: one of the keys from the tweet dictionary
    :type key: str
    :param key_count: dictionary with the number of occurrences of each unique value
    :type key_count: dict

    :return:
    """
    values = [value for value in key_count.values()]

    print('Number of unique %s: %d' % (key, len(values)))
    print('Average number of tweets per %s: %d' % (key, np.mean(values)))
    print('Maximum number of tweets per %s: %d' % (key, max(values)))
    print('Minimum number of tweets per %s: %d' % (key, min(values)))
    print('Standard deviation of tweets per %s: %d' % (key, np.std(values)))
    print()

    txt_file.write('Number of unique %s: %d \n' % (key, len(values)))
    txt_file.write('Average number of tweets per %s: %d \n' % (key, np.mean(values)))
    txt_file.write('Maximum number of tweets per %s: %d \n' % (key, max(values)))
    txt_file.write('Minimum number of tweets per %s: %d \n' % (key, min(values)))
    txt_file.write('Standard deviation of tweets per %s: %d \n\n' % (key, np.std(values)))

# Start the timer
start = timer()

# Load the files
n_data = 1000000
tweets = pickle.load(open('Tweets Data - ' + str(n_data) + ' Tweets.p', 'rb'))

# Open a text file to store information
txt_file = open("Dataset Analysis Output - " + str(n_data) + " Tweets.txt", "w")

# Distribution of number of tweets by keys
keys = ['artist', 'user', 'date', 'language']         # Keys to be used


# Count the number of occurrence for each unique value of each key and print some simple statistics about it
for key in keys:
    key_count = obtain_key_count(key=key)
    print_key_statistics(key=key, key_count=key_count)

# Print the code total execution time
end = timer()
print('Dataset Analysis Total Execution Time: %.2f' % (end-start))
print()
txt_file.write('Dataset Analysis Total Execution Time: %.2f' % (end-start))

# Close the text file
txt_file.close()