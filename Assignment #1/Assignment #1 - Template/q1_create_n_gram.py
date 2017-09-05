import re

'''
the function create_n_gram() accepts two arguments
documents - the list of strings, where each string is the content of one
            document
n - the size of n-gram

the n-gram data structure is a dictionary that has the group of n words as a key
and the empirical probability of this words group as the value.

the function get_document_words() accepts a string (document) as an input
and returns the list of words or numbers in this string ordered by their
occurence in the original string
'''

def get_document_words(document):
    return list(filter(lambda x: x!= '', re.split(r"[^A-Z,a-z]", document.lower().replace(","," "))))

def create_n_gram(documents,n):
    n_gram = dict()
    total_occurence_count = 0.
    for document in documents:
        words = get_document_words(document)
        for i in range(len(words)-n+1):
            gram = " ".join(words[i:i+n])
            total_occurence_count += 1
            if gram in n_gram:
                n_gram[gram] += 1
            else:
                n_gram[gram] = 1

    for gram in n_gram:
        n_gram[gram] /= total_occurence_count

    return n_gram
