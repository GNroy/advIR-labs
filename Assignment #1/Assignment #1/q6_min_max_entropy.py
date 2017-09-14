import math as m

'''
the function compute_min_max_entropy() accepts an n-gram data structure as an
input that is a dictionary with words of the vocabulary as keys and
the probabilities of these words as values

always test your code
'''

def compute_min_max_entropy(one_gram):
    return 0, m.log(len(one_gram), 2)

def test_compute_min_max_entropy(ds):
    return compute_min_max_entropy(ds)

if __name__ == "__main__":
    ds = ['Mother washed the 2nd frame', 'The frog jumped unsuccessfully', 'I won 2nd prize']

    print(test_compute_min_max_entropy(ds))
