import math as m

'''
the function entropy() accepts an n-gram data structure as an input
that is a dictionary with words of the vocabulary as keys and the probabilities
of these words as values
'''

def entropy(n_gram):
    ent = 0.
    for key in n_gram.keys():
        ent -= n_gram[key] * m.log(n_gram[key], 2)
    return ent

def test_entropy(ds):
    from q1_create_n_gram import create_n_gram
    n_gram = create_n_gram(ds)
    return entropy(n_gram)

if __name__ == "__main__":
    ds = ['Mother washed the 2nd frame', 'The frog jumped unsuccessfully', 'I won 2nd prize']
    
    print(test_entropy(ds))
