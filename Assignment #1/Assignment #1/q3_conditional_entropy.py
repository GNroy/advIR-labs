import math as m

'''
the function conditional_entropy() accepts an n-gram data structures as an input
that are dictionaries with words/word pairs as keys and the probabilities
of these words/word pairs as values
'''

def conditional_entropy(two_gram, one_gram, biass_coeff=1):
    ent = 0.
	
    for key in two_gram.keys():
        ent -= two_gram[key] * m.log(two_gram[key] / (one_gram[' '.join(key.split()[:-1])] * biass_coeff), 2)
    return ent

def test_conditional_entropy(ds):
    from q1_create_n_gram import create_n_gram
    return conditional_entropy(create_n_gram(ds, 2), create_n_gram(ds))

if __name__ == "__main__":
    ds = ['Mother washed the 2nd frame', 'The frog jumped unsuccessfully', 'I won 2nd prize']
	
    print(test_conditional_entropy(ds))
