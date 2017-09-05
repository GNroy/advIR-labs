import math as m

'''
the function conditional_entropy() accepts an n-gram data structures as an input
that are dictionaries with words/word pairs as keys and the probabilities
of these words/word pairs as values

always test your code
'''

def conditional_entropy(two_gramm,one_gram):
    '''
    the phrase under consideration is 'Y X'
    therefore when you split the key into words use the correct order
    y,x = key.split()
    '''
    entropy = 0.
    for key in two_gramm:
        x,y = key.split()
	raise NotImplementedError
        '''
        write your code here
        use m.log(val,2) for computing logarithm
        '''
    return entropy

def test_conditional_entropy():
    raise NotImplementedError

if __name__ == "__main__":
    test_conditional_entropy()
