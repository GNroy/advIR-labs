import math as m

'''
the function mutual_information() accepts an n-gram data structures as an input
that are dictionaries with words/word pairs as keys and the probabilities
of these words/word pairs as values

always test your code
'''

def mutual_information(two_gram,one_gram):
    '''
    the phrase under consideration is 'Y X'
    therefore when you split the key into words use the correct order
    y,x = key.split()
    '''
    m_inf = 0.
    for key in two_gram:
        x,y = key.split()
	raise NotImplementedError
        '''
        write your code here
        use m.log(val,2) for computing logarithm
        '''
    return m_inf

def test_mutual_information():
    raise NotImplementedError

if __name__ == "__main__":
    test_mutual_information()
