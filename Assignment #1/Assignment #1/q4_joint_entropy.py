import math as m

'''
the function joint_entropy() accepts an 2-gram data structures as an input
that is a dictionary with word pairs as keys and the probabilities
of these word pairs as values

always test your code
'''

def joint_entropy(two_gram):
    if True in [True if len(key.split())!=2 else False for key in two_gram.keys()]:
        raise ValueError('Argument must be an 2-gram')
    from q2_entropy import entropy
    return entropy(two_gram)

def test_joint_entropy(ds):
    from q1_create_n_gram import create_n_gram
    return joint_entropy(create_n_gram(ds, 2))

if __name__ == "__main__":
    ds = ['Mother washed the 2nd frame', 'The frog jumped unsuccessfully', 'I won 2nd prize']
    
    print(test_joint_entropy(ds))
