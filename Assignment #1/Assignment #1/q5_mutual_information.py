import math as m

'''
the function mutual_information() accepts an n-gram data structures as an input
that are dictionaries with words/word pairs as keys and the probabilities
of these words/word pairs as values

always test your code
'''

def mutual_information(two_gram, one_gram, bias_coeff=1):
    m_inf = 0.
    for key in two_gram.keys():
        tmp_split = key.split()
        x, y = ' '.join(tmp_split[:-1]), tmp_split[-1]
        m_inf += two_gram[key] * m.log(two_gram[key] / (one_gram[x] * one_gram[x] * bias_coeff**2), 2)
    return m_inf

def test_mutual_information():
    from q1_create_n_gram import create_n_gram
    return mutual_information(create_n_gram(ds, 2), create_n_gram(ds))

if __name__ == "__main__":
    ds = ['Mother washed the 2nd frame', 'The frog jumped unsuccessfully', 'I won 2nd prize']
    
    print(test_mutual_information(ds))
