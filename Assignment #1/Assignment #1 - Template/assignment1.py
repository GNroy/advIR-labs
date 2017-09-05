import PyPDF2
from os import path, listdir, getcwd

from q1_create_n_gram import create_n_gram
from q2_entropy import entropy
from q3_conditional_entropy import conditional_entropy
from q4_joint_entropy import joint_entropy
from q5_mutual_information import mutual_information
from q6_min_max_entropy import compute_min_max_entropy

'''
This script is used to load all the PDF documents, calculate the frequencies of
1-gram and 2-grams, calculate the valuse of entropy, conditional entropy,
mutial information, etc.
'''

dataset_directory = path.dirname(path.abspath(__file__))+"/data/LinkedIn/"

files = listdir(dataset_directory)


def read_pdfs(path):
    files_content = []
    for the_file in files:
        if the_file[-3:]!='pdf':
            continue
        pdfFileObj = open(path+the_file, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        file_content = ""
        for page_ind,page in enumerate(range(pdfReader.numPages)):
            pageObj = pdfReader.getPage(page_ind)
            page_content = pageObj.extractText()
            file_content += page_content
        files_content.append(file_content)
        pdfFileObj.close()
    return files_content



files_content = read_pdfs(dataset_directory)

print "\n"

one_gram = create_n_gram(files_content,1)
voc = one_gram.keys()
print "Vocabulary size: " + repr(len(voc))

two_gram = create_n_gram(files_content,2)
print "Number of 2-grams: " + repr(len(two_gram.keys()))

min_e,max_e = compute_min_max_entropy(voc)
print "Minimum entropy: " + repr(min_e) + "\nMaximum entropy: " + repr(max_e)

ent = entropy(one_gram)
print "Entropy wtr given vocabulary: " + repr(ent)

cond_e = conditional_entropy(two_gram,one_gram)
print "Conditional entropy of a 2-word phrase \'Y X\' H(Y|X): " + repr(cond_e)

m_inf = mutual_information(two_gram,one_gram)
print "Mutual information of words in 2-word phrase \'Y X\' I(X,Y): " + repr(m_inf)

j_ent = joint_entropy(two_gram)
print "Joint entropy of two words in a phrase \'Y X\' H(X,Y): " + repr(j_ent)

print "\n"
