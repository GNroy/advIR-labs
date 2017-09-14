import PyPDF2
from os import path, listdir, getcwd


#dataset_directory = path.dirname(path.abspath(__file__))+'/'
dataset_directory = path.join(path.dirname(getcwd()), 'data', 'LinkedIn')

files = listdir(dataset_directory)


def read_pdfs(f_path):
    files_content = []
    for the_file in files:
        if the_file[-3:]!='pdf':
            continue
        pdfFileObj = open(path.join(f_path, the_file), 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        file_content = ""
        for page_ind,page in enumerate(range(pdfReader.numPages)):
            pageObj = pdfReader.getPage(page_ind)
            page_content = pageObj.extractText()
            file_content += page_content
        files_content.append(file_content)
        pdfFileObj.close()
    return files_content

documents = read_pdfs(dataset_directory)

sum_out = open("sum_train.txt", "w", encoding='utf8')
exp_out = open("exp_train.txt", "w", encoding='utf8')
edu_out = open("edu_train.txt", "w", encoding='utf8')

sum, exp, edu = [], [], []

for document in documents:
    sum_b = document.find('\nSummary\n')
    exp_b = document.find('\nExperience\n')
    edu_b = document.find('\nEducation\n')

    sum += [document[sum_b:exp_b]]
    exp += [document[exp_b:edu_b]]
    edu += [document[edu_b:]]

sum_out.write("*****".join(sum))
exp_out.write("*****".join(exp))
edu_out.write("*****".join(edu))

sum_out.close()
exp_out.close()
edu_out.close()
