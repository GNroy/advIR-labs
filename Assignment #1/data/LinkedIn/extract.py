import PyPDF2
from os import path, listdir, getcwd


dataset_directory = path.dirname(path.abspath(__file__))+'/'

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

documents = read_pdfs(dataset_directory)

sum_out = open("sum_train.txt","w")
exp_out = open("exp_train.txt","w")
edu_out = open("edu_train.txt","w")

for document in documents:
    sum_b = document.find('\nSummary\n')
    exp_b = document.find('\nExperience\n')
    edu_b = document.find('\nEducation\n')

    sum_out.write(document[sum_b:exp_b]+"\n")
    exp_out.write(document[exp_b:edu_b]+"\n")
    edu_out.write(document[edu_b:]+"\n")

sum_out.close()
exp_out.close()
edu_out.close()
