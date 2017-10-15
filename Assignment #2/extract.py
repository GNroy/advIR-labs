import PyPDF2
from os import path, listdir, makedirs

curr_dir = path.dirname(path.abspath(__file__))
dataset_directory = path.join(curr_dir, 'LinkedIn')
out_path = path.join(curr_dir, 'extracted')


def read_pdfs(dataset_dir, files):
    files_content = []
    for the_file in files:
        if the_file[-3:] != 'pdf':
            continue
        pdfFileObj = open(path.join(dataset_dir, the_file), 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        file_content = ""
        for page_ind, page in enumerate(range(pdfReader.numPages)):
            pageObj = pdfReader.getPage(page_ind)
            page_content = pageObj.extractText()
            file_content += page_content
        files_content.append(file_content)
        pdfFileObj.close()
    return files_content


def extract_docs(training_percents=90):
    assert 0 <= training_percents <= 100

    files = listdir(dataset_directory)

    documents = read_pdfs(dataset_directory, files)

    # by default, split 90% / 10% for training and test data sets respectively
    divider = int(len(documents) * training_percents / 100)

    training, test = documents[:divider], documents[divider:]

    save_extracted(training, 'training')
    save_extracted(test, 'test')


def save_extracted(documents, d_type):
    if not path.exists(out_path):
        makedirs(out_path)
    if d_type == 'test':
        sum_out = open(path.join(out_path, 'sum_test.txt'), 'w', encoding='utf-8')
        exp_out = open(path.join(out_path, 'exp_test.txt'), 'w', encoding='utf-8')
        edu_out = open(path.join(out_path, 'edu_test.txt'), 'w', encoding='utf-8')

    elif d_type == 'training':
        sum_out = open(path.join(out_path, 'sum_train.txt'), 'w', encoding='utf-8')
        exp_out = open(path.join(out_path, 'exp_train.txt'), 'w', encoding='utf-8')
        edu_out = open(path.join(out_path, 'edu_train.txt'), 'w', encoding='utf-8')

    else:
        raise NotImplementedError(d_type)

    for document in documents:
        sum_b = document.find('\nSummary\n')
        exp_b = document.find('\nExperience\n')
        edu_b = document.find('\nEducation\n')

        sum_out.write(document[sum_b:exp_b].replace('Summary', '').replace('Page', '') + "\n")
        exp_out.write(document[exp_b:edu_b].replace('Experience', '').replace('Page', '') + "\n")
        edu_out.write(document[edu_b:].replace('Education', '').replace('Page', '') + "\n")

    sum_out.close()
    exp_out.close()
    edu_out.close()

if __name__ == '__main__':
    extract_docs(training_percents=70)
