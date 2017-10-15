import os

import PyPDF2

from est_linkedin import read_training_docs
from estimator import ResumeEstimator

data_path = os.path.join(os.path.dirname(__file__), 'other/')
report_path = os.path.join(os.path.dirname(__file__), 'reports/')


def read_test_docs(docs_count=10):
    # skip not-pdf
    pdf_files_titles = list(filter(lambda f_name: f_name[-3:].lower() == 'pdf', os.listdir(data_path)))

    assert 0 < docs_count < len(pdf_files_titles)

    # stay only docs_count docs
    pdf_files_titles = sorted(pdf_files_titles)[:docs_count]

    files_content = []
    for f_title in pdf_files_titles:
        with open(data_path + f_title, mode='rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)

            file_content = ''
            for page_ind, page in enumerate(range(pdf_reader.numPages)):
                page = pdf_reader.getPage(page_ind)

                page_content = page.extractText()
                file_content += page_content

            files_content.append(file_content)

    return files_content


def run():
    sections_docs_training = read_training_docs()

    estimator = ResumeEstimator(**sections_docs_training)

    _, d_count = estimator.sections_docs_count()
    print('Trained for', d_count, 'documents with total vocabulary size', estimator.total_voc_size())

    files_contents = read_test_docs()

    for f_content in files_contents:
        print(estimator.mle(f_content))


if __name__ == '__main__':
    run()
