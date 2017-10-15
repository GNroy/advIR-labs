import os
from typing import Tuple, List, Dict, Any
import codecs

from estimator import Sections, Estimator

data_path = os.path.join(os.path.dirname(__file__), 'extracted')
report_path = os.path.join(os.path.dirname(__file__), 'reports')

report_template = \
    '''
"{text}"
----------------------------------------------------------------------
{results}
Expected section: {expected}
Predicted section: {predicted}
{verdict}
'''


def read_training_docs():
    sections_docs = dict()

    with open(os.path.join(data_path, 'edu_train.txt'), encoding='utf-8') as f_edu, \
            open(os.path.join(data_path, 'exp_train.txt'), encoding='utf-8') as f_exp, \
            open(os.path.join(data_path, 'sum_train.txt'), encoding='utf-8') as f_sum:
        sections_docs[Sections.e] = [text for text in f_edu.read().split('\n\n') if len(text) > 5]
        sections_docs[Sections.w] = [text for text in f_exp.read().split('\n\n') if len(text) > 5]
        sections_docs[Sections.s] = [text for text in f_sum.read().split('\n\n') if len(text) > 5]

    return sections_docs


def read_test_docs() -> List[Tuple[str, str]]:
    sections_docs = []

    with open(os.path.join(data_path, 'edu_test.txt'), encoding='utf-8') as f_edu, \
            open(os.path.join(data_path, 'exp_test.txt'), encoding='utf-8') as f_exp, \
            open(os.path.join(data_path, 'sum_test.txt'), encoding='utf-8') as f_sum:
        sections_docs.extend((Sections.e, text) for text in f_edu.read().split('\n\n') if len(text) > 5)
        sections_docs.extend((Sections.w, text) for text in f_exp.read().split('\n\n') if len(text) > 5)
        sections_docs.extend((Sections.s, text) for text in f_sum.read().split('\n\n') if len(text) > 5)

    return sections_docs


class SectionValues:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.relevant = 0
        self.selected = 0

    def accuracy(self):
        try:
            return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        except ZeroDivisionError:
            return 0

    def precision(self):
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0

    def recall(self):
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0

    def f1_score(self):
        try:
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        except ZeroDivisionError:
            return 0

def create_report(sections_docs_test, estimator, method):
    report_text = ''
    sections_values = {Sections.s: SectionValues(), Sections.e: SectionValues(), Sections.w: SectionValues()}
    if method == 'mle':
        args = [estimator.calculate_mu()]
    else:
        args = None

    for real_s_title, s_doc in sections_docs_test:
        est, predicted_s_title = estimator.estimate(s_doc, method, args=args)

        # update_selections
        real, predicted = sections_values[real_s_title], sections_values[predicted_s_title]
        real.relevant += 1
        predicted.selected += 1
        if real_s_title == predicted_s_title:
            predicted.tp += 1
            # increase TN of each other section
            for s, v in sections_values.items():
                if s != predicted_s_title:
                    v.tn += 1
        else:
            predicted.fp += 1
            real.fn += 1
            # increase TN of each other section
            for s, v in sections_values.items():
                if s != real_s_title and s != predicted_s_title:
                    v.tn += 1
    
    def values_str(sections_values):
        r_str = '\nBy sections:\n\n'
        for s, v in sections_values.items():
            r_str += 'Section: ' + s + '\n'
            r_str += 'Docs count: ' + str(sections_values.get(s).relevant) + '\n'
            r_str += 'Section selected: ' + str(sections_values.get(s).selected) + '\n'
            r_str += 'Accuracy: ' + str(sections_values.get(s).accuracy()) + '\n'
            r_str += 'Precision: ' + str(sections_values.get(s).precision()) + '\n'
            r_str += 'Recall: ' + str(sections_values.get(s).recall()) + '\n'
            r_str += 'F1-score: ' + str(sections_values.get(s).f1_score()) + '\n\n'

        r_str += 'Average overall:\n\n'
        accuracy = [v.accuracy() for v in sections_values.values()]
        r_str += 'Accuracy: ' + str(sum(accuracy) / len(accuracy)) + '\n'
        precision = [v.precision() for v in sections_values.values()]
        r_str += 'Precision: ' + str(str(sum(precision) / len(precision))) + '\n'
        recall = [v.recall() for v in sections_values.values()]
        r_str += 'Recall: ' + str(str(sum(recall) / len(recall))) + '\n'
        f1 = [v.f1_score() for v in sections_values.values()]
        r_str += 'F1-score: ' + str(str(sum(f1) / len(f1))) + '\n\n'

        return r_str
    return values_str(sections_values)


def run(verbose=True):
    methods = {'mle': 'Maximum Likelihood Estimator', 'kld': 'Kullback–Leibler divergence'}
    smoothings = {'Laplace': 'laplace', 'Dirichlet': 'dirichlet', 'No': ''}
    report = ''
    sections_docs_training = read_training_docs()
    estimator = Estimator(smoothing='', **sections_docs_training)
    _, d_count = estimator.sections_docs_count()
    report += 'Trained for' + str(d_count) + ' documents with total vocabulary size ' + str(estimator.total_voc_size()) + '\n'
    sections_docs_test = read_test_docs()
    for k, v in methods.items():
        report += '\n' + v + '\n'
        if k=='mle':
            for kk, vv in smoothings.items():
                report += '\n' + kk + ' smoothing ' + '\n'
                estimator = Estimator(smoothing=vv, **sections_docs_training)
                report += create_report(sections_docs_test, estimator, k)
        else:
            estimator = Estimator(smoothing='', **sections_docs_training)
            report += create_report(sections_docs_test, estimator, k)
    file = codecs.open('report.txt', 'a', encoding='utf-8')
    file.write(report)
    file.close()
    if verbose:
        print(report)

def estimate_mu(verbose=True):
    sections_docs_training = read_training_docs()
    estimator = Estimator(**sections_docs_training)
    report = '\nEstimated mu: {}\n\n'.format(estimator.calculate_mu())
    file = codecs.open('report.txt', 'w', encoding='utf-8')
    file.write(report)
    file.close()
    print(report)

if __name__ == '__main__':
    estimate_mu()
    # Run tests for Maximum Likelihood Estimator and Kullback–Leibler divergence
    # for dirichlet, laplace and no smoothing
    run()

