import os
from typing import Tuple, List, Dict, Any

from estimator import Sections, ResumeEstimator

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


class Report:
    def __init__(self, sections_values: Dict[str, SectionValues], report_text: List[str]):
        self.sections_values = sections_values
        self.report_text = report_text

    @staticmethod
    def _mean(l: List[Any]):
        return sum(l) / len(l)

    def accuracy(self, section=None):
        """
        Returns accuracy in the section, if specified. Otherwise - average accuracy in all sections
        """
        if section:
            s_values = self.sections_values.get(section)
            if not s_values:
                raise ValueError(section)

            return s_values.accuracy()

        return self._mean([v.accuracy() for v in self.sections_values.values()])

    def precision(self, section=None):
        if section:
            s_values = self.sections_values.get(section)
            if not s_values:
                raise ValueError(section)

            return s_values.precision()

        return self._mean([v.precision() for v in self.sections_values.values()])

    def recall(self, section=None):
        if section:
            s_values = self.sections_values.get(section)
            if not s_values:
                raise ValueError(section)

            return s_values.recall()

        return self._mean([v.recall() for v in self.sections_values.values()])

    def f1_score(self, section=None):
        if section:
            s_values = self.sections_values.get(section)
            if not s_values:
                raise ValueError(section)

            return s_values.f1_score()

        return self._mean([v.f1_score() for v in self.sections_values.values()])

    def relevant(self, section=None):
        if section:
            s_values = self.sections_values.get(section)
            if not s_values:
                raise ValueError(section)

            return s_values.relevant

        return sum([v.relevant for v in self.sections_values.values()])

    def selected(self, section=None):
        if section:
            s_values = self.sections_values.get(section)
            if not s_values:
                raise ValueError(section)

            return s_values.selected

        return sum([v.selected for v in self.sections_values.values()])

    def values_str(self):
        r_str = '\nBy sections:\n\n'
        for s, v in self.sections_values.items():
            r_str += 'Section: ' + s + '\n'
            r_str += 'Docs count: ' + str(self.relevant(s)) + '\n'
            r_str += 'Section selected: ' + str(self.selected(s)) + '\n'
            r_str += 'Accuracy: ' + str(self.accuracy(s)) + '\n'
            r_str += 'Precision: ' + str(self.precision(s)) + '\n'
            r_str += 'Recall: ' + str(self.recall(s)) + '\n'
            r_str += 'F1-score: ' + str(self.f1_score(s)) + '\n\n'

        r_str += 'Average overall:\n\n'
        r_str += 'Accuracy: ' + str(self.accuracy()) + '\n'
        r_str += 'Precision: ' + str(self.precision()) + '\n'
        r_str += 'Recall: ' + str(self.recall()) + '\n'
        r_str += 'F1-score: ' + str(self.f1_score()) + '\n\n'

        return r_str


def create_report(sections_docs_test, estimator, method):
    report_text = ''
    sections_values = {Sections.s: SectionValues(), Sections.e: SectionValues(), Sections.w: SectionValues()}

    for real_s_title, s_doc in sections_docs_test:
        est, predicted_s_title = estimator.estimate(s_doc, method)

        report_text += report_template.format(text=s_doc,
                                              results=str(est),
                                              expected=real_s_title,
                                              predicted=predicted_s_title,
                                              verdict='CORRECT' if real_s_title == predicted_s_title else 'INCORRECT')

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

    return Report(sections_values, report_text)


def run(method='mle'):
    sections_docs_training = read_training_docs()

    smoothing = 'dirichlet'
    estimator = ResumeEstimator(smoothing=smoothing, **sections_docs_training)

    _, d_count = estimator.sections_docs_count()
    print('Trained for', d_count, 'documents with total vocabulary size', estimator.total_voc_size())

    sections_docs_test = read_test_docs()
    report = create_report(sections_docs_test, estimator, method)

    print(report.values_str())

    if not os.path.exists(report_path):
        os.makedirs(report_path)
    with open(os.path.join(report_path, 'report.txt'), mode='w', encoding='utf-8') as f:
        f.write(report.report_text)

        print('\nReport saved at', os.path.join(report_path, 'report.txt'))


def estimate_mu():
    sections_docs_training = read_training_docs()
    estimator = ResumeEstimator(**sections_docs_training)
    print('\nEstimated mu: {}\n'.format(estimator.get_optimal_mu(start_mu=1.0, eps=1e-1, max_iters=100)))

if __name__ == '__main__':
    # Implemented methods: 'kld', 'mle'
    run(method='kld')

    estimate_mu()
