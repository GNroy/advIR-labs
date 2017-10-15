import re
from collections import defaultdict, namedtuple
from functools import reduce
from itertools import chain
from math import log, inf
from typing import List, Dict, Tuple


class Sections:
    w = 'work_experience'
    e = 'education'
    s = 'summary'


_SectionsEstimation = namedtuple('SectionsEstimation', [Sections.w, Sections.e, Sections.s])


class SectionsEstimation:
    def __init__(self, **sections):
        self.__dict__ = sections

        # self._normalize()

    def _normalize(self):
        sum_value = sum(self.__dict__.values())

        for k in self.__dict__.keys():
            self.__dict__[k] = self.__dict__[k] / sum_value

    def arg_max(self):
        max_val, max_arg = -inf, 'unknown'
        for k, v in self.__dict__.items():
            try:
                v = float(v)
            except TypeError:
                continue
            if v > max_val:
                max_arg, max_val = k, v

        return max_arg

    def __str__(self):
        return str(_SectionsEstimation(**self.__dict__))

    def arg_min(self):
        min_val, min_arg = inf, 'unknown'
        for k, v in self.__dict__.items():
            try:
                v = float(v)
            except TypeError:
                continue
            if v < min_val:
                min_arg, min_val = k, v

        return min_arg


class Vocabulary:
    _word_pattern = re.compile('[A-Za-zа-яА-Я]+')
    _stop_words = ['of', 'and', 'i', 'the', 'at', 'to', 'for', 'in', 'on', 'a', 'with', 'as']

    def __init__(self, *documents: str):
        self._docs = list(documents)

        self._gen_voc()

    def _gen_voc(self):
        self._voc = defaultdict(int)
        self._size = 0

        for doc in self._docs:
            doc_lemmas = [w.lower()
                          for w in re.findall(self._word_pattern, doc)
                          if w.lower() not in self._stop_words]

            for l in doc_lemmas:
                self._voc[l] += 1
                self._size += 1

    def count(self, lemma, default=0) -> int:
        return self._voc.get(lemma, default)

    def prob(self, lemma) -> float:
        default_prob = 1e-7

        if not self._size:
            raise ValueError('Empty vocabulary')

        return (self.count(lemma) / self._size) if self.count(lemma) else default_prob

    def lemmas(self) -> List[str]:
        return list(self._voc.keys())

    def total_count(self) -> int:
        return self._size

    def unique_count(self) -> int:
        return len(self._voc.keys())

    def docs(self):
        return self._docs

    def union(self, other):
        return Vocabulary(*chain(self._docs, other.docs()))


class ResumeEstimator:
    def __init__(self, smoothing=None, **sections_docs: Dict[str, List[str]]):
        """
        :param smoothing: 'laplace', 'dirichlet' or None
        :param sections_docs: section titles with their docs
        """
        self._sections_vocs = {s_title: Vocabulary(*s_docs)
                               for s_title, s_docs in sections_docs.items()}

        if smoothing not in ('laplace', 'dirichlet', None):
            print('Warning: unknown smoothing method %s' % smoothing)
        self._smoothing = smoothing

        self._sections_count = len(self._sections_vocs.keys())
        self._docs_count = sum(len(s) for s in sections_docs.values())

        self._total_voc = reduce(lambda v1, v2: v1.union(v2), self._sections_vocs.values())

    def sections_docs_count(self):
        return self._sections_count, self._docs_count

    def total_voc_size(self):
        return self._total_voc.total_count()

    def total_voc(self) -> Vocabulary:
        return self._total_voc

    def get_optimal_mu(self, start_mu=1.0, eps=1e-1, max_iters=100):
        """
        Using Newton method, estimates parameter µ for a Dirichlet method
        """

        def leave_one_out(mu, sign=1.0):
            f_value = 0

            for v in self._sections_vocs.values():
                voc_size = v.total_count()
                for lemma in self._total_voc.lemmas():
                    p_ref = self._total_voc.prob(lemma)
                    l_count = v.count(lemma, default=1)

                    f_value += sign * l_count * (log(l_count - 1 + mu * p_ref) - log(voc_size - 1 + mu))

            return f_value
        
        from scipy.optimize import minimize
        res = minimize(leave_one_out, start_mu, args=(-1.0,))
        return res.x[0]
        '''
        def der_leave_one_out(mu):
            f_value = 0

            for v in self._sections_vocs.values():
                voc_size = v.total_count()
                for lemma in self._total_voc.lemmas():
                    p_ref = self._total_voc.prob(lemma)
                    l_count = v.count(lemma, default=1)

                    f_value += l_count * ((voc_size - 1) * p_ref - l_count + 1) / (
                        (voc_size - 1 + mu) * (l_count - 1 + mu * p_ref))

            return f_value

        def der2_leave_one_out(mu):
            f_value = 0

            for v in self._sections_vocs.values():
                voc_size = v.total_count()
                for lemma in self._total_voc.lemmas():
                    p_ref = self._total_voc.prob(lemma)
                    l_count = v.count(lemma, default=1)

                    f_value -= l_count * ((voc_size - 1) * p_ref - l_count + 1) * (
                        2 * mu * p_ref + (voc_size - 1) * p_ref + l_count - 1) / (
                        pow(voc_size - 1 + mu, 2) * pow(l_count - 1 + mu * p_ref, 2))

            return f_value

        optimal_mu, curr_mu = start_mu, start_mu
        iteration = 0

        while iteration < max_iters:
            print('Iteration', iteration, ', Mu=', optimal_mu, 'f_mu=', leave_one_out(optimal_mu))
            optimal_mu -= der_leave_one_out(optimal_mu) / der2_leave_one_out(optimal_mu)
            if abs(leave_one_out(curr_mu) - leave_one_out(optimal_mu)) < eps:
                break

            curr_mu = optimal_mu
            iteration += 1

        return optimal_mu
        '''

    def _mle_for_section(self, s_voc: Vocabulary, doc_voc: Vocabulary) -> float:
        def smooth(lemma, method='laplace'):
            if method == 'laplace':
                _lambda = 0.9

                return (1 - _lambda) * s_voc.prob(lemma) + _lambda * self._total_voc.prob(lemma)

            elif method == 'dirichlet':
                mu = 8.5
                # mu = 1082614019
                section_words_count, section_lemma_count = s_voc.total_count(), s_voc.count(lemma)
                p_ref = self._total_voc.prob(lemma)

                return (section_lemma_count + mu * p_ref) / (section_words_count + mu)

            else:
                return s_voc.prob(lemma)

        return sum(log(smooth(l, self._smoothing))
                   for l in doc_voc.lemmas())

    def mle(self, doc: str):
        estimations = dict()

        doc_voc = Vocabulary(doc)

        for s_title, s_voc in self._sections_vocs.items():
            estimations[s_title] = self._mle_for_section(s_voc, doc_voc)

        return SectionsEstimation(**estimations)

    def _kld_for_section(self, s_voc: Vocabulary, doc_voc: Vocabulary):
        score = 0
        for l in doc_voc.lemmas():
            p_s, p_d = s_voc.prob(l), doc_voc.prob(l)

            score += p_d * (log(p_d) - log(p_s))

        return score

    def kld(self, doc: str):

        estimations = dict()

        doc_voc = Vocabulary(doc)

        for s_title, s_voc in self._sections_vocs.items():
            estimations[s_title] = self._kld_for_section(s_voc, doc_voc)

        return SectionsEstimation(**estimations)

    def estimate(self, doc: str, method='mle') -> Tuple[SectionsEstimation, str]:
        """
        Estimates section of the given document
        :param doc: document to label
        :param method: 'mle' or 'kld'
        :return: section title
        """
        if method == 'mle':
            est = self.mle(doc)
            return est, est.arg_max()

        elif method == 'kld':
            est = self.kld(doc)
            return est, est.arg_min()

        else:
            raise NotImplementedError('Unknown method %s' % str(method))