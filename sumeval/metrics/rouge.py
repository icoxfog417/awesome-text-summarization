import re
from collections import Counter
from sumeval.metrics.lang_setting import LangSetting


class RougeCalculator():

    def __init__(self, tokenizer=None, stopwords=True, stemming=False,
                 word_limit=-1, length_limit=-1, lang="en"):
        self.tokenizer = tokenizer
        self.stemming = stemming
        self.stopwords = stopwords
        self.word_limit = word_limit
        self.length_limit = length_limit
        self.lang_setting = LangSetting.get(lang)

    def preprocess(self, text_or_words, is_reference=False):
        """
        reference:
        https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1820
        """
        words = text_or_words
        # tokenization
        if isinstance(words, str):
            if self.tokenizer:
                words = self.tokenizer.tokenize(text_or_words)
            else:
                words = self.lang_setting.tokenize(text_or_words)

        words = [w.strip().lower() for w in words if w.strip()]

        # limit length
        if self.word_limit > 0:
            words = words[:self.word_limit]
        elif self.length_limit > 0:
            _words = []
            length = 0
            for w in words:
                if length + len(w) < self.length_limit:
                    _words.append(w)
                else:
                    break
            words = _words

        if self.stopwords:
            words = self.lang_setting.exec_stop_words(words)

        if self.stemming and is_reference:
            # stemming is only adopted to reference
            # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1416

            # min_length ref: https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L2629
            words = self.lang_setting.exec_stemming(words, min_length=3)

        return words

    def len_ngram(self, words, n):
        return max(len(words) - n + 1, 0)

    def ngram_iter(self, words, n):
        for i in range(self.len_ngram(words, n)):
            n_gram = words[i:i+n]
            yield tuple(n_gram)

    def count_ngrams(self, words, n):
        c = Counter(self.ngram_iter(words, n))
        return c

    def count_overlap(self, summary_ngrams, reference_ngrams):
        result = 0
        for k, v in summary_ngrams.items():
            result += min(v, reference_ngrams[k])
        return result

    def rouge_n(self, summary, references, n, alpha=0.5):
        """
        alpha: alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))
        """
        _summary = self.preprocess(summary)
        summary_ngrams = self.count_ngrams(_summary, n)
        matches = 0
        count_for_recall = 0
        for r in references:
            _r = self.preprocess(r, True)
            r_ngrams = self.count_ngrams(_r, n)
            matches += self.count_overlap(summary_ngrams, r_ngrams)
            count_for_recall += self.len_ngram(_r, n)
        count_for_prec = len(references) * self.len_ngram(_summary, n)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1

    def _calc_f1(self, matches, count_for_recall, count_for_precision, alpha):
        def safe_div(x1, x2):
            return 0 if x2 == 0 else x1 / x2
        recall = safe_div(matches, count_for_recall)
        precision = safe_div(matches, count_for_precision)
        denom = (1.0 - alpha) * precision + alpha * recall
        return safe_div(precision * recall, denom)

    def lcs(self, a, b):
        longer = a
        base = b
        if len(longer) < len(base):
            longer, base = base, longer

        if len(base) == 0:
            return 0

        row = [0] * len(base)
        for c_a in longer:
            left = 0
            upper_left = 0
            for i, c_b in enumerate(base):
                up = row[i]
                if c_a == c_b:
                    value = upper_left + 1
                else:
                    value = max(left, up)
                row[i] = value
                left = value
                upper_left = up

        return left

    def rouge_l(self, summary, references, alpha=0.5):
        matches = 0
        count_for_recall = 0
        _summary = self.preprocess(summary)
        for r in references:
            _r = self.preprocess(r, True)
            matches += self.lcs(_r, _summary)
            count_for_recall += len(_r)
        count_for_prec = len(references) * len(_summary)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1
