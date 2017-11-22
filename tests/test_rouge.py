import os
import json
import sys
import unittest
from rougescore import rouge_n, rouge_l
from pythonrouge.pythonrouge import Pythonrouge
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from sumeval.metrics.rouge import RougeCalculator


class TestRouge(unittest.TestCase):
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data/rouge")

    def load_test_data(self):
        test_file = os.path.join(self.DATA_DIR, "ROUGE-test.json")
        with open(test_file, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_rouge(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=False)
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    baseline = Pythonrouge(
                                summary_file_exist=False,
                                summary=[[s]],
                                reference=[[[r] for r in references]],
                                n_gram=n, recall_only=False,
                                length_limit=False,
                                stemming=False, stopwords=False)
                    b1_v = baseline.calc_score()
                    b2_v = rouge_n(rouge.preprocess(s),
                                   [rouge.preprocess(r) for r in references],
                                   n, 0.5)
                    v = rouge.rouge_n(s, references, n)
                    self.assertLess(abs(b2_v - v), 1e-5)
                    self.assertLess(abs(b1_v["ROUGE-{}-F".format(n)] - v), 1e-5) # noqa

    def test_rouge_with_stop_word(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=True)
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    baseline = Pythonrouge(
                                summary_file_exist=False,
                                summary=[[s]],
                                reference=[[[r] for r in references]],
                                n_gram=n, recall_only=False,
                                length_limit=False,
                                stemming=False, stopwords=True)
                    b1_v = baseline.calc_score()
                    b2_v = rouge_n(rouge.preprocess(s),
                                   [rouge.preprocess(r) for r in references],
                                   n, 0.5)
                    v = rouge.rouge_n(s, references, n)
                    self.assertLess(abs(b2_v - v), 1e-5)
                    self.assertLess(abs(b1_v["ROUGE-{}-F".format(n)] - v), 1e-5) # noqa

    def test_rouge_with_length_limit(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=True, length_limit=50)
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    baseline = Pythonrouge(
                                summary_file_exist=False,
                                summary=[[s]],
                                reference=[[[r] for r in references]],
                                n_gram=n, recall_only=False,
                                length_limit=True, length=50,
                                stemming=False, stopwords=True)
                    b1_v = baseline.calc_score()
                    b2_v = rouge_n(rouge.preprocess(s),
                                   [rouge.preprocess(r) for r in references],
                                   n, 0.5)
                    v = rouge.rouge_n(s, references, n)
                    self.assertLess(abs(b2_v - v), 1e-5)
                    self.assertLess(abs(b1_v["ROUGE-{}-F".format(n)] - v), 1e-5) # noqa

    def test_rouge_l(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=True)
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for s in summaries:
                baseline = Pythonrouge(
                            summary_file_exist=False,
                            summary=[[s]],
                            reference=[[[r] for r in references]],
                            n_gram=1, recall_only=False, ROUGE_L=True,
                            length_limit=True, length=50,
                            stemming=False, stopwords=True)
                b1_v = baseline.calc_score()
                b2_v = rouge_l(rouge.preprocess(s),
                               [rouge.preprocess(r) for r in references],
                               0.5)
                v = rouge.rouge_l(s, references)
                self.assertLess(abs(b2_v - v), 1e-5)
                self.assertLess(abs(b1_v["ROUGE-L-F"] - v), 1e-5)


if __name__ == "__main__":
    unittest.main()
