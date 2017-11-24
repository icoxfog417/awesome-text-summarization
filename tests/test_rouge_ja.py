import os
import json
import sys
import unittest
from rougescore import rouge_n, rouge_l
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from sumeval.metrics.rouge import RougeCalculator


class TestRougeJA(unittest.TestCase):

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data/rouge")

    def load_test_data(self):
        test_file = os.path.join(self.DATA_DIR, "ROUGE-test-ja.json")
        with open(test_file, encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def _split(self, text):
        _txt = text.replace("。", " ").replace("、", " ").strip()
        words = _txt.split(" ")
        words = [w.strip() for w in words if w.strip()]
        return words

    def test_rouge(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=False, lang="ja")
        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    v = rouge.rouge_n(s, references, n)
                    b_v = rouge_n(self._split(s), 
                                  [self._split(r) for r in references],
                                  n, 0.5)
                    self.assertLess(abs(b_v - v), 1e-5)

    def test_rouge_with_stop_words(self):
        data = self.load_test_data()
        rouge = RougeCalculator(stopwords=True, lang="ja")

        def split(text):
            words = self._split(text)
            words = rouge.lang_setting.exec_stop_words(words)
            return words

        for eval_id in data:
            summaries = data[eval_id]["summaries"]
            references = data[eval_id]["references"]
            for n in [1, 2]:
                for s in summaries:
                    v = rouge.rouge_n(s, references, n)
                    b_v = rouge_n(split(s), 
                                  [split(r) for r in references],
                                  n, 0.5)
                    self.assertLess(abs(b_v - v), 1e-5)


if __name__ == "__main__":
    unittest.main()

