import os
import re
from pathlib import Path


class LangSetting():

    def __init__(self, lang):
        self.lang = lang
        self._stopwords = []
        self._stemming = {}
        self._symbol_replace = re.compile("[^A-Za-z0-9\-]")
        self._valid_word = re.compile("^[A-Za-z0-9\$]")

    def tokenize(self, text):
        _txt = text.replace("-", " - ")
        _txt = self._symbol_replace.sub(" ", _txt)
        words = _txt.split(" ")
        words = [w.strip() for w in words if w.strip()]
        words = [w for w in words if self._valid_word.match(w)]
        return words

    def exec_stop_words(self, words):
        if len(self._stopwords) == 0:
            self.load_stopwords()
        _words = [w for w in words if w not in self._stopwords]
        return _words

    def exec_stemming(self, words, min_length=-1):
        if len(self._stemming) == 0:
            self.load_stemming_dict()

        _words = []
        for w in words:
            if min_length > 0 and len(w) < min_length:
                _words.append[w]
            elif w in self._stemming:
                _words.append(self._stemming[w])
            else:
                _words.append[w]
        return _words

    def load_stopwords(self):
        p = Path(os.path.dirname(__file__))
        p = p.joinpath("data", self.lang, "stop_words.txt")
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                lines = f.readlines()
                lines = [ln.strip() for ln in lines]
                lines = [ln for ln in lines if ln]
            self._stopwords = lines

    def load_stemming_dict(self):
        p = Path(os.path.dirname(__file__))
        p = p.joinpath("data", self.lang, "stemming.txt")
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                lines = f.readlines()
                lines = [ln.strip() for ln in lines]
                lines = [ln for ln in lines if ln]
            self._stemming = dict(lines)

    @classmethod
    def get(cls, lang):
        if lang == "ja":
            return JASetting()
        else:
            return LangSetting("en")


class JASetting(LangSetting):

    def __init__(self):
        super().__init__("ja")
        from janome.tokenizer import Tokenizer
        self.tokenizer = Tokenizer()
        self._symbol_replace = re.compile("[^ぁ-んァ-ン一-龥ーa-zA-Zａ-ｚＡ-Ｚ0-9０-９]")

    def tokenize(self, text):
        _txt = self._symbol_replace.sub(" ", text)
        words = [t.surface for t in self.tokenizer.tokenize(_txt)]
        words = [w.strip() for w in words if w.strip()]
        return words
