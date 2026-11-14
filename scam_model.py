"""
The model used in step 3 and the demo app.

It looks at a message two ways:
  - words (e.g. "KYC", "lottery", "OTP")
  - letter patterns of 2-5 characters (helps with Hindi/Marathi and typos like "Calll")
Then a logistic regression decides scam (1) or not scam (0).
"""

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TwoViewModel:
    """Word features + letter-pattern features, then logistic regression."""

    def __init__(self):
        # token_pattern: split only on spaces/punctuation, so Hindi/Marathi words with
        # vowel signs (like "आपका") stay whole. The default pattern chopped them ("आपक").
        self.words = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1,
                                     token_pattern=r"(?u)[^\s.,!?।:;\"'()\[\]*/|=+-]{2,}")
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    def _features(self, texts, fit=False):
        if fit:
            return hstack([self.words.fit_transform(texts), self.chars.fit_transform(texts)])
        return hstack([self.words.transform(texts), self.chars.transform(texts)])

    def fit(self, texts, y, weights):
        self.clf.fit(self._features(texts, fit=True), y, sample_weight=weights)
        return self

    def predict(self, texts):
        return self.clf.predict(self._features(texts))

    def scam_probability(self, texts):
        return self.clf.predict_proba(self._features(texts))[:, 1]

    def explain(self, text, top=5):
        """Return the words in this message that pushed it most toward 'scam'."""
        vocab = self.words.vocabulary_
        coefs = self.clf.coef_[0][: len(vocab)]  # first part of the features = words
        row = self.words.transform([text])
        pushes = []
        for idx in row.nonzero()[1]:
            word = self.words.get_feature_names_out()[idx]
            pushes.append((word, coefs[idx] * row[0, idx]))
        pushes.sort(key=lambda x: x[1], reverse=True)
        return [w for w, score in pushes[:top] if score > 0]
