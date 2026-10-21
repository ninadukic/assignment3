"""
A multinomial Naive Bayes classifier with Lidstone smoothing.
"""
import numpy as np

from nb.utils.tokenizer import WhitespaceTokenizer


class NaiveBayes:
    """
    Multinomial Naive Bayes text classifier.

    Expects a pandas DataFrame with a "text" column and an "author" column
    whose labels are the integers 0, 1, ..., n_classes - 1.

    After train() runs, the fitted model is stored in three attributes:
        vocabulary  -- dict mapping each token to a column index
        priors      -- array of shape (n_classes,), P(class)
        likelihoods -- array of shape (n_classes, len(vocabulary)),
                       P(word | class); each row sums to 1
    """

    def __init__(self, alpha=0.1, tokenizer=None):
        """
        :param alpha: Lidstone smoothing parameter (alpha=1 is add-one)
        :param tokenizer: any object with a tokenize(text) method;
            defaults to whitespace tokenization
        """
        self.alpha = alpha
        if tokenizer is None:
            tokenizer = WhitespaceTokenizer()
        self.tokenizer = tokenizer
        self.vocabulary = None
        self.priors = None
        self.likelihoods = None

    def train(self, df):
        """
        Estimates the priors and likelihoods from labeled training data
        :param df: a pandas DataFrame with "text" and "author" columns
        :return: self, so calls can be chained
        """
        labels = df["author"].to_numpy()
        documents = [self.tokenizer.tokenize(text) for text in df["text"]]
        n_docs = len(documents)
        n_classes = df["author"].nunique()

        # Map every token in the training data to a column index. Sorting
        # makes the index of each word the same on every run.
        all_tokens = set()
        for tokens in documents:
            all_tokens.update(tokens)
        self.vocabulary = {token: idx for idx, token in
                           enumerate(sorted(all_tokens))}
        vocab_size = len(self.vocabulary)

        # Priors: share of training documents in each class
        self.priors = np.zeros(n_classes)
        for c in range(n_classes):
            self.priors[c] = np.sum(labels == c) / n_docs

        # Bag-of-words matrix: one row per document, one column per word
        training_matrix = np.zeros((n_docs, vocab_size))
        for i, tokens in enumerate(documents):
            for token in tokens:
                training_matrix[i, self.vocabulary[token]] += 1

        # Total count of each word within each class
        word_counts_per_class = np.zeros((n_classes, vocab_size))
        for c in range(n_classes):
            word_counts_per_class[c] = training_matrix[labels == c].sum(axis=0)

        # Likelihoods with Lidstone smoothing. The denominator adds alpha once
        # per vocabulary word, so each row is a probability distribution over
        # exactly the words in the matrix and sums to 1.
        self.likelihoods = np.zeros((n_classes, vocab_size))
        for c in range(n_classes):
            total_words_c = word_counts_per_class[c].sum()
            self.likelihoods[c] = ((word_counts_per_class[c] + self.alpha) /
                                   (total_words_c + self.alpha * vocab_size))
        return self

    def test(self, df):
        """
        Predicts a class for every document in df. Words that never appeared
        in the training data are skipped.
        :param df: a pandas DataFrame with a "text" column
        :return: a list of integer class predictions, one per row of df
        """
        if self.likelihoods is None:
            raise RuntimeError("Call train() before test().")

        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)

        class_predictions = []
        for text in df["text"]:
            test_vector = np.zeros(len(self.vocabulary))
            for token in self.tokenizer.tokenize(text):
                if token in self.vocabulary:
                    test_vector[self.vocabulary[token]] += 1
            # log P(class) + sum over words of count * log P(word | class)
            scores = log_priors + test_vector @ log_likelihoods.T
            class_predictions.append(int(np.argmax(scores)))
        return class_predictions