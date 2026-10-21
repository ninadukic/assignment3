"""
Tokenizers used by the Naive Bayes classifier.
"""


class WhitespaceTokenizer:
    """
    Splits text based on whitespace. This is the same behavior as the
    text.split() function calls in Assignment 2, moved into its own class 
    so the classifier does not need to know how tokenization works.
    """

    def tokenize(self, text):
        """
        :param text: a string
        :return: a list of whitespace-separated tokens
        """
        return text.split()