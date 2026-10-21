"""
Builds an inverted index over a collection of 20 Newsgroups documents
stored as XML, then answers simple boolean AND queries against it.

Usage:
    python index.py -f [path_to_collection] -t [tokenizer] -s
    e.g. python index.py -f 20newsgroups-initial.xml -t whitespace -s

Tokenizers: whitespace, nltk, ngram. The -s flag turns on Porter stemming.
"""
import argparse
import string
import xml.etree.ElementTree as ET
from collections import Counter

from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


# (A) Read and parse the XML collection

def parse_collection(path):
    """
    Parses an XML file laid out as
        <root>
          <doc>
            <docid>3</docid>
            <msgtext>From: ... Subject: ... message body</msgtext>
          </doc>
          ...
        </root>
    :param path: path to an XML file such as 20newsgroups-initial.xml
    :return: a list of (docid, text) tuples, one per document
    """
    root = ET.parse(path).getroot()
    documents = []
    for doc in root.findall("doc"):
        docid = doc.findtext("docid").strip()
        text = doc.findtext("msgtext", default="")
        documents.append((docid, text))
    return documents


# (B) Tokenizers

class Tokenizer:
    """
    Interface for every tokenizer. Two properties hold for all tokenizers:
      1. tokenize(text) takes one string and returns a list of string tokens.
      2. Every tokenizer's output is normalized the same way (lowercased,
         surrounding punctuation stripped, empty tokens dropped), so the
         indexes differ only in how the text was split.
    Subclasses must override tokenize(); normalize() and process() are shared.
    """

    def tokenize(self, text):
        raise NotImplementedError("Subclasses must implement tokenize()")

    def normalize(self, tokens):
        normalized = [token.lower().strip(string.punctuation)
                      for token in tokens]
        return [token for token in normalized if token.strip()]

    def process(self, text):
        return self.normalize(self.tokenize(text))


class WhitespaceTokenizer(Tokenizer):
    """Splits on runs of whitespace."""

    def tokenize(self, text):
        return text.split()


class NLTKTokenizer(Tokenizer):
    """
    Uses NLTK's off-the-shelf Penn Treebank tokenizer, which splits
    punctuation and contractions off words ("system's" -> "system", "'s").
    It needs no extra data downloads.
    """

    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)


class NgramTokenizer(Tokenizer):
    """
    Splits text into overlapping character n-grams. Runs of whitespace are
    collapsed to a single space first, so "the  system" with n=4 gives
    "the ", "he s", "e sy", " sys", "syst", "yste", "stem".
    """

    def __init__(self, n=4):
        self.n = n

    def tokenize(self, text):
        text = " ".join(text.split())
        return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]


TOKENIZERS = {
    "whitespace": WhitespaceTokenizer,
    "nltk": NLTKTokenizer,
    "ngram": NgramTokenizer,
}



# (C) Stemming

STEMMER = PorterStemmer()


def stem(tokens):
    """
    :param tokens: a list of tokens
    :return: the Porter-stemmed tokens
    """
    return [STEMMER.stem(token) for token in tokens]


def analyze(text, tokenizer, use_stemming):
    """
    Runs the full text-processing pipeline used for both documents and
    queries: tokenize, normalize, and optionally stem.
    """
    tokens = tokenizer.process(text)
    if use_stemming:
        tokens = stem(tokens)
    return tokens


# (D) Building the inverted index

def build_index(documents, tokenizer, use_stemming):
    """
    :param documents: list of (docid, text) tuples
    :param tokenizer: a Tokenizer instance
    :param use_stemming: whether to Porter-stem tokens
    :return: dict mapping each term to its postings list, a list of
        (docid, term frequency) tuples
    """
    index = dict()
    for docid, text_body in documents:
        tokens = analyze(text_body, tokenizer, use_stemming)
        term_frequency = Counter(tokens)
        for token in term_frequency:
            if token in index:
                index[token].append((docid, term_frequency[token]))
            else:
                index[token] = [(docid, term_frequency[token])]
    return index


# (E)-(G) Querying

def search(index, query, tokenizer, use_stemming):
    """
    Returns the ids of documents that contain every query term (boolean AND).
    Each query word goes through the same pipeline as the documents, so the
    search is case-insensitive. With the n-gram tokenizer, a word becomes
    several n-grams and a document must contain all of them.
    :return: a sorted list of document ids
    """
    terms = []
    for word in query.split():
        terms.extend(analyze(word, tokenizer, use_stemming))
    if not terms:
        return []

    matching = None
    for term in terms:
        docids = {docid for docid, tf in index.get(term, [])}
        if matching is None:
            matching = docids
        else:
            matching = matching & docids
    return sorted(matching, key=int)


def main():
    parser = argparse.ArgumentParser(description="Build an inverted index")
    parser.add_argument("-f", "--file", required=True,
                        help="Path to the XML collection")
    parser.add_argument("-t", "--tokenizer", choices=TOKENIZERS,
                        default="whitespace", help="Tokenizer to use")
    parser.add_argument("-s", "--stem", action="store_true",
                        help="Apply Porter stemming")
    parser.add_argument("-n", type=int, default=4,
                        help="n for the ngram tokenizer (default: 4)")
    args = parser.parse_args()

    if args.tokenizer == "ngram":
        tokenizer = NgramTokenizer(n=args.n)
    else:
        tokenizer = TOKENIZERS[args.tokenizer]()

    documents = parse_collection(args.file)
    index = build_index(documents, tokenizer, args.stem)
    print(f"Collection: {args.file}")
    print(f"Tokenizer: {args.tokenizer}, stemming: {args.stem}")
    print(f"Documents: {len(documents)}, unique terms: {len(index)}\n")

    queries = ["system", "compatibility", "system compatibility"]
    for query in queries:
        results = search(index, query, tokenizer, args.stem)
        print(f'Query "{query}": {len(results)} documents')
        print(f"  {results}\n")


if __name__ == "__main__":
    main()