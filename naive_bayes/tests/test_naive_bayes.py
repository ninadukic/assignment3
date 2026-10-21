"""
Unit tests for the NaiveBayes class, using the worked example from
Jurafsky & Martin, Speech and Language Processing, Appendix B.3.

Training documents (tests/data/docs.txt):
    -  just plain boring
    -  entirely predictable and lacks energy
    -  no surprises and very few laughs
    +  very powerful
    +  the most fun film of the summer

With add-one smoothing (alpha=1), the textbook gives:
    P(-) = 3/5, P(+) = 2/5
    |V| = 20, negative class has 14 tokens, positive class has 9
    P(predictable|-) = 2/34   P(predictable|+) = 1/29
    P(no|-)          = 2/34   P(no|+)          = 1/29
    P(fun|-)         = 1/34   P(fun|+)         = 2/29
"""
from pathlib import Path

import pandas as pd
import pytest

from nb.nb import NaiveBayes

DOCS_PATH = Path(__file__).parent / "data" / "docs.txt"
NEG, POS = 0, 1
LABEL_TO_ID = {"-": NEG, "+": POS}


def train_on_textbook_example():
    """
    Loads docs.txt into a DataFrame and trains a NaiveBayes model on it.
    Each line of docs.txt is a label (- or +), a space, then the text.
    The label goes in the "author" column because that is the column name
    the NaiveBayes class reads labels from.
    """
    rows = []
    with open(DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                label, text = line.strip().split(" ", 1)
                rows.append({"author": LABEL_TO_ID[label], "text": text})
    df = pd.DataFrame(rows)
    return NaiveBayes(alpha=1).train(df)


def test_nb_class_priors():
    model = train_on_textbook_example()
    assert model.priors == pytest.approx([3 / 5, 2 / 5])


def test_nb_class_likelihoods():
    model = train_on_textbook_example()
    vocab = model.vocabulary
    lk = model.likelihoods

    # One row per class, one column per vocabulary word (no extra column)
    assert len(vocab) == 20
    assert lk.shape == (2, 20)

    assert lk[NEG, vocab["predictable"]] == pytest.approx(2 / 34)
    assert lk[POS, vocab["predictable"]] == pytest.approx(1 / 29)
    assert lk[NEG, vocab["no"]] == pytest.approx(2 / 34)
    assert lk[POS, vocab["no"]] == pytest.approx(1 / 29)
    assert lk[NEG, vocab["fun"]] == pytest.approx(1 / 34)
    assert lk[POS, vocab["fun"]] == pytest.approx(2 / 29)


def test_nb_class_likelihoods_sum_to_one():
    model = train_on_textbook_example()
    # Summing across the words (axis=1) gives one total per class
    row_sums = model.likelihoods.sum(axis=1)
    assert row_sums == pytest.approx([1.0, 1.0])