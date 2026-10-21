# Naive Bayes Speech Classifier

A multinomial Naive Bayes classifier, written from scratch with NumPy, that predicts whether a presidential speech was given by John F. Kennedy or Lyndon B. Johnson. It trains on labeled speeches, estimates class priors and word likelihoods with Lidstone smoothing, and predicts the author of each unlabeled speech.

This is a refactor of our Assignment 2 script into an installable Python project, with a `NaiveBayes` class and unit tests.

## Project structure

```
naive_bayes/
    data/                    raw speeches (ignored by git)
        kennedy/             training speeches by Kennedy
        johnson/             training speeches by Johnson
        unlabeled/           test speeches, true author in the file name
    nb/
        nb.py                NaiveBayes class with train() and test()
        bin/main.py          command-line interface
        utils/load_data.py   reads the speeches into pandas DataFrames
        utils/tokenizer.py   WhitespaceTokenizer
    tests/
        data/docs.txt        Jurafsky & Martin worked example
        test_naive_bayes.py  unit tests for priors and likelihoods
```

## Installation

From the `naive_bayes/` folder:

```bash
conda env create -f environment.yml
conda activate dsan5400
pip install -e .
```

## Usage

```bash
python -m nb.bin.main -f data
```

| Flag | Meaning |
| --- | --- |
| `-f`, `--indir` | Data directory containing `kennedy/`, `johnson/` and `unlabeled/` (required) |
| `-a`, `--alpha` | Lidstone smoothing parameter (default `0.1`) |

The program prints the priors, the vocabulary size, the shape of the likelihood matrix, the predicted author for each unlabeled file, and the overall accuracy and F1 score.

## Tests

```bash
pytest tests/
```

The tests train on the five-document example from Jurafsky & Martin (Appendix B.3) with add-one smoothing and check that the priors and likelihoods match the textbook and that each class's likelihoods sum to 1.