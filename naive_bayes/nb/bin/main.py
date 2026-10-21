"""
Command-line interface: trains the Naive Bayes classifier on the Kennedy and
Johnson speeches and predicts the author of each unlabeled speech.

Run from the naive_bayes/ folder:
    python -m nb.bin.main -f data
"""
import argparse

from sklearn import metrics

from nb.nb import NaiveBayes
from nb.utils.load_data import ID_TO_AUTHOR, build_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True,
                        help="Data directory containing kennedy/, johnson/ "
                             "and unlabeled/ subfolders")
    parser.add_argument("-a", "--alpha", type=float, default=0.1,
                        help="Lidstone smoothing parameter (default: 0.1)")
    return parser.parse_args()


def main():
    # (a) Ask the user for an input directory
    args = parse_args()

    # (b) Build the DataFrames from the text in the data subfolders
    training_df, test_df = build_dataframe(args.indir)
    print(f"Loaded {len(training_df)} training speeches and "
          f"{len(test_df)} unlabeled speeches.\n")

    # (c) + (d) Run the train method, which estimates priors and likelihoods
    model = NaiveBayes(alpha=args.alpha)
    model.train(training_df)
    for class_id, prior in enumerate(model.priors):
        print(f"Prior P({ID_TO_AUTHOR[class_id]}) = {prior:.3f}")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print(f"Likelihood matrix shape: {model.likelihoods.shape}\n")

    # (e) Run the test method on the unlabeled speeches
    predictions = model.test(test_df)

    # (f) Print the prediction for each file
    print(f"{'File':<24}{'Predicted':<12}{'Actual':<10}Correct")
    for filename, actual, predicted in zip(test_df["filename"],
                                           test_df["author"], predictions):
        correct = "yes" if actual == predicted else "no"
        print(f"{filename:<24}{ID_TO_AUTHOR[predicted]:<12}"
              f"{ID_TO_AUTHOR[actual]:<10}{correct}")

    n_correct = sum(a == p for a, p in zip(test_df["author"], predictions))
    accuracy = metrics.accuracy_score(test_df["author"], predictions)
    f1 = metrics.f1_score(test_df["author"], predictions)
    print(f"\nAccuracy: {accuracy:.2%} ({n_correct}/{len(predictions)} "
          f"correct), F1: {f1:.2%}")


if __name__ == "__main__":
    main()