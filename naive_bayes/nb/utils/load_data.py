"""
Utility functions for reading the presidential speeches from disk into
pandas DataFrames.
"""
from pathlib import Path

import pandas as pd

AUTHOR_TO_ID = {"kennedy": 0, "johnson": 1}
ID_TO_AUTHOR = {idx: name for name, idx in AUTHOR_TO_ID.items()}


def make_df_from_dir(folder, label_from_filename=False):
    """
    Reads every .txt file in a folder into a DataFrame with one row per file
    :param folder: a Path to a directory of .txt files
    :param label_from_filename: if False, the author is the folder name
        (kennedy/ or johnson/); if True, the author is the last part of the
        file name (speech_10_johnson.txt -> johnson), as in unlabeled/
    :return: a pandas DataFrame with columns filename, author, text
    """
    rows = []
    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        if label_from_filename:
            author = file_path.stem.split("_")[-1]
        else:
            author = folder.name
        rows.append({
            "filename": file_path.name,
            "author": AUTHOR_TO_ID.get(author),
            "text": text,
        })
    return pd.DataFrame(rows, columns=["filename", "author", "text"])


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    (kennedy/ and johnson/) and one for the test data (unlabeled/)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames (df_train, df_test)
    """
    path = Path(folder)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {path}")

    df_train = pd.concat(
        [make_df_from_dir(path / author) for author in AUTHOR_TO_ID],
        ignore_index=True,
    )
    df_test = make_df_from_dir(path / "unlabeled", label_from_filename=True)
    return df_train, df_test