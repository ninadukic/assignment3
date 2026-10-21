# Assignment 3: Discussion

## Problem 1

### (A) Comparing Trevor's Naive Bayes with mine

Both implementations make the same 10 predictions on the unlabeled speeches (8/10 correct), so the differences are in correctness of the probabilities and code quality rather than in the final output.

**Differences and what can be done better**

1. **Smoothing denominator.** In the denominator of the likelihood formula, Trevor's code adds `alpha * (len(vocabulary) + 1)`, while mine adds `alpha * len(vocabulary)`. As I understand it, the extra `+ 1` leaves room for a word that never appeared in training. But the likelihood matrix has no column for that word, and the test() method skips unknown words anyway. Because of this, each class's probabilities add up to a little less than 1. On the speeches the difference is tiny (0.99999906 and 0.99999959), but on the small Jurafsky & Martin example it is easy to see: the rows add up to 0.971 and 0.967, and P(predictable | −) comes out as 2/35 instead of the textbook's 2/34. So Trevor's version would fail two of my three tests. One fix is to remove the `+ 1`, which is what my code does. The other is to add an extra column for unknown words.
2. **Word order in the vocabulary.** Trevor's code builds the vocabulary from a `set` without sorting it. Python doesn't keep sets of words in the same order from one run to the next, so the same word can end up in a different column each time the script runs. This doesn't change the predictions, but it makes results harder to compare and debug. I sorted the words first, so each word always gets the same column.
3. **Building the DataFrame.** Trevor's code adds each file to the DataFrame one at a time with `pd.concat`, which copies the whole table every time. I collected all the rows in a list first and made the DataFrame once at the end, which is simpler. To get the author's name from the file name, he uses `str(f).split('_')[-1][:-4]`, where `[:-4]` cuts off the last four characters to remove `.txt`. I used `f.stem.split('_')[-1]` instead, because `.stem` already gives the file name without the extension, which makes it easier to read.

**What I could have done differently**

1. In my Assignment 2 code, my last two `print` lines were outside the `if __name__ == "__main__":` block. Because of that, importing my file from another file crashed with a `NameError`. In this assignment, where files import each other, that would have broken everything. Now that code sits inside the `if __name__ == "__main__":` block in `main.py`, so it only runs when I run the file directly.
2. I didn't save the file names when I loaded the unlabeled speeches. So in Part F I could only say how many speeches were predicted for each author, not which author each file got. Now `build_dataframe` keeps a `filename` column, and `main.py` prints every file next to its prediction (see Part D).
3. I split each training speech into words twice: once to build the vocabulary and once to count the words. I also recalculated `np.log(likelihoods)` for every test speech, even though the result was the same every time. Now each of these happens only once.
4. I read the files in whatever order my computer listed them, which can be different on another computer. Now they're sorted, so the output comes out in the same order everywhere.

### (B) The NaiveBayes class

I turned my Assignment 2 functions into a class called `NaiveBayes`, in `nb/nb.py`. To use it, I create a model with `NaiveBayes(alpha=0.1)`, where `alpha` is the smoothing value. The class has two methods:

- `train(df)` learns from the training speeches. It builds the vocabulary and works out the priors and likelihoods, and the model keeps them so I don't have to pass them around separately like in Assignment 2.
- `test(df)` predicts an author for each speech and returns the predictions as a list of 0s (Kennedy) and 1s (Johnson). Words that never appeared in training are skipped.

I also moved the code that splits text into words into its own small class, `WhitespaceTokenizer`, in `nb/utils/tokenizer.py`, and the code that reads the speech files into `nb/utils/load_data.py`. That way each file has one job.

### (C) Unit tests

The tests use the five practice movie reviews from Jurafsky & Martin, Appendix B.3, which are saved in `tests/data/docs.txt`. They train my class on these reviews with add-one smoothing (`alpha=1`) and check that the results match the numbers worked out in the textbook:

1. `test_nb_class_priors` checks the priors: 3 of the 5 reviews are negative and 2 are positive, so P(−) = 3/5 and P(+) = 2/5.
2. `test_nb_class_likelihoods` checks that there are 20 different words, that the likelihood table has 2 rows and 20 columns, and that six probabilities match the textbook: P(predictable | −) = 2/34, P(predictable | +) = 1/29, P(no | −) = 2/34, P(no | +) = 1/29, P(fun | −) = 1/34, P(fun | +) = 2/29.
3. `test_nb_class_likelihoods_sum_to_one` checks that each row of the likelihood table adds up to 1, since each row is a set of probabilities for one class.

Output of `pytest tests/`:

```
============================= test session starts ==============================
platform darwin -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/ninadjukic/Desktop/assignment3/naive_bayes
configfile: pytest.ini
plugins: anyio-4.14.2
collected 3 items

tests/test_naive_bayes.py ...                                            [100%]

============================== 3 passed in 2.14s ===============================
```

### (D) Running main.py

Command: `python -m nb.bin.main -f data`

```
Loaded 102 training speeches and 10 unlabeled speeches.

Prior P(kennedy) = 0.353
Prior P(johnson) = 0.647
Vocabulary size: 24390
Likelihood matrix shape: (2, 24390)

File                    Predicted   Actual    Correct
speech_10_johnson.txt   johnson     johnson   yes
speech_11_kennedy.txt   kennedy     kennedy   yes
speech_12_kennedy.txt   kennedy     kennedy   yes
speech_13_kennedy.txt   kennedy     kennedy   yes
speech_14_kennedy.txt   kennedy     kennedy   yes
speech_5_johnson.txt    johnson     johnson   yes
speech_6_johnson.txt    kennedy     johnson   no
speech_7_johnson.txt    johnson     johnson   yes
speech_8_johnson.txt    kennedy     johnson   no
speech_9_kennedy.txt    kennedy     kennedy   yes

Accuracy: 80.00% (8/10 correct), F1: 75.00%
```

These results match Assignment 2 exactly, which confirms the refactor did not change the classifier's behavior. The two errors are `speech_6_johnson.txt` and `speech_8_johnson.txt`, both Johnson speeches predicted as Kennedy.


## Problem 2

I ran `index.py` with each of the three tokenizers, with stemming on:

```
python index.py -f 20newsgroups-initial.xml -t whitespace -s
python index.py -f 20newsgroups-initial.xml -t nltk -s
python index.py -f 20newsgroups-initial.xml -t ngram -s
```

All three indexes returned the same documents for every query. I also ran the whitespace index without `-s`, and the documents were the same.

| Index | "system" | "compatibility" | "system" AND "compatibility" |
| --- | --- | --- | --- |
| Whitespace | 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638 | 8772, 14398 | 8772, 14398 |
| NLTK | 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638 | 8772, 14398 | 8772, 14398 |
| 4-grams | 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638 | 8772, 14398 | 8772, 14398 |

Each query goes through the same steps as the posts (split, lowercase, remove punctuation, and stem), so the search ignores capital letters. For "system compatibility", the program only keeps the posts that contain both words.

### (B) Tokenizer interface

The assignment asks what universal properties a tokenizer should have. In my code, the base class `Tokenizer` defines two things every tokenizer shares:

1. It takes one piece of text and returns a list of tokens (the `tokenize` method). The base class doesn't split anything itself. Each tokenizer writes its own version, and calling it on the base class raises an error.
2. Its output is cleaned the same way: lowercased, with punctuation stripped from the start and end of each token, and empty tokens removed (the `normalize` method). Because this step is shared, the three indexes only differ in how the text was split.

The three tokenizers (`WhitespaceTokenizer`, `NLTKTokenizer`, which uses NLTK's Treebank tokenizer, and `NgramTokenizer` with n = 4) are subclasses of `Tokenizer`. Each one only writes its own `tokenize` method and inherits the rest.

### (E) Query "system"

Ten of the 20 posts contain "system", and all three indexes found the same ten.

- Lowercasing matters. Post 3 only writes "System" with a capital S ("System Board"), so without lowercasing it would have been missed.
- Stemming didn't add any posts, because every post that says "systems" also says "system" somewhere. It did change the counts: with stemming, post 14398 has "system" 3 times instead of 1, because "Systems" now counts too. It also made the index smaller, with 1,686 different words instead of 1,905.
- The whitespace and NLTK tokenizers agree here because removing punctuation already turns "system," and "system." into "system". They would differ on something like "system's", which NLTK splits into "system" and "'s", but that doesn't appear in these posts.
- With 4-grams, "system" becomes "syst", "yste" and "stem", and a post needs all three. This matches any word containing those letters, so it would also find words like "ecosystem", but none appear in these posts. The 4-gram index is also much bigger: 10,794 entries instead of 1,686.

### (F) Query "compatibility"

All three indexes return posts 8772 and 14398. Post 8772 writes "Compatibility" with a capital C, so lowercasing matters here too. With stemming, the query becomes "compat", which would also match "compatible", but that word doesn't appear in these posts, so the result is the same.

### (G) Query "system" AND "compatibility"

All three indexes return posts 8772 and 14398, the posts that appear in both (E) and (F). But they are not equally good matches. In 8772 the two words are right next to each other ("System Compatibility with old TSENG 4000"), and the post is actually about compatibility. In 14398, "compatibility" is part of a question about DOS 6.0, but "system" only shows up in the poster's email signature. The index only stores which posts contain a word and how many times, not where in the post it appears, so it can't tell these two cases apart.

### Testing the second file

I ran the same program on `20newsgroups-additional.xml` without changing anything, and it worked. "system" returned 7 posts (615, 3598, 6851, 13029, 13591, 14378, 16841), "compatibility" returned 3598 and 5811, and both together returned only 3598 ("backward system compatibility for motif").