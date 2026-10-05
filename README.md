# Assignment 1 starter kit — Train a minimal LM from scratch

Read the handout first (`A1_minimal_lm.pdf` on Moodle). This kit contains the tests you
must pass, the adapter file that connects them to your code, and skeletons for the
data preparation, training and reporting steps.

```
lm/                     your code goes here (any structure you like)
tests/
  adapters.py           the only test file you edit: point each function at your code
  test_tokenizer.py     Part 1 (20 points)
  test_model.py         Part 2 modules and accounting (15 points)
  test_nn_utils.py      loss, clipping, generation (5 points)
  test_optimizer.py     AdamW, cosine schedule (5 points)
  test_data.py          batches, checkpoints (5 points)
  _reference.py         slow reference implementations the tests compare against; do not edit
  fixtures/corpus.txt   a tiny corpus with special tokens
scripts/prepare_data.py tokenise a text file into a uint16 memmap with your tokenizer
scripts/train.py        training-loop skeleton with the results.json plumbing done
notebooks/setup.ipynb   Colab: mount Drive, install, run tests, train
report/report.tex       the report template (three pages plus figures)
grading/grade.py        exactly what we run: points per component
grading/validate_results.py   checks results.json before you submit
```

## Getting started

```
uv sync                     # or: pip install -e .
uv run pytest -q            # 74 tests, all failing until you fill in tests/adapters.py
uv run pytest -q tests/test_tokenizer.py -x     # one part at a time, stop at the first failure
python grading/grade.py     # your current automatic score
```

The tests run on CPU in a few seconds. Signatures, shapes and conventions are documented
in `tests/adapters.py`; the handout is the specification. If a test and the handout seem
to disagree, ask on the forum: the tests win.

## Submitting

One archive per group containing this directory (without data and without checkpoints),
`results.json` (run `python grading/validate_results.py results.json` first), the report
PDF, the notebook you trained with, and a Drive link to the reference checkpoint.
