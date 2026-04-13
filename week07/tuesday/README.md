# Week 07 - Tuesday Assignment

This folder contains the Tuesday deliverables for NLP Foundations (Word2Vec, polysemy disambiguation, and representation similarity comparison).

## Structure

- `src/tuesday_solution.py` - Main modular implementation.
- `tuesday_solution.ipynb` - Notebook runner.
- `outputs/` - Generated markdown and JSON results.
- `requirements.txt` - Required Python packages.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/tuesday_solution.py --reviews_path "C:\Users\siddp\Downloads\shopsense_reviews.csv" --customers_path "C:\Users\siddp\Downloads\shopsense_customers.csv" --output_dir outputs
```

## Outputs

- `outputs/tuesday_results.json`
- `outputs/tuesday_results.md`

These include:
- Q1(a) cosine comparisons for polysemous word `cheap`
- Q1(b) context-based disambiguation prediction
- Q1(c) window size comparison (`2` vs `10`)
- Q2 similarity scores across BOW, TF-IDF, Word2Vec averaging, and Sentence-BERT
