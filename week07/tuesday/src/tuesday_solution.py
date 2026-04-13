from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None


EXPECTED_REVIEW_COLUMNS = {
    "review_id",
    "customer_id",
    "product_id",
    "category",
    "review_text",
    "rating",
    "sentiment_label",
}


@dataclass
class W2VBundle:
    model_w2: Word2Vec
    model_w10: Word2Vec
    tokenized_reviews: List[List[str]]
    clean_reviews: List[str]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_reviews(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not read reviews CSV: {exc}") from exc
    missing = EXPECTED_REVIEW_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required review columns: {sorted(missing)}")
    return df


def load_customers(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not read customers CSV: {exc}") from exc


def train_word2vec_models(reviews: Sequence[str], vector_size: int = 100) -> W2VBundle:
    tokenized = [tokenize(t) for t in reviews]
    tokenized = [t for t in tokenized if t]
    clean_reviews = [" ".join(t) for t in tokenized]

    model_w2 = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=2,
        min_count=1,
        workers=1,
        seed=42,
        sg=1,
    )
    model_w10 = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=10,
        min_count=1,
        workers=1,
        seed=42,
        sg=1,
    )
    return W2VBundle(model_w2=model_w2, model_w10=model_w10, tokenized_reviews=tokenized, clean_reviews=clean_reviews)


def word_cosine(model: Word2Vec, a: str, b: str) -> float:
    if a not in model.wv.key_to_index or b not in model.wv.key_to_index:
        return float("nan")
    return float(cosine_similarity(model.wv[a].reshape(1, -1), model.wv[b].reshape(1, -1))[0, 0])


def sentence_vector_w2v(model: Word2Vec, sentence: str) -> np.ndarray:
    toks = tokenize(sentence)
    vecs = [model.wv[t] for t in toks if t in model.wv.key_to_index]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(np.array(vecs), axis=0)


def classify_cheap_meaning(model: Word2Vec, sentence: str) -> Dict[str, float | str]:
    affordable_anchors = ["affordable", "budget", "value", "worth"]
    low_quality_anchors = ["flimsy", "fragile", "poor", "cheaply"]

    context_words = [t for t in tokenize(sentence) if t != "cheap" and t in model.wv.key_to_index]
    if not context_words:
        return {"label": "unknown", "affordable_score": 0.0, "low_quality_score": 0.0}

    context_vec = np.mean(np.array([model.wv[t] for t in context_words]), axis=0)

    def anchor_mean(words: Sequence[str]) -> np.ndarray:
        present = [model.wv[w] for w in words if w in model.wv.key_to_index]
        if not present:
            return np.zeros(model.vector_size)
        return np.mean(np.array(present), axis=0)

    aff_vec = anchor_mean(affordable_anchors)
    low_vec = anchor_mean(low_quality_anchors)

    aff_score = float(cosine_similarity(context_vec.reshape(1, -1), aff_vec.reshape(1, -1))[0, 0])
    low_score = float(cosine_similarity(context_vec.reshape(1, -1), low_vec.reshape(1, -1))[0, 0])
    label = "affordable" if aff_score >= low_score else "low-quality"
    return {"label": label, "affordable_score": aff_score, "low_quality_score": low_score}


def compare_windows(bundle: W2VBundle) -> Dict[str, float | str]:
    semantic_pairs = [("battery", "charger"), ("camera", "photos"), ("quality", "durable")]
    syntactic_pairs = [("buy", "bought"), ("deliver", "delivery"), ("use", "using")]

    def avg_sim(model: Word2Vec, pairs: Sequence[Tuple[str, str]]) -> float:
        sims: List[float] = []
        for a, b in pairs:
            s = word_cosine(model, a, b)
            if not np.isnan(s):
                sims.append(s)
        return float(np.mean(sims)) if sims else 0.0

    sem_w2 = avg_sim(bundle.model_w2, semantic_pairs)
    sem_w10 = avg_sim(bundle.model_w10, semantic_pairs)
    syn_w2 = avg_sim(bundle.model_w2, syntactic_pairs)
    syn_w10 = avg_sim(bundle.model_w10, syntactic_pairs)

    note = (
        "Larger window generally captures broader semantic context, while smaller window tends to preserve local syntactic patterns."
    )
    return {
        "semantic_avg_window2": sem_w2,
        "semantic_avg_window10": sem_w10,
        "syntactic_avg_window2": syn_w2,
        "syntactic_avg_window10": syn_w10,
        "interpretation": note,
    }


def q2_similarity_all_models(model: Word2Vec) -> Dict[str, float]:
    review_a = "incredible camera but terrible battery life"
    review_b = "Battery drains fast, although photos are stunning"

    bow_vec = CountVectorizer()
    bow_m = bow_vec.fit_transform([review_a, review_b]).toarray()
    bow_cos = float(cosine_similarity(bow_m[0:1], bow_m[1:2])[0, 0])

    tfidf_vec = TfidfVectorizer()
    tfidf_m = tfidf_vec.fit_transform([review_a, review_b]).toarray()
    tfidf_cos = float(cosine_similarity(tfidf_m[0:1], tfidf_m[1:2])[0, 0])

    a_w2v = sentence_vector_w2v(model, review_a).reshape(1, -1)
    b_w2v = sentence_vector_w2v(model, review_b).reshape(1, -1)
    w2v_cos = float(cosine_similarity(a_w2v, b_w2v)[0, 0])

    sbert_cos = float("nan")
    if SentenceTransformer is not None:
        try:
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            embs = encoder.encode([review_a, review_b])
            sbert_cos = float(cosine_similarity(embs[0:1], embs[1:2])[0, 0])
        except Exception:  # noqa: BLE001
            sbert_cos = float("nan")

    return {
        "bow_cosine": bow_cos,
        "tfidf_cosine": tfidf_cos,
        "word2vec_avg_cosine": w2v_cos,
        "sentence_bert_cosine": sbert_cos,
    }


def save_results(output_dir: Path, payload: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tuesday_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Tuesday Assignment Results",
        "",
        "## Q1(a) Polysemy",
        f"- cosine(cheap, affordable): **{payload['q1a']['cheap_affordable_cosine']}**",
        f"- cosine(cheap, flimsy): **{payload['q1a']['cheap_flimsy_cosine']}**",
        "",
        "## Q1(b) Disambiguation",
        f"- Sentence: `{payload['q1b']['sample_sentence']}`",
        f"- Predicted meaning: **{payload['q1b']['prediction']['label']}**",
        "",
        "## Q1(c) Window Comparison",
        f"- semantic avg (w=2): {payload['q1c']['semantic_avg_window2']}",
        f"- semantic avg (w=10): {payload['q1c']['semantic_avg_window10']}",
        f"- syntactic avg (w=2): {payload['q1c']['syntactic_avg_window2']}",
        f"- syntactic avg (w=10): {payload['q1c']['syntactic_avg_window10']}",
        f"- Interpretation: {payload['q1c']['interpretation']}",
        "",
        "## Q2 Similarity",
        f"- BOW cosine: {payload['q2']['bow_cosine']}",
        f"- TF-IDF cosine: {payload['q2']['tfidf_cosine']}",
        f"- Word2Vec average cosine: {payload['q2']['word2vec_avg_cosine']}",
        f"- Sentence-BERT cosine: {payload['q2']['sentence_bert_cosine']}",
        "",
        "### Q2(a) Which captures similarity best?",
        "- Sentence-BERT is expected to best capture semantic similarity because it encodes sentence-level meaning beyond exact token overlap.",
        "",
        "### Q2(b) Exact overlap for BOW failure",
        "- Overlap is weak because one review uses `camera/photos` and `battery life/drains fast`, so lexical mismatch lowers count-based similarity.",
        "",
        "### Q2(c) Semantic gap progression",
        "- BOW uses exact counts only; TF-IDF reweights informative words; Word2Vec introduces shared semantic space; Sentence-BERT best captures contextual sentence semantics.",
    ]
    (output_dir / "tuesday_results.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(reviews_path: Path, customers_path: Path, output_dir: Path) -> None:
    reviews_df = load_reviews(reviews_path)
    _customers_df = load_customers(customers_path)
    review_texts = reviews_df["review_text"].fillna("").astype(str).tolist()

    bundle = train_word2vec_models(review_texts, vector_size=100)

    q1a = {
        "cheap_affordable_cosine": word_cosine(bundle.model_w10, "cheap", "affordable"),
        "cheap_flimsy_cosine": word_cosine(bundle.model_w10, "cheap", "flimsy"),
    }

    sample_sentence = "This cheap phone is great value and works reliably."
    q1b_pred = classify_cheap_meaning(bundle.model_w10, sample_sentence)

    q1c = compare_windows(bundle)
    q2 = q2_similarity_all_models(bundle.model_w10)

    payload = {
        "q1a": q1a,
        "q1b": {"sample_sentence": sample_sentence, "prediction": q1b_pred},
        "q1c": q1c,
        "q2": q2,
    }
    save_results(output_dir, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week07 Tuesday assignment solution")
    parser.add_argument("--reviews_path", type=Path, required=True)
    parser.add_argument("--customers_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.reviews_path, args.customers_path, args.output_dir)
