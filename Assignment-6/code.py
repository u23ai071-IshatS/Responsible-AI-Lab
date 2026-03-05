import os
import re
import random
import warnings
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_PATH      = "../Assignment-4/Data/train.csv"
TEST_PATH       = "../Assignment-4/Data/test.csv"
FASTTEXT_MODEL  = "cc.en.300.bin"   
SAMPLE_SIZE     = 10000              # rows to sample (RAM-friendly)
WORD2VEC_DIM    = 300
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT REDIRECTION
# ══════════════════════════════════════════════════════════════════════════════

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_data(train_path: str, test_path: str, sample: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load and lightly clean the Amazon review CSVs."""
    col_names = ["polarity", "title", "review"]

    train = pd.read_csv(train_path, header=None, names=col_names)
    test  = pd.read_csv(test_path,  header=None, names=col_names)
    df    = pd.concat([train, test], ignore_index=True)

    # Drop rows with missing review text
    df.dropna(subset=["review"], inplace=True)
    df["review"] = df["review"].astype(str)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"[DATA] Loaded {len(df):,} reviews.")
    return df


def clean_text(text: str) -> str:
    """Lowercase and strip special characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentence_embedding(text: str, model, method: str = "fasttext") -> np.ndarray:
    """
    Average word vectors to produce a sentence embedding.
    Works for both FastText and Word2Vec gensim models.
    """
    words  = clean_text(text).split()
    vecs   = []

    if method == "fasttext":
        # fasttext library: model.get_word_vector() handles OOV via subwords
        for w in words:
            vecs.append(model.get_word_vector(w))
    else:
        # gensim KeyedVectors: skip OOV words
        for w in words:
            if w in model:
                vecs.append(model[w])

    if vecs:
        return np.mean(vecs, axis=0)
    else:
        dim = model.get_dimension() if method == "fasttext" else model.vector_size
        return np.zeros(dim)


# ══════════════════════════════════════════════════════════════════════════════
# Q1 – FastText Semantic Search
# ══════════════════════════════════════════════════════════════════════════════

def q1_fasttext_semantic_search(df: pd.DataFrame, ft_model) -> None:
    print("\n" + "═"*70)
    print("Q1 – FASTTEXT SEMANTIC SEARCH")
    print("═"*70)

    reviews = df["review"].tolist()

    # Build review embedding matrix
    print("[Q1] Building review embeddings …")
    review_embeddings = np.array(
        [get_sentence_embedding(r, ft_model, "fasttext") for r in reviews]
    )  # shape: (N, 300)

    # User query
    query = "battery life is very poor"
    query_vec = get_sentence_embedding(query, ft_model, "fasttext").reshape(1, -1)
    print(f"[Q1] Query: \"{query}\"")

    # Cosine similarity
    sims = cosine_similarity(query_vec, review_embeddings)[0]  # (N,)
    top5_idx = np.argsort(sims)[::-1][:5]

    print("\n[Q1] Top-5 Semantically Similar Reviews:")
    print("-"*70)
    for rank, idx in enumerate(top5_idx, 1):
        print(f"Rank {rank}  |  Similarity: {sims[idx]:.4f}")
        print(f"Review   : {reviews[idx][:200]}")
        print("-"*70)


# ══════════════════════════════════════════════════════════════════════════════
# Q2 – Noise Injection & Word2Vec vs FastText Comparison
# ══════════════════════════════════════════════════════════════════════════════

def introduce_noise(text: str, noise_prob: float = 0.20) -> str:
    """
    Randomly corrupt words in a sentence:
      - swap two adjacent chars  (spelling mistake)
      - duplicate a char
      - insert a random char
      - append random suffix (noisy word)
    """
    words  = text.split()
    noisy  = []
    for w in words:
        if len(w) < 3 or random.random() > noise_prob:
            noisy.append(w)
            continue

        op = random.choice(["swap", "dup", "insert", "suffix"])
        if op == "swap" and len(w) >= 2:
            i = random.randint(0, len(w) - 2)
            w = w[:i] + w[i+1] + w[i] + w[i+2:]
        elif op == "dup":
            i = random.randint(0, len(w) - 1)
            w = w[:i] + w[i] + w[i:]
        elif op == "insert":
            i = random.randint(0, len(w))
            w = w[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + w[i:]
        else:  # suffix
            w = w + "".join(random.choices("zxqkj", k=random.randint(1, 3)))

        noisy.append(w)
    return " ".join(noisy)


def train_word2vec(sentences: list) -> "gensim.models.Word2Vec":
    from gensim.models import Word2Vec
    tokenized = [s.split() for s in sentences]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=WORD2VEC_DIM,
        window=5,
        min_count=2,
        workers=4,
        epochs=5,
        seed=RANDOM_SEED
    )
    return model


def embedding_norm(text: str, model, method: str) -> float:
    """Return L2-norm of the sentence embedding (proxy for embedding quality)."""
    vec = get_sentence_embedding(text, model, method)
    return float(np.linalg.norm(vec))


def q2_noise_comparison(df: pd.DataFrame, ft_model) -> None:
    print("\n" + "═"*70)
    print("Q2 – NOISE COMPARISON: Word2Vec vs FastText")
    print("═"*70)

    clean_reviews = [clean_text(r) for r in df["review"].tolist()]

    # Introduce noise
    noisy_reviews = [introduce_noise(r) for r in clean_reviews]

    # ── Train Word2Vec on clean corpus ──────────────────────────────────────
    print("[Q2] Training Word2Vec on clean reviews …")
    w2v_model = train_word2vec(clean_reviews)
    w2v_kv    = w2v_model.wv

    # ── Sample 10 reviews for qualitative comparison ────────────────────────
    sample_idx = random.sample(range(len(clean_reviews)), 10)

    ft_clean_norms,  ft_noisy_norms  = [], []
    w2v_clean_norms, w2v_noisy_norms = [], []
    w2v_oov_clean,   w2v_oov_noisy   = [], []

    for i in sample_idx:
        c, n = clean_reviews[i], noisy_reviews[i]

        ft_clean_norms.append(embedding_norm(c, ft_model, "fasttext"))
        ft_noisy_norms.append(embedding_norm(n, ft_model, "fasttext"))

        w2v_clean_norms.append(embedding_norm(c, w2v_kv, "word2vec"))
        w2v_noisy_norms.append(embedding_norm(n, w2v_kv, "word2vec"))

        # OOV rate for Word2Vec
        def oov_rate(text, kv):
            ws = text.split()
            return sum(1 for w in ws if w not in kv) / max(len(ws), 1)

        w2v_oov_clean.append(oov_rate(c, w2v_kv))
        w2v_oov_noisy.append(oov_rate(n, w2v_kv))

    print("\n[Q2] Results Table (10-review sample):")
    print(f"{'Metric':<40} {'Clean':>10} {'Noisy':>10}")
    print("-"*62)
    print(f"{'FastText  – Avg embedding norm':<40} {np.mean(ft_clean_norms):>10.4f} {np.mean(ft_noisy_norms):>10.4f}")
    print(f"{'Word2Vec  – Avg embedding norm':<40} {np.mean(w2v_clean_norms):>10.4f} {np.mean(w2v_noisy_norms):>10.4f}")
    print(f"{'Word2Vec  – Avg OOV rate':<40} {np.mean(w2v_oov_clean):>10.4f} {np.mean(w2v_oov_noisy):>10.4f}")

    print("\n[Q2] Qualitative Examples (first 3 samples):")
    print("-"*70)
    for rank, i in enumerate(sample_idx[:3], 1):
        c, n = clean_reviews[i], noisy_reviews[i]
        print(f"\n  Example {rank}")
        print(f"  Clean : {c[:120]}")
        print(f"  Noisy : {n[:120]}")

        # FastText: always produces non-zero vectors (subword)
        fc = get_sentence_embedding(c, ft_model, "fasttext")
        fn = get_sentence_embedding(n, ft_model, "fasttext")
        ft_sim = cosine_similarity(fc.reshape(1,-1), fn.reshape(1,-1))[0][0]

        # Word2Vec: may fall back to zero vector
        wc = get_sentence_embedding(c, w2v_kv, "word2vec")
        wn = get_sentence_embedding(n, w2v_kv, "word2vec")
        if np.any(wc) and np.any(wn):
            w2v_sim = cosine_similarity(wc.reshape(1,-1), wn.reshape(1,-1))[0][0]
        else:
            w2v_sim = 0.0

        print(f"  FastText  cosine(clean, noisy) = {ft_sim:.4f}")
        print(f"  Word2Vec  cosine(clean, noisy) = {w2v_sim:.4f}  "
              f"({'degraded – OOV words ignored' if w2v_sim < 0.8 else 'stable'})")

    print("\n[Q2] Key Insight:")
    print("  FastText uses character n-grams → handles typos gracefully.")
    print("  Word2Vec treats each token as atomic → OOV tokens → zero vectors → quality drops.")


# ══════════════════════════════════════════════════════════════════════════════
# Q3 – Vocabulary Coverage: Word2Vec vs FastText
# ══════════════════════════════════════════════════════════════════════════════

def q3_vocabulary_coverage(df: pd.DataFrame, ft_model, w2v_model=None) -> None:
    print("\n" + "═"*70)
    print("Q3 – VOCABULARY COVERAGE: Word2Vec vs FastText")
    print("═"*70)

    # Collect all unique words from the dataset
    all_words = set()
    for review in df["review"]:
        for word in clean_text(review).split():
            all_words.add(word)

    print(f"[Q3] Total unique words in dataset: {len(all_words):,}")

    # Train Word2Vec if not supplied
    if w2v_model is None:
        print("[Q3] Training Word2Vec …")
        clean_reviews = [clean_text(r) for r in df["review"].tolist()]
        w2v_model = train_word2vec(clean_reviews)

    w2v_kv = w2v_model.wv

    # ── Word2Vec coverage ───────────────────────────────────────────────────
    w2v_in_vocab  = [w for w in all_words if w     in w2v_kv]
    w2v_oov       = [w for w in all_words if w not in w2v_kv]
    w2v_coverage  = len(w2v_in_vocab) / len(all_words) * 100

    # ── FastText coverage ────────────────────────────────────────────────────
    # FastText always generates an embedding via subword n-grams;
    # we test whether the word appears as a full entry in the model vocab.
    ft_in_vocab   = []
    ft_oov_served = []   # OOV but FastText still gives embedding via subwords
    ft_vocab      = set(ft_model.get_words())

    for w in all_words:
        if w in ft_vocab:
            ft_in_vocab.append(w)
        else:
            ft_oov_served.append(w)   # subword embedding still possible

    ft_exact_coverage = len(ft_in_vocab) / len(all_words) * 100
    ft_total_coverage = 100.0  # FastText handles ALL words via subwords

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n[Q3] Coverage Summary:")
    print(f"  {'Metric':<45} {'Word2Vec':>12} {'FastText':>12}")
    print("  " + "-"*70)
    print(f"  {'Words in dataset vocabulary':<45} {len(all_words):>12,} {len(all_words):>12,}")
    print(f"  {'Words with exact vocab entry':<45} {len(w2v_in_vocab):>12,} {len(ft_in_vocab):>12,}")
    print(f"  {'OOV (no exact entry)':<45} {len(w2v_oov):>12,} {len(ft_oov_served):>12,}")
    print(f"  {'Exact vocab coverage (%)':<45} {w2v_coverage:>11.2f}% {ft_exact_coverage:>11.2f}%")
    print(f"  {'Effective coverage (%) – subword fallback':<45} {'N/A (0 for OOV)':>12} {'100.00%':>12}")

    # ── OOV embedding behaviour ──────────────────────────────────────────────
    print("\n[Q3] Embedding behaviour for OOV words (first 10 OOV examples):")
    print(f"  {'Word':<25} {'W2V embedding':<30} {'FT embedding norm':>20}")
    print("  " + "-"*78)

    demo_oov = w2v_oov[:10]
    for w in demo_oov:
        w2v_vec  = w2v_kv[w] if w in w2v_kv else None
        ft_vec   = ft_model.get_word_vector(w)
        ft_norm  = np.linalg.norm(ft_vec)

        w2v_str  = "ZERO VECTOR (OOV)" if w2v_vec is None else f"norm={np.linalg.norm(w2v_vec):.3f}"
        print(f"  {w:<25} {w2v_str:<30} {ft_norm:>18.4f}")

    # ── Subword demo ─────────────────────────────────────────────────────────
    print("\n[Q3] FastText Subword Demonstration:")
    deliberate_oov = ["producttt", "baterrylife", "amazingg", "shiping", "xyzabc123"]
    for w in deliberate_oov:
        ft_norm = np.linalg.norm(ft_model.get_word_vector(w))
        in_w2v  = w in w2v_kv
        print(f"  \"{w}\"  → FT norm={ft_norm:.4f}  |  W2V={'in vocab' if in_w2v else 'OOV → zero vector'}")

    print("\n[Q3] Key Insight:")
    print("  Word2Vec can only embed words seen during training (min_count threshold).")
    print("  FastText decomposes any word into character n-grams → always non-zero embedding.")
    print("  This makes FastText superior for noisy, domain-specific, or morphologically")
    print("  rich text (e.g., misspellings, rare product names, compound words).")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import fasttext

    # Redirect output to file
    output_filename = "output.txt"
    f = open(output_filename, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    try:
        # ── Load data ────────────────────────────────────────────────────────────
        df = load_data(TRAIN_PATH, TEST_PATH, sample=SAMPLE_SIZE)

        # ── Load pretrained FastText model ───────────────────────────────────────
        print(f"\n[INIT] Loading FastText model from: {FASTTEXT_MODEL} (this may take ~30s) …")
        ft_model = fasttext.load_model(FASTTEXT_MODEL)
        print(f"[INIT] FastText model loaded. Dimensions: {ft_model.get_dimension()}")

        # ── Q1 ───────────────────────────────────────────────────────────────────
        q1_fasttext_semantic_search(df, ft_model)

        # ── Q2 & Q3 share a Word2Vec model (train once) ─────────────────────────
        print("\n[INIT] Training Word2Vec for Q2 / Q3 …")
        clean_reviews = [clean_text(r) for r in df["review"].tolist()]
        w2v_model     = train_word2vec(clean_reviews)

        q2_noise_comparison(df, ft_model)
        q3_vocabulary_coverage(df, ft_model, w2v_model)

        print("\n" + "═"*70)
        print("ALL QUESTIONS COMPLETE")
        print("═"*70)

    finally:
        sys.stdout = original_stdout
        f.close()
        print(f"\n[INFO] All output has been saved to: {output_filename}")


if __name__ == "__main__":
    main()