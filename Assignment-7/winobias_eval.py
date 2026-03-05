"""
WinoBias Mask Prediction Evaluation
====================================
Evaluates gender bias in NLP models using the WinoBias dataset.

Metrics computed:
  - Accuracy            : % of correct pronoun predictions
  - Gender Accuracy Gap : accuracy(male) - accuracy(female)
  - Stereotype Pref.    : P(male pronoun | female-stereotyped role) — should be ~0.5 if unbiased

Models evaluated:
  (a) BERT-base-uncased  — Encoder-only / MLM model
  (b) GPT-2              — Decoder-only / LLM model

Dataset: https://huggingface.co/datasets/uclanlp/wino_bias

Usage:
    pip install transformers datasets torch
    python winobias_eval.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    pipeline,
)
import torch
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MALE_PRONOUNS   = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
ALL_PRONOUNS    = MALE_PRONOUNS | FEMALE_PRONOUNS

# Occupations labelled as stereotypically FEMALE per WinoBias paper
# (Zhao et al., 2018 — based on BLS occupational statistics)
FEMALE_STEREOTYPED_OCCUPATIONS = {
    "nurse", "receptionist", "librarian", "socialworker", "hairdresser",
    "secretary", "assistant", "housekeeper", "teacher", "cleaner",
    "cashier", "counselor", "attendant", "sewer",
}

BERT_MODEL = "bert-base-uncased"
GPT2_MODEL = "gpt2"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading & pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def load_wino_bias():
    """Load all four WinoBias splits and combine them."""
    splits = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    records = []
    for split in splits:
        try:
            ds = load_dataset("uclanlp/wino_bias", split=split)
        except Exception:
            # Some HuggingFace versions use different split names
            ds = load_dataset("uclanlp/wino_bias", name=split, split="validation")

        for item in ds:
            records.append({
                "sentence": item["tokens"],   # list of word tokens
                "coreference": item.get("coreference_clusters", []),
                "split": split,
            })
    print(f"Loaded {len(records)} examples across {len(splits)} splits.")
    return records


def find_pronoun_position(tokens):
    """
    Return (index, pronoun) of the first pronoun token in the sentence.
    Tokens is a list of strings.
    """
    for i, tok in enumerate(tokens):
        if tok.lower() in ALL_PRONOUNS:
            return i, tok.lower()
    return None, None


def build_masked_sentence(tokens, mask_idx, mask_token="[MASK]"):
    """Replace pronoun at mask_idx with mask_token and return as string."""
    masked = tokens[:mask_idx] + [mask_token] + tokens[mask_idx + 1:]
    return " ".join(masked)


def get_gender(pronoun):
    """Return 'male' or 'female' for a given pronoun string."""
    if pronoun in MALE_PRONOUNS:
        return "male"
    if pronoun in FEMALE_PRONOUNS:
        return "female"
    return "unknown"


def occupation_in_sentence(tokens):
    """Heuristic: return first occupation token found, else None."""
    for tok in tokens:
        clean = re.sub(r"[^a-z]", "", tok.lower())
        if clean in FEMALE_STEREOTYPED_OCCUPATIONS:
            return clean
    return None


def preprocess(records):
    """
    Build a flat list of evaluation examples.
    Each example: sentence str with [MASK], true_pronoun, gender, occupation, split.
    """
    examples = []
    for rec in records:
        tokens = rec["sentence"]
        idx, pronoun = find_pronoun_position(tokens)
        if idx is None:
            continue
        masked_sentence = build_masked_sentence(tokens, idx, "[MASK]")
        gender    = get_gender(pronoun)
        occ       = occupation_in_sentence(tokens)
        examples.append({
            "masked_sentence": masked_sentence,
            "true_pronoun":    pronoun,
            "gender":          gender,
            "occupation":      occ,
            "split":           rec["split"],
        })
    print(f"Preprocessed {len(examples)} valid examples (with pronoun).")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# BERT (Encoder-only) — Masked Language Model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bert(examples, model_name=BERT_MODEL):
    """
    Use BERT fill-mask pipeline to predict the pronoun at [MASK].
    Returns list of predicted pronouns (lowercased).
    """
    print(f"\n{'='*60}")
    print(f"Evaluating BERT: {model_name}")
    print(f"{'='*60}")

    fill_mask = pipeline(
        "fill-mask",
        model=model_name,
        top_k=20,          # retrieve top-20 so pronouns are captured
        device=0 if torch.cuda.is_available() else -1,
    )

    predictions = []
    for i, ex in enumerate(examples):
        if i % 200 == 0:
            print(f"  [{i}/{len(examples)}] processing...")

        sentence = ex["masked_sentence"]
        results  = fill_mask(sentence)

        # Pick the highest-scoring token that is a known pronoun
        best_pronoun = None
        best_score   = -1.0
        for r in results:
            token = r["token_str"].strip().lower()
            if token in ALL_PRONOUNS and r["score"] > best_score:
                best_pronoun = token
                best_score   = r["score"]

        # Fallback: pick the top-1 token regardless
        if best_pronoun is None:
            best_pronoun = results[0]["token_str"].strip().lower()

        predictions.append(best_pronoun)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 (Decoder-only) — Causal LM  (pseudo-likelihood scoring)
# ─────────────────────────────────────────────────────────────────────────────

def score_sentence_gpt2(model, tokenizer, sentence, device):
    """
    Compute the sum of log-probabilities for all tokens in `sentence`
    using teacher forcing (pseudo-likelihood / PLL).
    """
    enc = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        out    = model(**enc, labels=enc["input_ids"])
        # out.loss is mean negative log-likelihood per token
        nll    = out.loss.item()
        n_toks = enc["input_ids"].shape[1]
    return -nll * n_toks   # total log-likelihood (higher = more probable)


def evaluate_gpt2(examples, model_name=GPT2_MODEL):
    """
    For each example replace [MASK] with each candidate pronoun and score
    the resulting sentence under GPT-2. Pick the highest-scoring pronoun.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating GPT-2: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Candidate pronouns to compare (cover nominative / objective / possessive)
    CANDIDATES = {
        "male":   ["he", "him", "his"],
        "female": ["she", "her", "hers"],
    }
    ALL_CANDS = CANDIDATES["male"] + CANDIDATES["female"]

    predictions = []
    for i, ex in enumerate(examples):
        if i % 200 == 0:
            print(f"  [{i}/{len(examples)}] processing...")

        base = ex["masked_sentence"]    # contains [MASK]

        best_p     = None
        best_score = float("-inf")
        for cand in ALL_CANDS:
            filled    = base.replace("[MASK]", cand)
            score     = score_sentence_gpt2(model, tokenizer, filled, device)
            if score > best_score:
                best_score = score
                best_p     = cand

        predictions.append(best_p)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(examples, predictions, model_label):
    """
    Computes:
      - Overall Accuracy
      - Male Accuracy
      - Female Accuracy
      - Gender Accuracy Gap  = acc_male - acc_female
      - Stereotype Preference Score = P(male_pred | female-stereotyped occupation)
    """
    true_labels = [ex["true_pronoun"] for ex in examples]
    genders     = [ex["gender"]       for ex in examples]
    occs        = [ex["occupation"]   for ex in examples]

    n_total   = len(true_labels)
    correct   = [p == t for p, t in zip(predictions, true_labels)]

    # ── Overall accuracy ──────────────────────────────────────────────────
    accuracy = sum(correct) / n_total

    # ── Per-gender accuracy ───────────────────────────────────────────────
    male_correct = [c for c, g in zip(correct, genders) if g == "male"]
    fem_correct  = [c for c, g in zip(correct, genders) if g == "female"]

    acc_male   = np.mean(male_correct)   if male_correct else float("nan")
    acc_female = np.mean(fem_correct)    if fem_correct  else float("nan")
    gap        = acc_male - acc_female

    # ── Stereotype Preference Score ───────────────────────────────────────
    # = P(predicted pronoun is male | sentence has a female-stereotyped occupation)
    stereo_preds = [p for p, o in zip(predictions, occs) if o is not None]
    stereo_male  = [p for p in stereo_preds if p in MALE_PRONOUNS]
    sps          = len(stereo_male) / len(stereo_preds) if stereo_preds else float("nan")

    metrics = {
        "Model":                       model_label,
        "Accuracy":                    round(accuracy,  4),
        "Acc (Male targets)":          round(acc_male,  4),
        "Acc (Female targets)":        round(acc_female,4),
        "Gender Accuracy Gap (M-F)":   round(gap,       4),
        "Stereotype Preference Score": round(sps,       4),
        "N examples":                  n_total,
        "N stereo examples":           len(stereo_preds),
    }
    return metrics


def print_metrics(metrics):
    print(f"\n{'─'*55}")
    for k, v in metrics.items():
        print(f"  {k:<38} {v}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def per_split_accuracy(examples, predictions):
    """Return accuracy broken down by WinoBias split."""
    from collections import defaultdict
    split_correct = defaultdict(list)
    for ex, pred in zip(examples, predictions):
        split_correct[ex["split"]].append(pred == ex["true_pronoun"])
    return {s: round(np.mean(v), 4) for s, v in split_correct.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load & preprocess
    records  = load_wino_bias()
    examples = preprocess(records)

    all_metrics = []

    # ── (a) BERT ─────────────────────────────────────────────────────────
    bert_preds   = evaluate_bert(examples)
    bert_metrics = compute_metrics(examples, bert_preds, "BERT-base-uncased")
    print_metrics(bert_metrics)
    all_metrics.append(bert_metrics)

    print("\n  Per-split accuracy (BERT):")
    for split, acc in per_split_accuracy(examples, bert_preds).items():
        print(f"    {split:<25} {acc}")

    # ── (b) GPT-2 ────────────────────────────────────────────────────────
    gpt2_preds   = evaluate_gpt2(examples)
    gpt2_metrics = compute_metrics(examples, gpt2_preds, "GPT-2")
    print_metrics(gpt2_metrics)
    all_metrics.append(gpt2_metrics)

    print("\n  Per-split accuracy (GPT-2):")
    for split, acc in per_split_accuracy(examples, gpt2_preds).items():
        print(f"    {split:<25} {acc}")

    # ── Summary table ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics).set_index("Model")
    print("\n\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(df.to_string())

    df.to_csv("winobias_results.csv")
    print("\nResults saved to winobias_results.csv")


if __name__ == "__main__":
    main()