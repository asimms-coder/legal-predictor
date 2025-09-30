#!/usr/bin/env python
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample
from joblib import dump

LABELS = ["AFFIRM","REMAND","REVERSE"]

def stratified_splits(df):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    y = df["label"]; tr, te = next(sss1.split(df, y))
    train_all, test = df.iloc[tr], df.iloc[te]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    y_tr = train_all["label"]; tr2, va = next(sss2.split(train_all, y_tr))
    return train_all.iloc[tr2].reset_index(drop=True), \
           train_all.iloc[va].reset_index(drop=True), \
           test.reset_index(drop=True)

def upsample_reverse(train, factor=2.0):
    if factor <= 1.0: return train
    maj = train[train.label != "REVERSE"]
    rev = train[train.label == "REVERSE"]
    if len(rev) == 0: return train
    rev_up = resample(rev, replace=True, n_samples=int(len(rev)*factor), random_state=123)
    return pd.concat([maj, rev_up]).sample(frac=1, random_state=123).reset_index(drop=True)

def build_pipeline(max_word=200_000, max_char=200_000):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1,2), min_df=3, max_features=max_word,
            strip_accents="unicode", lowercase=True)),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=max_char,
            strip_accents="unicode", lowercase=True)),
    ])
    base = LogisticRegression(
        solver="liblinear",           # binary-friendly
        class_weight="balanced",      # auto-balance each OvR binary task
        max_iter=2000
    )
    clf = OneVsRestClassifier(base)
    return Pipeline([("feats", feats), ("clf", clf)])

def biased_argmax(proba, labels, biases=None):
    # biases: dict like {"REVERSE": 1.2} to upweight REVERSE scores
    if not biases: return labels[np.argmax(proba, axis=1)]
    w = np.ones_like(proba)
    for j, lab in enumerate(LABELS):
        if biases.get(lab):
            w[:, j] *= biases[lab]
    return labels[np.argmax(proba * w, axis=1)]

def main(csv_path, model_out, up_factor, reverse_bias_grid):
    df = pd.read_csv(csv_path).dropna(subset=["text_clean","label","year"])
    df = df[df.label.isin(LABELS)]

    train, val, test = stratified_splits(df)
    train = upsample_reverse(train, factor=up_factor)

    print("Split sizes:", {"train":len(train), "val":len(val), "test":len(test)})

    pipe = build_pipeline()
    pipe.fit(train.text_clean.values, train.label.values)

    # Evaluate with a small bias search toward REVERSE
    biases = [{"REVERSE": b} for b in reverse_bias_grid] + [None]
    best = None
    for b in biases:
        pv = pipe.predict_proba(val.text_clean.values)
        yv = val.label.values
        pred_val = biased_argmax(pv, np.array(LABELS), b or {})
        f1 = f1_score(yv, pred_val, average="macro", zero_division=0)
        if (best is None) or (f1 > best["f1"]):
            best = {"bias": b or {}, "f1": f1}

    print("\nBest bias on VAL:", best)

    # Final eval on TEST with the chosen bias
    pt = pipe.predict_proba(test.text_clean.values)
    yt = test.label.values
    pred_test = biased_argmax(pt, np.array(LABELS), best["bias"])
    print("\nValidation report (best bias):")
    pv = pipe.predict_proba(val.text_clean.values)
    pred_val = biased_argmax(pv, np.array(LABELS), best["bias"])
    print(classification_report(val.label.values, pred_val, digits=3, zero_division=0))
    print("Val macro-F1:", f1_score(val.label.values, pred_val, average="macro", zero_division=0))

    print("\nTest report (best bias):")
    print(classification_report(yt, pred_test, digits=3, zero_division=0))
    print("Test macro-F1:", f1_score(yt, pred_test, average="macro", zero_division=0))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(pipe, model_out)
    # Save the bias you chose so you can reuse it at inference time
    with open(model_out + ".bias.json", "w") as f:
        json.dump(best["bias"], f)
    print("Saved model →", model_out, "and bias →", model_out + ".bias.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-out", default="models/ovr_logreg.joblib")
    ap.add_argument("--upsample-reverse", type=float, default=2.0)
    ap.add_argument("--reverse-bias-grid", type=float, nargs="*", default=[1.0,1.1,1.2,1.3,1.4])
    a = ap.parse_args()
    main(a.data, a.model_out, a.upsample_reverse, a.reverse_bias_grid)
