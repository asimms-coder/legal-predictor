#!/usr/bin/env python
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample
from joblib import dump

LABELS = ["AFFIRM","REMAND","REVERSE"]

def stratified_splits(df):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    y = df["label"]; tr_idx, te_idx = next(sss1.split(df, y))
    train_all, test = df.iloc[tr_idx], df.iloc[te_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    y_tr = train_all["label"]; tr2_idx, va_idx = next(sss2.split(train_all, y_tr))
    return (train_all.iloc[tr2_idx].reset_index(drop=True),
            train_all.iloc[va_idx].reset_index(drop=True),
            test.reset_index(drop=True))

def upsample_reverse(train, factor=2.0):
    if factor <= 1.0: return train
    maj = train[train.label != "REVERSE"]
    rev = train[train.label == "REVERSE"]
    if len(rev) == 0: return train
    rev_up = resample(rev, replace=True, n_samples=int(len(rev)*factor), random_state=123)
    return pd.concat([maj, rev_up]).sample(frac=1, random_state=123).reset_index(drop=True)

def build_pipeline(C=0.75, penalty="l2", max_features=300_000):
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=2, max_features=max_features,
        strip_accents="unicode", lowercase=True
    )
    base = LogisticRegression(
        solver="liblinear", class_weight="balanced",
        max_iter=3000, C=C, penalty=penalty
    )
    return Pipeline([("tfidf", vec), ("clf", OneVsRestClassifier(base))])

def predict_with_thresholds(pipe, X, thr):
    # preserve class order from the fitted classifier
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba(X)
    out = []
    for row in proba:
        # apply thresholds per class
        scores = row.copy()
        for j, lab in enumerate(classes):
            if scores[j] < float(thr.get(lab, 0.0)):
                scores[j] = -1e9
        j = int(np.argmax(scores))
        # if nothing passed (all negative), fall back to plain argmax
        if scores[j] < 0:
            j = int(np.argmax(row))
        out.append(classes[j])
    return np.array(out), proba

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--upsample-reverse", type=float, default=2.0)
    ap.add_argument("--c-grid", type=float, nargs="*", default=[0.5, 0.75, 1.0])
    ap.add_argument("--penalty", choices=["l1","l2"], default="l2")
    ap.add_argument("--thr-affirm", type=float, nargs="*", default=[0.50, 0.55, 0.60])
    ap.add_argument("--thr-remand", type=float, nargs="*", default=[0.40, 0.50])
    ap.add_argument("--thr-reverse", type=float, nargs="*", default=[0.20, 0.25, 0.30, 0.35, 0.40])
    ap.add_argument("--model-out", default="models/ovr_logreg_tuned.joblib")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df = df[df.label.isin(LABELS)].dropna(subset=["text_clean"])
    train, val, test = stratified_splits(df)
    train = upsample_reverse(train, a.upsample_reverse)
    print("Split sizes:", {"train": len(train), "val": len(val), "test": len(test)})

    best = None
    for C in a.c_grid:
        pipe = build_pipeline(C=C, penalty=a.penalty)
        pipe.fit(train.text_clean.values, train.label.values)

        # grid thresholds on VAL
        for ta in a.thr_affirm:
            for trm in a.thr_remand:
                for trv in a.thr_reverse:
                    thr = {"AFFIRM": ta, "REMAND": trm, "REVERSE": trv}
                    pred_val, _ = predict_with_thresholds(pipe, val.text_clean.values, thr)
                    f = f1_score(val.label.values, pred_val, average="macro", zero_division=0)
                    if best is None or f > best["f1"]:
                        best = {"C": C, "thr": thr, "f1": f, "pipe": pipe}

    print("Best on VAL:", json.dumps({"C": best["C"], "thresholds": best["thr"], "macroF1": best["f1"]}, indent=2))

    # Final test
    pred_test, _ = predict_with_thresholds(best["pipe"], test.text_clean.values, best["thr"])
    print("\nTest report:\n", classification_report(test.label.values, pred_test, digits=3, zero_division=0))
    print("Test macro-F1:", f1_score(test.label.values, pred_test, average="macro", zero_division=0))

    os.makedirs(os.path.dirname(a.model_out), exist_ok=True)
    dump(best["pipe"], a.model_out)
    with open(a.model_out + ".thresholds.json", "w") as f:
        json.dump({"C": best["C"], "thresholds": best["thr"]}, f, indent=2)
