#!/usr/bin/env python
import argparse, os, json
import numpy as np, pandas as pd
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
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(s1.split(df, df.label))
    train_all, test = df.iloc[tr], df.iloc[te]
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    tr2, va = next(s2.split(train_all, train_all.label))
    return train_all.iloc[tr2].reset_index(drop=True), train_all.iloc[va].reset_index(drop=True), test.reset_index(drop=True)

def upsample_reverse(train, factor=2.0):
    if factor <= 1: return train
    rev = train[train.label=="REVERSE"]; maj = train[train.label!="REVERSE"]
    if len(rev)==0: return train
    rev_up = resample(rev, replace=True, n_samples=int(len(rev)*factor), random_state=123)
    return pd.concat([maj, rev_up]).sample(frac=1, random_state=123).reset_index(drop=True)

def build_pipeline(C=0.5, penalty="l2"):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2,
                           max_features=400_000, strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=2,
                           max_features=250_000, strip_accents="unicode", lowercase=True)
    feats = FeatureUnion([("w", word), ("c", char)])
    base = LogisticRegression(solver="liblinear", class_weight="balanced",
                              penalty=penalty, C=C, max_iter=3000)
    return Pipeline([("feats", feats), ("clf", OneVsRestClassifier(base))])

def predict_with_thresholds(pipe, X, thr):
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba(X)
    out = []
    for row in proba:
        scores = row.copy()
        for j, lab in enumerate(classes):
            if scores[j] < float(thr.get(lab, 0.0)):
                scores[j] = -1e9
        j = int(np.argmax(scores))
        if scores[j] < 0: j = int(np.argmax(row))
        out.append(classes[j])
    return np.array(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--upsample-reverse", type=float, default=3.0)
    ap.add_argument("--c-grid", type=float, nargs="*", default=[0.4, 0.5, 0.75])
    ap.add_argument("--penalty", choices=["l1","l2"], default="l2")
    ap.add_argument("--thr-affirm", type=float, nargs="*", default=[0.55, 0.60])
    ap.add_argument("--thr-remand", type=float, nargs="*", default=[0.45, 0.50])
    ap.add_argument("--thr-reverse", type=float, nargs="*", default=[0.18, 0.22, 0.25, 0.28, 0.32, 0.35])
    ap.add_argument("--model-out", default="models/ovr_logreg_hybrid.joblib")
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
        for ta in a.thr_affirm:
            for trm in a.thr_remand:
                for trv in a.thr_reverse:
                    thr = {"AFFIRM": ta, "REMAND": trm, "REVERSE": trv}
                    pred = predict_with_thresholds(pipe, val.text_clean.values, thr)
                    f = f1_score(val.label.values, pred, average="macro", zero_division=0)
                    if best is None or f > best["f1"]:
                        best = {"C": C, "thr": thr, "f1": f, "pipe": pipe}
    print("Best on VAL:", json.dumps({"C": best["C"], "thresholds": best["thr"], "macroF1": best["f1"]}, indent=2))

    pred_te = predict_with_thresholds(best["pipe"], test.text_clean.values, best["thr"])
    print("\nTest report:\n", classification_report(test.label.values, pred_te, digits=3, zero_division=0))
    print("Test macro-F1:", f1_score(test.label.values, pred_te, average="macro", zero_division=0))

    os.makedirs("models", exist_ok=True)
    dump(best["pipe"], a.model_out)
    with open(a.model_out + ".thresholds.json", "w") as f:
        json.dump({"C": best["C"], "thresholds": best["thr"]}, f, indent=2)
