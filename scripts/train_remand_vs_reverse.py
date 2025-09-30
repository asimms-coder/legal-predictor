#!/usr/bin/env python
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from joblib import dump

def add_meta_tokens(df: pd.DataFrame) -> pd.Series:
    court = "__COURT_" + df.court_name.str.lower().fillna("").str.replace(r"\W+","_", regex=True)
    year  = "__YEAR_"  + df.year.astype(str)
    return (df.text_clean.astype(str) + " " + court + " " + year).str.strip()

def stratified_splits(df):
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=52)
    tr, te = next(s1.split(df, df["y"]))
    train_all, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=53)
    tr2, va = next(s2.split(train_all, train_all["y"]))
    return train_all.iloc[tr2].reset_index(drop=True), train_all.iloc[va].reset_index(drop=True), test

def upsample_reverse(train, factor=1.5):
    if factor <= 1: return train
    rev = train[train.y==1]
    rem = train[train.y==0]
    if len(rev)==0: return train
    rev_up = resample(rev, replace=True, n_samples=int(len(rev)*factor), random_state=123)
    return pd.concat([rem, rev_up]).sample(frac=1, random_state=123).reset_index(drop=True)

def build_pipeline(C=1.0):
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(
        C=C, penalty="l2", solver="liblinear", max_iter=2000,
        class_weight="balanced"
    )
    return Pipeline([("vec", vec), ("clf", clf)])

def predict_with_thr(clf, X, thr_reverse=0.30):
    # class 1 = REVERSE
    proba = clf.predict_proba(X)[:,1]
    return (proba >= thr_reverse).astype(int), proba

def main(data_csv, upsample_factor, c_grid, thr_rev_grid, model_out):
    df = pd.read_csv(data_csv)
    df = df[df.label.isin(["REMAND","REVERSE"])].copy()
    df["text_plus"] = add_meta_tokens(df)
    df["y"] = (df.label == "REVERSE").astype(int)  # 1=REVERSE, 0=REMAND

    train, val, test = stratified_splits(df)
    train = upsample_reverse(train, factor=upsample_factor)
    print("Split sizes:", {k:len(v) for k,v in [("train",train),("val",val),("test",test)]})
    print("Class balance (train y):", dict(train.y.value_counts()))

    best = {"C": None, "thr_reverse": None, "macroF1": -1}
    best_clf = None

    for C in c_grid:
        pipe = build_pipeline(C=C)
        pipe.fit(train.text_plus.values, train.y.values)
        for thr in thr_rev_grid:
            pv,_ = predict_with_thr(pipe, val.text_plus.values, thr_reverse=thr)
            f1_macro = f1_score(val.y.values, pv, average="macro", zero_division=0)
            if f1_macro > best["macroF1"]:
                best = {"C": C, "thr_reverse": thr, "macroF1": f1_macro}
                best_clf = pipe

    print("Best VAL:", json.dumps(best, indent=2))

    pt, _ = predict_with_thr(best_clf, test.text_plus.values, thr_reverse=best["thr_reverse"])
    print("\nTest report (0=REMAND, 1=REVERSE):")
    print(classification_report(test.y.values, pt, digits=3, zero_division=0))
    print("Test macro-F1:", f1_score(test.y.values, pt, average="macro", zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(test.y.values, pt))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(best_clf, model_out)
    with open(model_out + ".thr.json","w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved Stage-B model → {model_out} and threshold json.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--upsample-reverse", type=float, default=1.5)
    ap.add_argument("--c-grid", nargs="+", type=float, default=[0.5,0.75,1.0,1.5])
    ap.add_argument("--thr-reverse-grid", nargs="+", type=float, default=[0.25,0.30,0.35,0.40,0.45])
    ap.add_argument("--model-out", default="models/stageB_remand_vs_reverse.joblib")
    a = ap.parse_args()
    main(a.data, a.upsample_reverse, a.c_grid, a.thr_reverse_grid, a.model_out)
