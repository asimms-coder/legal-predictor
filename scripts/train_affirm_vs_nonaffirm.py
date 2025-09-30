#!/usr/bin/env python
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
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
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    y = df["y"]
    tr, te = next(s1.split(df, y))
    train_all, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    tr2, va = next(s2.split(train_all, train_all["y"]))
    return train_all.iloc[tr2].reset_index(drop=True), train_all.iloc[va].reset_index(drop=True), test

def upsample_minority(train, minority_label=1, factor=1.5):
    if factor <= 1: return train
    minor = train[train.y==minority_label]
    major = train[train.y!=minority_label]
    if len(minor)==0: return train
    minor_up = resample(minor, replace=True, n_samples=int(len(minor)*factor), random_state=123)
    return pd.concat([major, minor_up]).sample(frac=1, random_state=123).reset_index(drop=True)

def build_pipeline(C=0.75):
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=3, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(
        C=C, penalty="l2", solver="liblinear", max_iter=2000,
        class_weight="balanced", n_jobs=None
    )
    return Pipeline([("vec", vec), ("clf", clf)])

def predict_with_thr(clf, X, thr=0.5):
    # class 1 = NON-AFFIRM
    proba = clf.predict_proba(X)[:,1]
    return (proba >= thr).astype(int), proba

def main(data_csv, upsample_factor, c_grid, thr_grid, model_out):
    df = pd.read_csv(data_csv)
    df = df[["court_name","year","label","text_clean"]].dropna()
    df["text_plus"] = add_meta_tokens(df)
    df["y"] = (df.label != "AFFIRM").astype(int)  # 1 = NON-AFFIRM

    train, val, test = stratified_splits(df)
    train = upsample_minority(train, minority_label=1, factor=upsample_factor)
    print("Split sizes:", {k:len(v) for k,v in [("train",train),("val",val),("test",test)]})
    print("Class balance (train y):", dict(train.y.value_counts()))

    best = {"C": None, "thr": None, "macroF1": -1}
    best_clf = None

    for C in c_grid:
        pipe = build_pipeline(C=C)
        pipe.fit(train.text_plus.values, train.y.values)
        # grid thresholds on VAL
        for thr in thr_grid:
            pv, _ = predict_with_thr(pipe, val.text_plus.values, thr)
            f1_macro = f1_score(val.y.values, pv, average="macro", zero_division=0)
            if f1_macro > best["macroF1"]:
                best = {"C": C, "thr": thr, "macroF1": f1_macro}
                best_clf = pipe

    print("Best VAL:", json.dumps(best, indent=2))

    # Test
    pt, proba = predict_with_thr(best_clf, test.text_plus.values, best["thr"])
    print("\nTest report (0=AFFIRM, 1=NON-AFFIRM):")
    print(classification_report(test.y.values, pt, digits=3, zero_division=0))
    print("Test macro-F1:", f1_score(test.y.values, pt, average="macro", zero_division=0))
    try:
        print("ROC-AUC:", roc_auc_score(test.y.values, proba))
    except Exception:
        pass
    print("Confusion matrix:\n", confusion_matrix(test.y.values, pt))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(best_clf, model_out)
    with open(model_out + ".thr.json","w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved Stage-A model → {model_out} and threshold json.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--upsample-nonaffirm", type=float, default=1.5)
    ap.add_argument("--c-grid", nargs="+", type=float, default=[0.5,0.75,1.0])
    ap.add_argument("--thr-grid", nargs="+", type=float, default=[0.50,0.55,0.60,0.65])
    ap.add_argument("--model-out", default="models/stageA_aff_vs_nonaff.joblib")
    a = ap.parse_args()
    main(a.data, a.upsample_nonaffirm, a.c_grid, a.thr_grid, a.model_out)
