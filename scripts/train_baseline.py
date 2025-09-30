#!/usr/bin/env python
import argparse, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump

def temporal_split(df, train_end=2015, val_end=2018):
    train = df[df.year <= train_end]
    val   = df[(df.year > train_end) & (df.year <= val_end)]
    test  = df[df.year > val_end]
    return train, val, test

def stratified_fallback(df):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    y = df["label"]
    train_idx, test_idx = next(sss1.split(df, y))
    train_all, test = df.iloc[train_idx], df.iloc[test_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    y_tr = train_all["label"]
    tr_idx, va_idx = next(sss2.split(train_all, y_tr))
    train, val = train_all.iloc[tr_idx], train_all.iloc[va_idx]
    return train, val, test

def main(csv_path, model_out, max_features):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text_clean","label","year"])
    df = df[df["label"].isin(["AFFIRM","REVERSE","REMAND"])]

    train, val, test = temporal_split(df)
    use_fallback = (len(val) == 0 or len(test) == 0)
    if use_fallback:
        print("Temporal split had empty val/test; using stratified fallback.")
        train, val, test = stratified_fallback(df)

    print("Split sizes:", {"train": len(train), "val": len(val), "test": len(test)})

    # For tiny datasets, simpler features & solver are more stable
    tiny = len(train) < 200
    tfidf = TfidfVectorizer(
        ngram_range=(1,1) if tiny else (1,2),
        max_features=50000 if tiny else max_features,
        min_df=1 if tiny else 3,
        strip_accents="unicode",
        lowercase=True,
    )
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear" if tiny else "saga",
        C=2.0 if tiny else 1.0,
        n_jobs=None if tiny else -1
    )

    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    pipe.fit(train.text_clean.values, train.label.values)

    print("\nValidation:")
    pv = pipe.predict(val.text_clean.values)
    print(classification_report(val.label.values, pv, digits=3, zero_division=0))
    print("Val macro-F1:", f1_score(val.label.values, pv, average="macro", zero_division=0))

    print("\nTest:")
    pt = pipe.predict(test.text_clean.values)
    print(classification_report(test.label.values, pt, digits=3, zero_division=0))
    print("Test macro-F1:", f1_score(test.label.values, pt, average="macro", zero_division=0))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(pipe, model_out)
    print("Saved model →", model_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-out", default="models/baseline_tfidf_lr.joblib")
    ap.add_argument("--max-features", type=int, default=200000)
    a = ap.parse_args()
    main(a.data, a.model_out, a.max_features)
