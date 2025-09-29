#!/usr/bin/env python
import argparse, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from joblib import dump

def temporal_split(df, train_end=2015, val_end=2018):
    train = df[df.year <= train_end]
    val   = df[(df.year > train_end) & (df.year <= val_end)]
    test  = df[df.year > val_end]
    return train, val, test

def main(csv_path, model_out, max_features):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text_clean","label","year"])
    df = df[df["label"].isin(["AFFIRM","REVERSE","REMAND"])]

    train, val, test = temporal_split(df)
    print("Split sizes:", {k: len(v) for k,v in zip(["train","val","test"], [train,val,test])})

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_features=max_features,
            min_df=3,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1
        ))
    ])

    pipe.fit(train.text_clean.values, train.label.values)

    print("\nValidation:")
    yv = val.label.values
    pv = pipe.predict(val.text_clean.values)
    print(classification_report(yv, pv, digits=3))
    print("Val macro-F1:", f1_score(yv, pv, average="macro"))

    print("\nTest:")
    yt = test.label.values
    pt = pipe.predict(test.text_clean.values)
    print(classification_report(yt, pt, digits=3))
    print("Test macro-F1:", f1_score(yt, pt, average="macro"))

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
