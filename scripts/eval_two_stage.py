#!/usr/bin/env python
import argparse, json, pandas as pd, numpy as np
from joblib import load
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

def add_meta_tokens(df: pd.DataFrame):
    court = "__COURT_" + df.court_name.str.lower().fillna("").str.replace(r"\W+","_", regex=True)
    year  = "__YEAR_"  + df.year.astype(str)
    return (df.text_clean.astype(str) + " " + court + " " + year).str.strip()

def split(df):
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=99)
    tr, te = next(s1.split(df, df["label"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

def main(data, stageA_path, stageB_path):
    df = pd.read_csv(data)
    df = df[["court_name","year","label","text_clean"]].dropna()
    df["text_plus"] = add_meta_tokens(df)
    train, test = split(df)

    A = load(stageA_path)
    A_thr = json.load(open(stageA_path + ".thr.json"))["thr"]
    B = load(stageB_path)
    B_thr = json.load(open(stageB_path + ".thr.json"))["thr_reverse"]

    p_nonaff = A.predict_proba(test.text_plus.values)[:,1] >= A_thr
    preds = np.array(["AFFIRM"] * len(test), dtype=object)
    idx = np.where(p_nonaff)[0]
    if len(idx):
        subX = test.text_plus.values[idx]
        p_rev = B.predict_proba(subX)[:,1] >= B_thr
        preds[idx] = np.where(p_rev, "REVERSE", "REMAND")

    print("End-to-end test:")
    print(classification_report(test.label.values, preds, digits=3, zero_division=0))
    print("Macro-F1:", f1_score(test.label.values, preds, average="macro", zero_division=0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stageA", default="models/stageA_aff_vs_nonaff.joblib")
    ap.add_argument("--stageB", default="models/stageB_remand_vs_reverse.joblib")
    a = ap.parse_args()
    main(a.data, a.stageA, a.stageB)
