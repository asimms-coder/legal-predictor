#!/usr/bin/env python
import argparse, json, numpy as np, pandas as pd
from joblib import load
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

def add_meta_tokens(df: pd.DataFrame):
    court = "__COURT_" + df.court_name.str.lower().fillna("").str.replace(r"\W+","_", regex=True)
    year  = "__YEAR_"  + df.year.astype(str)
    return (df.text_clean.astype(str) + " " + court + " " + year).str.strip()

def temporal_split_or_sss(df: pd.DataFrame):
    # Fallback: stratified split 80/20
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=99)
    tr, te = next(s1.split(df, df["label"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)

def main(data, stageA_path, stageB_path):
    df = pd.read_csv(data)
    df = df[["court_name","year","label","text_clean"]].dropna()
    df["text_plus"] = add_meta_tokens(df)

    train, test = temporal_split_or_sss(df)

    # Load Stage A (AFFIRM vs NON-AFFIRM) + threshold
    A = load(stageA_path)
    A_thr = json.load(open(stageA_path + ".thr.json"))["thr"]

    # Load Stage B (REMAND vs REVERSE among non-affirm) + threshold
    B = load(stageB_path)
    B_thr = json.load(open(stageB_path + ".thr.json"))["thr_reverse"]

    # Stage A
    p_nonaff = A.predict_proba(test.text_plus.values)[:,1]
    nonaff_mask = p_nonaff >= A_thr
    preds = np.array(["AFFIRM"] * len(test), dtype=object)

    # Stage B on the non-affirm subset
    if nonaff_mask.any():
        sub_idx = np.where(nonaff_mask)[0]
        subX = test.text_plus.values[sub_idx]
        p_rev = B.predict_proba(subX)[:,1]
        preds[sub_idx] = np.where(p_rev >= B_thr, "REVERSE", "REMAND")

    # Report
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
