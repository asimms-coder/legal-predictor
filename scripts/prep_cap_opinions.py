#!/usr/bin/env python
import argparse, json, os, re, csv
from datetime import datetime
from dateutil.parser import isoparse
import re

HEADINGS = {"CONCLUSION","DISPOSITION","JUDGMENT","DECREE","ORDER","ORDERS","DECISION"}
DISP_PATTERNS = [re.compile(rx, re.I) for rx in [
    r"\b(we|this court|the court)\s+(therefore|accordingly|now)?\s*(affirm|reverse|vacate|remand)s?\b",
    r"\b(judgment|order|conviction)\s+(is|be)\s+(hereby\s+)?(affirmed|reversed|vacated|remanded)\b",
    r"\bso ordered\b",
]]

_COMBO_RULES = [
    (r"\baffirm\w+\s+in\s+part.*revers\w+\s+in\s+part.*remand\w+\b", "REMAND"),
    (r"\baffirm\w+\s+in\s+part.*vacat\w+.*remand\w+\b", "REMAND"),

    (r"\bvacat\w+[\s,;]+(?:and\s+)?remand\w+\b", "REMAND"),
    (r"\brevers\w+[\s,;]+(?:and\s+)?remand\w+\b", "REMAND"),

    (r"\baffirm\w+\s+in\s+part.*revers\w+\s+in\s+part\b", "REVERSE"),
    (r"\baffirm\w+\s+in\s+part.*vacat\w+(?:\s+in\s+part)?\b", "REVERSE"),
]



_SINGLE_RULES = [
    (r"\bremand(?:ed|s|ing)?\b", "REMAND"),
    (r"\bset\s+aside\b", "REVERSE"),                 
    (r"\boverturn(?:ed|s|ing)?\b", "REVERSE"),       
    (r"\brevers(?:e|ed|es|ing)?\b", "REVERSE"),
    (r"\bvacat(?:ed|es|ing)?\b", "REVERSE"),
    (r"\baffirm(?:ed|s|ing)?\b", "AFFIRM"),
]


def normalize_label(s: str) -> str | None:
    if not s:
        return None
    s = s.strip().lower()

    for rx, lab in _COMBO_RULES:
        if re.search(rx, s):
            return lab

    for rx, lab in _SINGLE_RULES:
        if re.search(rx, s):
            return lab

    return None


def split_sents(t):  # simple sentence splitter
    return re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])", (t or "").strip())

def derive_label_from_text(raw_text: str):
    sents = split_sents(raw_text)
    tail = " ".join(sents[-12:]).lower()
    if re.search(r"\b(remand|remanded|remanding)\b", tail): return "REMAND"
    if re.search(r"\b(vacate|vacated|vacating)\b", tail): return "REVERSE"
    if re.search(r"\b(reverse|reversed|reversing)\b", tail): return "REVERSE"
    if re.search(r"\b(affirm|affirmed|affirming)\b", tail): return "AFFIRM"
    return None

def majority_text(case):
    ops = case.get("casebody", {}).get("opinions", []) or []
    if not ops: return None
    maj = [o for o in ops if (o.get("type") or "").lower() == "majority"]
    return (maj[0] if maj else ops[0]).get("text")

def remove_headers(t):
    lines = [ln for ln in (t or "").splitlines()
             if not re.match(r"^\s*(appeal|on appeal|before|argued|decided|filed|docket|case no\.)", ln.strip(), re.I)]
    return "\n".join(lines)

def truncate_at_headings(t):
    lines = (t or "").splitlines()
    for i, ln in enumerate(lines):
        tok = re.sub(r"[^A-Z ]", "", ln.upper()).strip()
        if tok in HEADINGS:
            return "\n".join(lines[:i])
    return t

def drop_disposition_sents(sents):
    out = []
    for s in sents:
        if any(p.search(s) for p in DISP_PATTERNS): continue
        out.append(s)
    return out

def tail_trim(sents, k=3):
    if len(sents) > 10: return sents[:-k]
    return sents

def clean_text(t):
    t = remove_headers(t)
    t = truncate_at_headings(t)
    s = split_sents(t)
    s = drop_disposition_sents(s)
    s = tail_trim(s, 5)
    return " ".join(s).strip()

def looks_federal_by_citation(case: dict) -> bool:
    cits = case.get("citations") or []
    for c in cits:
        cite = (c.get("cite") or "").lower()
        if " u.s. " in f" {cite} ": return True
        if re.search(r"\bf\.\s?\d*d?\b", cite, re.I): return True   # F., F.2d, F.3d
        if re.search(r"\bf\.\s?4th\b", cite, re.I): return True      # F.4th
    return False

def is_federal_appellate_or_scotus(case: dict) -> bool:
    court = (case.get("court", {}) or {}).get("name_abbreviation") or case.get("court", {}).get("name") or ""
    juris = (case.get("jurisdiction", {}) or {}).get("name") or ""
    s = f"{court} {juris}".lower()
    if ("supreme court of the united states" in s or "u.s. supreme court" in s or
        "united states court of appeals" in s or "u.s. court of appeals" in s or
        "federal circuit" in s or
        any(tag in s for tag in ["1st cir","2d cir","2nd cir","3d cir","3rd cir","4th cir","5th cir",
                                 "6th cir","7th cir","8th cir","9th cir","10th cir","11th cir","d.c. cir"])):
        return True
    return looks_federal_by_citation(case)

def extract_label(case):
    for raw in [case.get("decision"), case.get("disposition"), case.get("casebody", {}).get("disposition")]:
        lab = normalize_label(raw)
        if lab: return lab
    hm = case.get("casebody", {}).get("head_matter") or ""
    m = re.search(r"DISPOSITION:\s*(.+)", hm, re.I)
    lab = normalize_label(m.group(1).strip()) if m else None
    if lab: return lab
    raw_text = majority_text(case) or ""
    if raw_text: return derive_label_from_text(raw_text)
    return None

def year_of(case):
    dt = case.get("decision_date") or case.get("date") or case.get("date_decision")
    if not dt: return None
    try: return isoparse(dt).year
    except Exception:
        try: return datetime.strptime(dt[:10], "%Y-%m-%d").year
        except Exception: return None

def iter_cases(path):
    for root, _, files in os.walk(path):
        for fn in files:
            p = os.path.join(root, fn)
            if fn.endswith(".jsonl"):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try: yield json.loads(line)
                        except Exception: continue
            elif fn.endswith(".json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict) and "casebody" in obj:
                        yield obj
                    elif isinstance(obj, dict) and "cases" in obj:
                        for rec in obj["cases"]:
                            if isinstance(rec, dict) and "casebody" in rec:
                                yield rec
                except Exception:
                    continue

def main(in_dir, out_csv, start_year, end_year):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["case_id","court_name","year","label","text_clean"])
        for case in iter_cases(in_dir):
            try:
                if not is_federal_appellate_or_scotus(case): 
                    continue
                y = extract_label(case)
                if y not in {"AFFIRM","REVERSE","REMAND"}:
                    continue
                yr = year_of(case)
                if not yr or yr < start_year or yr > end_year:
                    continue
                txt = majority_text(case)
                if not txt or len(txt) < 500:
                    continue
                clean = clean_text(txt)
                if len(clean) < 300:
                    continue
                cid = case.get("id") or (case.get("citations") or [{}])[0].get("cite") or ""
                court = (case.get("court", {}) or {}).get("name_abbreviation") or (case.get("court", {}) or {}).get("name") or ""
                w.writerow([cid, court, yr, y, clean])
                written += 1
            except Exception:
                continue
    print(f"Wrote {written} rows to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    ap.add_argument("--start-year", type=int, default=1990)
    ap.add_argument("--end-year", type=int, default=2023)
    a = ap.parse_args()
    main(a.in_dir, a.out_csv, a.start_year, a.end_year)
