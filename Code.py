"""
BISTRO — Step 1: Data Preparation
==================================
- Defines all 20 inteins (N and C halves) with sequences from INBASE
- Builds all 400 pairs (20x20)
- Labels each pair: Circularise / Linearise / Not compatible
- Saves training CSV to data/pairs.csv

HOW TO USE:
  Replace the placeholder sequences below with the real ones from:
  https://inteins.com or https://www.neb.com/tools-and-resources/selection-charts/split-inteins
  Then run: python 1_prepare_data.py
"""

import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# INTEIN DEFINITIONS
# Replace n_seq and c_seq with real INBASE sequences for each intein.
# selfOnly = True means it ONLY splices with its own partner (orthogonal).
# group = inteins in the same group can cross-splice (Linearise).
# ─────────────────────────────────────────────────────────────────────────────

INTEINS = [
    # ── DnaE cyanobacterial group (cross-splice = Linearise) ─────────────────
    {"name": "Ssp DnaE",           "org": "Synechocystis sp. PCC6803",         "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25GXRNKA", "c_acc": "BBa_25DAKQQ8",
     "n_seq": "CLAEGTRIFDPVTGSSRYMARVYGQDKPDYVLGHRMHESTMVLENVLGLGEIPKDLNMQKLNKVLVSRSKPISGQEVMDNVKSDLIQFGDNHVSKLPELPEEFQEHIERLYQELRQKLAKQYLHLPKIKSEPGTDVQKLISEEDLGDPDYREQLATLIREQMKQEKM",
     "c_seq": "CFNGTAVGAKLNPQQMGQIERFNYGPTCQQYAAQALQDYLKQHQEKIQKQLAQRLADLQKQVRDLQHEVNARLATDHDLNQQMKQFNEQLKKQLTPQMQHRLQQELQKQIQDLQAQYNQL"},

    {"name": "Ter DnaE-3",         "org": "Trichodesmium erythraeum",           "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25D2AKSB", "c_acc": "BBa_25KKVAEW",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Tvu DnaE",           "org": "Thermosynechococcus vulcanus",       "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25D9G3XX", "c_acc": "BBa_25A9PDOX",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Sel-PCC6301 DnaE",   "org": "Synechococcus elongatus PCC6301",   "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25S5TNW3", "c_acc": "BBa_25IB6RAW",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Sel-PC7942 DnaE",    "org": "Synechococcus elongatus PC7942",    "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25EI8V66", "c_acc": "BBa_25AHFHPM",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Oli DnaE",           "org": "Oscillatoria limnetica",             "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25QL9R7D", "c_acc": "BBa_25F6FPLB",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Nsp-PCC7120 DnaE",   "org": "Nostoc sp. PCC7120",                "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25OVRRY7", "c_acc": "BBa_25WMHS8N",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Npu-PCC73102 DnaE",  "org": "Nostoc punctiforme PCC73102",       "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25E6N642", "c_acc": "BBa_25GW1TDZ",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Maer-NIES843 DnaE",  "org": "Microcystis aeruginosa NIES843",    "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25KLG34K", "c_acc": "BBa_25F8C5ZQ",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Cwa DnaE",           "org": "Crocosphaera watsonii",              "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25P1R6TZ", "c_acc": "BBa_25WI3QZ0",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Csp-PCC8801 DnaE",   "org": "Cyanothece sp. PCC8801",            "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25IGRU14", "c_acc": "BBa_255BD5ZB",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Csp-PCC7424 DnaE",   "org": "Cyanothece sp. PCC7424",            "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25RPMZQA", "c_acc": "BBa_25HBUDUF",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Csp-CCY0110 DnaE",   "org": "Cyanothece sp. CCY0110",            "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25G95DLK", "c_acc": "BBa_25KRZ21R",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Cra-CS505 DnaE",     "org": "Cylindrospermopsis raciborskii",     "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25Y30FDV", "c_acc": "BBa_25JWTAOY",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Ava DnaE",           "org": "Anabaena variabilis",                "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25FCPS4L", "c_acc": "BBa_25ZF1QF5",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Aha DnaE",           "org": "Aphanothece halophytica",            "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25CS4JLS", "c_acc": "BBa_25Z1ZFPY",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Aov DnaE",           "org": "Aphanizomenon ovalisporum",          "group": "DnaE", "selfOnly": False,
     "n_acc": "BBa_25TEW6CH", "c_acc": "BBa_25IKNR5T",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    # ── Orthogonal (self-only) inteins ────────────────────────────────────────
    {"name": "Mtu-T17 RecA",       "org": "Mycobacterium tuberculosis",         "group": "RecA", "selfOnly": True,
     "n_acc": "BBa_25VRTGF8", "c_acc": "BBa_25YYMD7Z",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Dra Snf2",           "org": "Deinococcus radiodurans",            "group": "Snf2", "selfOnly": True,
     "n_acc": "BBa_25IK6KEW", "c_acc": "BBa_25ZJCBN6",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Neq Pol",            "org": "Nanoarchaeum equitans",              "group": "Pol",  "selfOnly": True,
     "n_acc": "BBa_258Y84U0", "c_acc": "BBa_257FZ9V0",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},

    {"name": "Sce VMA",            "org": "Saccharomyces cerevisiae",           "group": "VMA",  "selfOnly": True,
     "n_acc": "BBa_25M8R987", "c_acc": "BBa_253EQDGS",
     "n_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE",
     "c_seq": "PLACEHOLDER_REPLACE_WITH_INBASE_SEQUENCE"},
]


# ─────────────────────────────────────────────────────────────────────────────
# LABELING LOGIC (from your PDF Boolean model)
# ─────────────────────────────────────────────────────────────────────────────

def get_label(n_intein, c_intein):
    """
    Returns: 'Circularise', 'Linearise', or 'Not compatible'
    Rules derived from your iGEM PDF Boolean model:
    - Cognate pair (same intein) → Circularise
    - selfOnly intein paired with anything else → Not compatible
    - Same group (both DnaE) → Linearise
    - Different groups → Not compatible
    """
    if n_intein["name"] == c_intein["name"]:
        # cognate pair — always circularises
        return "Circularise"

    if n_intein["selfOnly"] or c_intein["selfOnly"]:
        # orthogonal inteins don't cross-splice
        return "Not compatible"

    if n_intein["group"] == c_intein["group"]:
        # same family, non-cognate → linear ligation
        return "Linearise"

    return "Not compatible"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PAIRS DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

rows = []
for n_int in INTEINS:
    for c_int in INTEINS:
        label = get_label(n_int, c_int)
        rows.append({
            "n_name":  n_int["name"],
            "c_name":  c_int["name"],
            "n_org":   n_int["org"],
            "c_org":   c_int["org"],
            "n_group": n_int["group"],
            "c_group": c_int["group"],
            "n_acc":   n_int["n_acc"],
            "c_acc":   c_int["c_acc"],
            "n_seq":   n_int["n_seq"],
            "c_seq":   c_int["c_seq"],
            "label":   label,
        })

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

df.to_csv("data/pairs.csv", index=False)

# Print summary
print("=" * 50)
print("BISTRO — Data Preparation Complete")
print("=" * 50)
print(f"Total pairs generated : {len(df)}")
print(f"\nLabel distribution:")
print(df["label"].value_counts().to_string())
print(f"\nSaved to: data/pairs.csv")
print("\nNOTE: Replace PLACEHOLDER sequences in this file")
print("with real sequences from https://inteins.com")