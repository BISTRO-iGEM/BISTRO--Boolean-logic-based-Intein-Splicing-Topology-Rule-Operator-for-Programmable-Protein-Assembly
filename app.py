"""
BISTRO — Step 3: Flask API Backend
====================================
Serves the trained ML model via REST API.
The frontend (static/index.html) calls these endpoints.

Endpoints:
  GET  /api/inteins                   → list of all 20 inteins
  POST /api/predict                   → predict compatibility for a pair
  POST /api/hunt_c                    → find all C-term partners for given N-term name
  POST /api/hunt_n                    → find all N-term partners for given C-term name
  POST /api/search_custom_sequence    → find best match for custom sequence
  GET  /api/model_info                → model metadata

Run: python 3_app.py
Then open: static/index.html in your browser
"""

import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # allow frontend to call API from file://

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/bistro_model.pkl"
LE_PATH    = "models/bistro_label_encoder.pkl"
META_PATH  = "models/bistro_model_meta.json"

model, le, meta = None, None, {}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(LE_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"[BISTRO] Model loaded — {meta['feature_mode']} features, CV F1={meta['cv_f1_mean']:.3f}")
else:
    print("[BISTRO] WARNING: No trained model found. Run 2_train_model.py first.")
    print("[BISTRO] API will run in FALLBACK mode (Boolean rules only).")


# ─────────────────────────────────────────────────────────────────────────────
# INTEIN DATABASE (same as 1_prepare_data.py — keep in sync)
# ─────────────────────────────────────────────────────────────────────────────

INTEINS = [
    {"name":"Ssp DnaE",          "org":"Synechocystis sp. PCC6803",       "group":"DnaE","selfOnly":False,"n_acc":"BBa_25GXRNKA","c_acc":"BBa_25DAKQQ8",  "n_seq":"CLAEGTRIFDPVTGSSRYMARVYGQDKPDYVLGHRMHESTMVLENVLGLGEIPKDLNMQKLNKVLVSRSKPISGQEVMDNVKSDLIQFGDNHVSKLPELPEEFQEHIERLYQELRQKLAKQYLHLPKIKSEPGTDVQKLISEEDLGDPDYREQLATLIREQMKQEKM","c_seq":"CFNGTAVGAKLNPQQMGQIERFNYGPTCQQYAAQALQDYLKQHQEKIQKQLAQRLADLQKQVRDLQHEVNARLATDHDLNQQMKQFNEQLKKQLTPQMQHRLQQELQKQIQDLQAQYNQL"},
    {"name":"Ter DnaE-3",        "org":"Trichodesmium erythraeum",        "group":"DnaE","selfOnly":False,"n_acc":"BBa_25D2AKSB","c_acc":"BBa_25KKVAEW",  "n_seq":"","c_seq":""},
    {"name":"Tvu DnaE",          "org":"Thermosynechococcus vulcanus",    "group":"DnaE","selfOnly":False,"n_acc":"BBa_25D9G3XX","c_acc":"BBa_25A9PDOX",  "n_seq":"","c_seq":""},
    {"name":"Sel-PCC6301 DnaE",  "org":"Synechococcus elongatus PCC6301","group":"DnaE","selfOnly":False,"n_acc":"BBa_25S5TNW3","c_acc":"BBa_25IB6RAW",  "n_seq":"","c_seq":""},
    {"name":"Sel-PC7942 DnaE",   "org":"Synechococcus elongatus PC7942", "group":"DnaE","selfOnly":False,"n_acc":"BBa_25EI8V66","c_acc":"BBa_25AHFHPM",  "n_seq":"","c_seq":""},
    {"name":"Oli DnaE",          "org":"Oscillatoria limnetica",         "group":"DnaE","selfOnly":False,"n_acc":"BBa_25QL9R7D","c_acc":"BBa_25F6FPLB",  "n_seq":"","c_seq":""},
    {"name":"Nsp-PCC7120 DnaE",  "org":"Nostoc sp. PCC7120",             "group":"DnaE","selfOnly":False,"n_acc":"BBa_25OVRRY7","c_acc":"BBa_25WMHS8N",  "n_seq":"","c_seq":""},
    {"name":"Npu-PCC73102 DnaE", "org":"Nostoc punctiforme PCC73102",    "group":"DnaE","selfOnly":False,"n_acc":"BBa_25E6N642","c_acc":"BBa_25GW1TDZ",  "n_seq":"","c_seq":""},
    {"name":"Maer-NIES843 DnaE", "org":"Microcystis aeruginosa NIES843", "group":"DnaE","selfOnly":False,"n_acc":"BBa_25KLG34K","c_acc":"BBa_25F8C5ZQ",  "n_seq":"","c_seq":""},
    {"name":"Cwa DnaE",          "org":"Crocosphaera watsonii",          "group":"DnaE","selfOnly":False,"n_acc":"BBa_25P1R6TZ","c_acc":"BBa_25WI3QZ0",  "n_seq":"","c_seq":""},
    {"name":"Csp-PCC8801 DnaE",  "org":"Cyanothece sp. PCC8801",         "group":"DnaE","selfOnly":False,"n_acc":"BBa_25IGRU14","c_acc":"BBa_255BD5ZB",  "n_seq":"","c_seq":""},
    {"name":"Csp-PCC7424 DnaE",  "org":"Cyanothece sp. PCC7424",         "group":"DnaE","selfOnly":False,"n_acc":"BBa_25RPMZQA","c_acc":"BBa_25HBUDUF",  "n_seq":"","c_seq":""},
    {"name":"Csp-CCY0110 DnaE",  "org":"Cyanothece sp. CCY0110",         "group":"DnaE","selfOnly":False,"n_acc":"BBa_25G95DLK","c_acc":"BBa_25KRZ21R",  "n_seq":"","c_seq":""},
    {"name":"Cra-CS505 DnaE",    "org":"Cylindrospermopsis raciborskii", "group":"DnaE","selfOnly":False,"n_acc":"BBa_25Y30FDV","c_acc":"BBa_25JWTAOY",  "n_seq":"","c_seq":""},
    {"name":"Ava DnaE",          "org":"Anabaena variabilis",            "group":"DnaE","selfOnly":False,"n_acc":"BBa_25FCPS4L","c_acc":"BBa_25ZF1QF5",  "n_seq":"","c_seq":""},
    {"name":"Aha DnaE",          "org":"Aphanothece halophytica",        "group":"DnaE","selfOnly":False,"n_acc":"BBa_25CS4JLS","c_acc":"BBa_25Z1ZFPY",  "n_seq":"","c_seq":""},
    {"name":"Aov DnaE",          "org":"Aphanizomenon ovalisporum",      "group":"DnaE","selfOnly":False,"n_acc":"BBa_25TEW6CH","c_acc":"BBa_25IKNR5T",  "n_seq":"","c_seq":""},
    {"name":"Mtu-T17 RecA",      "org":"Mycobacterium tuberculosis",     "group":"RecA","selfOnly":True, "n_acc":"BBa_25VRTGF8","c_acc":"BBa_25YYMD7Z",  "n_seq":"","c_seq":""},
    {"name":"Dra Snf2",          "org":"Deinococcus radiodurans",        "group":"Snf2","selfOnly":True, "n_acc":"BBa_25IK6KEW","c_acc":"BBa_25ZJCBN6",  "n_seq":"","c_seq":""},
    {"name":"Neq Pol",           "org":"Nanoarchaeum equitans",          "group":"Pol", "selfOnly":True, "n_acc":"BBa_258Y84U0","c_acc":"BBa_257FZ9V0",  "n_seq":"","c_seq":""},
    {"name":"Sce VMA",           "org":"Saccharomyces cerevisiae",       "group":"VMA", "selfOnly":True, "n_acc":"BBa_25M8R987","c_acc":"BBa_253EQDGS",  "n_seq":"","c_seq":""},
]

INTEIN_BY_NAME = {it["name"]: it for it in INTEINS}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (mirrors 2_train_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def physicochemical_features(seq):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned  = "".join(aa if aa in valid_aa else "A" for aa in seq.upper())
    if len(cleaned) < 5:
        return [0.0] * 13
    try:
        a = ProteinAnalysis(cleaned)
        mw = a.molecular_weight(); pi = a.isoelectric_point()
        instab = a.instability_index(); gravy = a.gravy()
        helix, turn, sheet = a.secondary_structure_fraction()
        comp = a.get_amino_acids_percent()
        return [mw, pi, instab, gravy, helix, turn, sheet,
                comp.get("C",0), comp.get("H",0), comp.get("N",0),
                comp.get("S",0), comp.get("T",0), float(len(cleaned))]
    except Exception:
        return [0.0] * 13


def amino_acid_composition_features(seq):
    """Extract all 20 amino acid composition features."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(aa if aa in valid_aa else "A" for aa in seq.upper())
    if len(cleaned) < 5:
        return [0.0] * 20
    try:
        a = ProteinAnalysis(cleaned)
        aa_comp = a.get_amino_acids_percent()
        amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                       "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        return [aa_comp.get(aa, 0.0) for aa in amino_acids]
    except Exception:
        return [0.0] * 20

def motif_features(seq):
    seq = seq.upper()
    groups = [["HNF","HNY","HNL","HNC"],["NC","SC","TC"],
              ["CFK","SFK","TFK","CFN"],["SH","CH","TH"],["TXXH","SXXH"]]
    feats = []
    for g in groups:
        found = 0
        for m in g:
            if "X" in m:
                for i in range(len(seq)-len(m)+1):
                    w = seq[i:i+len(m)]
                    if all(a=="X" or a==b for a,b in zip(m,w)):
                        found=1; break
            elif m in seq:
                found=1; break
        feats.append(float(found))
    return feats

def build_feature_vector(n_seq, c_seq):
    nf = physicochemical_features(n_seq); cf = physicochemical_features(c_seq)
    naa = amino_acid_composition_features(n_seq); caa = amino_acid_composition_features(c_seq)
    nm = motif_features(n_seq); cm = motif_features(c_seq)
    nl = float(len(n_seq)); cl = float(len(c_seq))
    cross = [nl/(cl+1e-6), abs(nl-cl)]
    return np.array(nf+cf+naa+caa+nm+cm+cross, dtype=np.float32).reshape(1,-1)


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE SIMILARITY & MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def sequence_similarity(seq1, seq2):
    """
    Calculate sequence similarity using a simple alignment score.
    Returns a score between 0 and 1 (higher = more similar).
    """
    if not seq1 or not seq2:
        return 0.0
    
    # Normalize lengths
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if max_len == 0:
        return 0.0
    
    # Count matching characters
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    
    # Calculate similarity: matches weighted by length similarity
    length_sim = min_len / max_len
    seq_sim = matches / max_len
    
    return (seq_sim + length_sim) / 2.0


def find_best_match(input_seq, terminal_type="n"):
    """
    Find the best matching intein in database for a given sequence.
    terminal_type: 'n' for N-terminal, 'c' for C-terminal
    Returns: (best_intein_dict, similarity_score)
    """
    best_intein = None
    best_score = -1
    
    for intein in INTEINS:
        if terminal_type == "n":
            db_seq = intein.get("n_seq", "")
        else:
            db_seq = intein.get("c_seq", "")
        
        if not db_seq or "PLACEHOLDER" in db_seq:
            continue
        
        score = sequence_similarity(input_seq.upper(), db_seq.upper())
        if score > best_score:
            best_score = score
            best_intein = intein
    
    return best_intein, best_score


# ─────────────────────────────────────────────────────────────────────────────
# BOOLEAN FALLBACK (when no model is trained yet)
# ─────────────────────────────────────────────────────────────────────────────

def boolean_predict(n_intein, c_intein):
    if n_intein["name"] == c_intein["name"]:
        return "Circularise", 1.0
    if n_intein["selfOnly"] or c_intein["selfOnly"]:
        return "Not compatible", 1.0
    if n_intein["group"] == c_intein["group"]:
        return "Linearise", 1.0
    return "Not compatible", 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict_pair(n_name, c_name):
    n_int = INTEIN_BY_NAME.get(n_name)
    c_int = INTEIN_BY_NAME.get(c_name)
    if not n_int or not c_int:
        return None, None, "Intein name not found"

    if model is not None and n_int["n_seq"] and c_int["c_seq"]:
        # ML prediction
        feats  = build_feature_vector(n_int["n_seq"], c_int["c_seq"])
        pred   = model.predict(feats)[0]
        proba  = model.predict_proba(feats)[0]
        label  = le.inverse_transform([pred])[0]
        conf   = float(proba.max())
        source = "ML"
    else:
        # Boolean fallback
        label, conf = boolean_predict(n_int, c_int)
        source = "Boolean"

    return label, conf, source


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/inteins", methods=["GET"])
def get_inteins():
    """Return list of all inteins (without sequences for brevity)."""
    data = [{k:v for k,v in it.items() if k not in ("n_seq","c_seq")} for it in INTEINS]
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body: { "n_name": "Ssp DnaE", "c_name": "Ter DnaE-3" }
    Returns: { "prediction": "Linearise", "confidence": 0.87, "source": "ML" }
    """
    body   = request.get_json()
    n_name = body.get("n_name", "")
    c_name = body.get("c_name", "")

    label, conf, source = predict_pair(n_name, c_name)
    if label is None:
        return jsonify({"error": source}), 400

    return jsonify({
        "n_name":     n_name,
        "c_name":     c_name,
        "prediction": label,
        "confidence": round(conf, 3),
        "source":     source,
    })


@app.route("/api/hunt_c", methods=["POST"])
def hunt_c():
    """
    Given an N-terminal intein, return all C-terminal partners with predictions.
    Body: { "n_name": "Ssp DnaE" }
    """
    body   = request.get_json()
    n_name = body.get("n_name", "")
    results = []
    for c_int in INTEINS:
        if c_int["name"] == n_name:
            continue
        label, conf, source = predict_pair(n_name, c_int["name"])
        if label:
            results.append({
                "c_name":     c_int["name"],
                "c_org":      c_int["org"],
                "c_acc":      c_int["c_acc"],
                "prediction": label,
                "confidence": round(conf, 3),
                "source":     source,
            })
    # sort by confidence desc
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(results)


@app.route("/api/hunt_n", methods=["POST"])
def hunt_n():
    """
    Given a C-terminal intein, return all N-terminal partners with predictions.
    Body: { "c_name": "Ssp DnaE" }
    """
    body   = request.get_json()
    c_name = body.get("c_name", "")
    results = []
    for n_int in INTEINS:
        if n_int["name"] == c_name:
            continue
        label, conf, source = predict_pair(n_int["name"], c_name)
        if label:
            results.append({
                "n_name":     n_int["name"],
                "n_org":      n_int["org"],
                "n_acc":      n_int["n_acc"],
                "prediction": label,
                "confidence": round(conf, 3),
                "source":     source,
            })
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(results)


@app.route("/api/search_custom_sequence", methods=["POST"])
def search_custom_sequence():
    """
    Find best matching intein for a custom sequence and predict compatible pairs.
    Body: { "sequence": "MLGPG...", "terminal_type": "n" }
    terminal_type: 'n' for N-terminal, 'c' for C-terminal
    """
    body = request.get_json()
    sequence = body.get("sequence", "").upper()
    terminal_type = body.get("terminal_type", "n")
    
    if not sequence or len(sequence) < 5:
        return jsonify({"error": "Sequence too short (minimum 5 amino acids)"}), 400
    
    # Validate amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_aa for aa in sequence):
        return jsonify({"error": "Sequence contains invalid amino acids"}), 400
    
    # Find best match in database
    best_intein, similarity = find_best_match(sequence, terminal_type)
    
    if not best_intein:
        return jsonify({"error": "No matching inteins found in database"}), 404
    
    # Predict compatibility with best match
    if terminal_type == "n":
        # Custom N-term, find C-term partners
        partners = []
        for c_int in INTEINS:
            if c_int["name"] == best_intein["name"]:
                continue
            # Use best match for prediction with the custom sequence
            label, conf, source = predict_pair(best_intein["name"], c_int["name"])
            if label:
                partners.append({
                    "c_name":     c_int["name"],
                    "c_org":      c_int["org"],
                    "c_acc":      c_int["c_acc"],
                    "prediction": label,
                    "confidence": round(conf, 3),
                    "source":     source,
                })
        partners.sort(key=lambda x: x["confidence"], reverse=True)
    else:
        # Custom C-term, find N-term partners
        partners = []
        for n_int in INTEINS:
            if n_int["name"] == best_intein["name"]:
                continue
            label, conf, source = predict_pair(n_int["name"], best_intein["name"])
            if label:
                partners.append({
                    "n_name":     n_int["name"],
                    "n_org":      n_int["org"],
                    "n_acc":      n_int["n_acc"],
                    "prediction": label,
                    "confidence": round(conf, 3),
                    "source":     source,
                })
        partners.sort(key=lambda x: x["confidence"], reverse=True)
    
    return jsonify({
        "custom_sequence": sequence,
        "best_match": {
            "name": best_intein["name"],
            "org": best_intein["org"],
            "similarity": round(similarity, 3)
        },
        "compatible_partners": partners,
        "message": f"Found {len(partners)} compatible partners for best match"
    })


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Return model metadata."""
    if not meta:
        return jsonify({"status": "No model trained yet", "mode": "Boolean fallback"})
    return jsonify({**meta, "status": "Model loaded"})


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("BISTRO API — http://0.0.0.0:5000")
    print("Open static/index.html in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)