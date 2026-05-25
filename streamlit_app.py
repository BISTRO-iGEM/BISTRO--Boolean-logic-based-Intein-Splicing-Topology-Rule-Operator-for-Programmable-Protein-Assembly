"""
BISTRO — Streamlit Web App
==========================
Interactive UI for intein compatibility prediction using machine learning.

Run: streamlit run streamlit_app.py
Deploy to Streamlit Cloud: https://docs.streamlit.io/deploy/streamlit-cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BISTRO - Intein Compatibility Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem; }
    .metric-card { 
        background-color: #f0f2f6; 
        padding: 1rem; 
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-high { color: #28a745; font-weight: bold; }
    .prediction-medium { color: #ffc107; font-weight: bold; }
    .prediction-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_data():
    model, le, meta = None, None, {}
    
    MODEL_PATH = "models/bistro_model.pkl"
    LE_PATH    = "models/bistro_label_encoder.pkl"
    META_PATH  = "models/bistro_model_meta.json"
    
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
    
    return model, le, meta

# ─────────────────────────────────────────────────────────────────────────────
# INTEIN DATABASE
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
INTEIN_NAMES = [it["name"] for it in INTEINS]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def physicochemical_features(seq):
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(aa if aa in valid_aa else "A" for aa in seq.upper())
    if len(cleaned) < 5:
        return [0.0] * 13
    try:
        a = ProteinAnalysis(cleaned)
        mw = a.molecular_weight()
        pi = a.isoelectric_point()
        instab = a.instability_index()
        gravy = a.gravy()
        helix, turn, sheet = a.secondary_structure_fraction()
        comp = a.get_amino_acids_percent()
        return [mw, pi, instab, gravy, helix, turn, sheet,
                comp.get("C",0), comp.get("H",0), comp.get("N",0),
                comp.get("S",0), comp.get("T",0), float(len(cleaned))]
    except Exception:
        return [0.0] * 13

def amino_acid_composition_features(seq):
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
    nf = physicochemical_features(n_seq)
    cf = physicochemical_features(c_seq)
    naa = amino_acid_composition_features(n_seq)
    caa = amino_acid_composition_features(c_seq)
    nm = motif_features(n_seq)
    cm = motif_features(c_seq)
    nl = float(len(n_seq))
    cl = float(len(c_seq))
    cross = [nl/(cl+1e-6), abs(nl-cl)]
    return np.array(nf+cf+naa+caa+nm+cm+cross, dtype=np.float32).reshape(1,-1)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def boolean_predict(n_intein, c_intein):
    if n_intein["name"] == c_intein["name"]:
        return "Circularise", 1.0
    if n_intein["selfOnly"] or c_intein["selfOnly"]:
        return "Not compatible", 1.0
    if n_intein["group"] == c_intein["group"]:
        return "Linearise", 1.0
    return "Not compatible", 1.0

def predict_pair(n_name, c_name, model, le):
    n_int = INTEIN_BY_NAME.get(n_name)
    c_int = INTEIN_BY_NAME.get(c_name)
    if not n_int or not c_int:
        return None, None, "Intein name not found"

    if model is not None and n_int["n_seq"] and c_int["c_seq"]:
        # ML prediction
        feats = build_feature_vector(n_int["n_seq"], c_int["c_seq"])
        pred = model.predict(feats)[0]
        proba = model.predict_proba(feats)[0]
        label = le.inverse_transform([pred])[0]
        conf = float(proba.max())
        source = "ML"
    else:
        # Boolean fallback
        label, conf = boolean_predict(n_int, c_int)
        source = "Boolean"

    return label, conf, source

def sequence_similarity(seq1, seq2):
    if not seq1 or not seq2:
        return 0.0
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    length_sim = min_len / max_len
    seq_sim = matches / max_len
    return (seq_sim + length_sim) / 2.0

def find_best_match(input_seq, terminal_type="n"):
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
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def prediction_badge(prediction):
    """Color-coded prediction badge."""
    if prediction == "Circularise":
        return "🟢 **Circularise** (self-splicing)"
    elif prediction == "Linearise":
        return "🟡 **Linearise** (cross-splicing)"
    else:
        return "🔴 **Not compatible**"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("# 🧬 BISTRO — Intein Compatibility Prediction")
    st.markdown("**Boolean-logic-based Intein Splicing Topology Rule Operator for Programmable Protein Assembly**")
    
    # Load model
    model, le, meta = load_model_and_data()
    model_loaded = model is not None and le is not None
    
    # Sidebar info
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        BISTRO predicts intein compatibility and identifies optimal protein assembly pairs.
        
        **Features:**
        - 🔍 Pair prediction
        - 🎯 Find compatible partners
        - 🧪 Custom sequence search
        """)
        
        st.markdown("---")
        if model_loaded:
            st.success("✅ Model Loaded")
            st.markdown(f"**Mode:** {meta.get('feature_mode', 'Unknown')}")
            st.markdown(f"**CV F1:** {meta.get('cv_f1_mean', 'N/A'):.3f}")
        else:
            st.warning("⚠️ Using Boolean Fallback Mode")
            st.markdown("No trained model found. Using Boolean rules only.")
        
        st.markdown("---")
        st.markdown("📊 **Dataset:** 20 inteins, 400 pairs")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Pair Prediction",
        "🎯 Find Partners",
        "🧪 Custom Sequence",
        "📋 Database"
    ])
    
    # ─ TAB 1: PAIR PREDICTION
    with tab1:
        st.header("Predict Intein Compatibility")
        col1, col2 = st.columns(2)
        
        with col1:
            n_name = st.selectbox("N-terminal Intein", INTEIN_NAMES, key="n_pair")
        with col2:
            c_name = st.selectbox("C-terminal Intein", INTEIN_NAMES, key="c_pair", index=1)
        
        if st.button("🔮 Predict", key="predict_btn", use_container_width=True):
            if n_name == c_name:
                st.warning("⚠️ Please select different inteins for N and C terminals.")
            else:
                with st.spinner("Predicting..."):
                    label, conf, source = predict_pair(n_name, c_name, model, le)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", prediction_badge(label))
                    with col2:
                        st.metric("Confidence", f"{conf:.1%}")
                    with col3:
                        st.metric("Source", source)
                    
                    n_int = INTEIN_BY_NAME[n_name]
                    c_int = INTEIN_BY_NAME[c_name]
                    
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("N-Terminal")
                        st.markdown(f"**Name:** {n_int['name']}")
                        st.markdown(f"**Organism:** {n_int['org']}")
                        st.markdown(f"**Accession:** {n_int['n_acc']}")
                        st.markdown(f"**Group:** {n_int['group']}")
                    with col2:
                        st.subheader("C-Terminal")
                        st.markdown(f"**Name:** {c_int['name']}")
                        st.markdown(f"**Organism:** {c_int['org']}")
                        st.markdown(f"**Accession:** {c_int['c_acc']}")
                        st.markdown(f"**Group:** {c_int['group']}")
    
    # ─ TAB 2: FIND PARTNERS
    with tab2:
        st.header("Find Compatible Partners")
        col1, col2 = st.columns(2)
        
        with col1:
            search_type = st.radio("Search for:", ["N-terminal Partners", "C-terminal Partners"])
        with col2:
            if search_type == "N-terminal Partners":
                query_intein = st.selectbox("Given C-terminal:", INTEIN_NAMES, key="hunt_c_intein")
                hunt_type = "c"
            else:
                query_intein = st.selectbox("Given N-terminal:", INTEIN_NAMES, key="hunt_n_intein")
                hunt_type = "n"
        
        if st.button("🔎 Search Partners", use_container_width=True):
            with st.spinner("Searching..."):
                results = []
                for intein in INTEINS:
                    if intein["name"] == query_intein:
                        continue
                    if hunt_type == "c":
                        # Given C, find N partners
                        label, conf, source = predict_pair(intein["name"], query_intein, model, le)
                        if label:
                            results.append({
                                "Intein": intein["name"],
                                "Organism": intein["org"],
                                "Prediction": label,
                                "Confidence": f"{conf:.1%}",
                                "Source": source
                            })
                    else:
                        # Given N, find C partners
                        label, conf, source = predict_pair(query_intein, intein["name"], model, le)
                        if label:
                            results.append({
                                "Intein": intein["name"],
                                "Organism": intein["org"],
                                "Prediction": label,
                                "Confidence": f"{conf:.1%}",
                                "Source": source
                            })
                
                # Sort by confidence (extract numeric value)
                results.sort(key=lambda x: float(x["Confidence"].rstrip("%"))/100, reverse=True)
                
                if results:
                    st.success(f"Found {len(results)} compatible partners!")
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    circularise = sum(1 for r in results if r["Prediction"] == "Circularise")
                    linearise = sum(1 for r in results if r["Prediction"] == "Linearise")
                    with col1:
                        st.metric("🟢 Circularise", circularise)
                    with col2:
                        st.metric("🟡 Linearise", linearise)
                    with col3:
                        avg_conf = np.mean([float(r["Confidence"].rstrip("%"))/100 for r in results])
                        st.metric("Average Confidence", f"{avg_conf:.1%}")
                else:
                    st.info("ℹ️ No compatible partners found.")
    
    # ─ TAB 3: CUSTOM SEQUENCE
    with tab3:
        st.header("Search Custom Sequence")
        st.markdown("Enter a protein sequence to find the best matching intein and compatible partners.")
        
        col1, col2 = st.columns(2)
        with col1:
            terminal_type = st.radio("Sequence Type:", ["N-terminal", "C-terminal"])
        with col2:
            st.write("")  # Spacer
        
        sequence_input = st.text_area(
            "Paste protein sequence (single letter code):",
            placeholder="MLGPG...",
            height=150
        )
        
        if st.button("🔬 Find Best Match & Partners", use_container_width=True):
            seq = sequence_input.upper().strip()
            term_type = "n" if terminal_type == "N-terminal" else "c"
            
            if not seq:
                st.error("❌ Please enter a sequence.")
            elif len(seq) < 5:
                st.error("❌ Sequence too short (minimum 5 amino acids).")
            else:
                valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                invalid_aa = [aa for aa in seq if aa not in valid_aa]
                
                if invalid_aa:
                    st.error(f"❌ Invalid amino acids: {', '.join(set(invalid_aa))}")
                else:
                    with st.spinner("Finding best match..."):
                        best_intein, similarity = find_best_match(seq, term_type)
                        
                        if not best_intein:
                            st.warning("⚠️ No matching inteins found. Try different sequences.")
                        else:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Best Match", best_intein["name"])
                            with col2:
                                st.metric("Similarity", f"{similarity:.1%}")
                            with col3:
                                st.metric("Organism", best_intein["org"])
                            
                            st.divider()
                            st.subheader("Compatible Partners")
                            
                            results = []
                            if term_type == "n":
                                for c_int in INTEINS:
                                    if c_int["name"] == best_intein["name"]:
                                        continue
                                    label, conf, source = predict_pair(best_intein["name"], c_int["name"], model, le)
                                    if label:
                                        results.append({
                                            "Partner (C-term)": c_int["name"],
                                            "Organism": c_int["org"],
                                            "Prediction": label,
                                            "Confidence": f"{conf:.1%}",
                                            "Source": source
                                        })
                            else:
                                for n_int in INTEINS:
                                    if n_int["name"] == best_intein["name"]:
                                        continue
                                    label, conf, source = predict_pair(n_int["name"], best_intein["name"], model, le)
                                    if label:
                                        results.append({
                                            "Partner (N-term)": n_int["name"],
                                            "Organism": n_int["org"],
                                            "Prediction": label,
                                            "Confidence": f"{conf:.1%}",
                                            "Source": source
                                        })
                            
                            results.sort(key=lambda x: float(x["Confidence"].rstrip("%"))/100, reverse=True)
                            
                            if results:
                                df = pd.DataFrame(results)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.info("ℹ️ No compatible partners found for this sequence.")
    
    # ─ TAB 4: DATABASE
    with tab4:
        st.header("Intein Database")
        st.markdown("Complete list of 20 inteins in BISTRO.")
        
        # Display as table
        db_data = []
        for intein in INTEINS:
            db_data.append({
                "Name": intein["name"],
                "Organism": intein["org"],
                "Group": intein["group"],
                "Self-Only": "✓" if intein["selfOnly"] else "✗",
                "N-acc": intein["n_acc"],
                "C-acc": intein["c_acc"]
            })
        
        df = pd.DataFrame(db_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        st.markdown("### Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Inteins", len(INTEINS))
        with col2:
            st.metric("Total Pairs", len(INTEINS) * len(INTEINS))
        with col3:
            groups = set(it["group"] for it in INTEINS)
            st.metric("Groups", len(groups))
        with col4:
            orthogonal = sum(1 for it in INTEINS if it["selfOnly"])
            st.metric("Orthogonal", orthogonal)
        
        st.markdown("### Intein Groups")
        groups_info = {}
        for intein in INTEINS:
            g = intein["group"]
            if g not in groups_info:
                groups_info[g] = []
            groups_info[g].append(intein["name"])
        
        for group, inteins in sorted(groups_info.items()):
            with st.expander(f"{group} ({len(inteins)} inteins)"):
                st.write(", ".join(inteins))

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
