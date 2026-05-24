"""
BISTRO — Step 2: Model Training
================================
- Loads data/pairs.csv from step 1
- Extracts features:
    Option A: Physicochemical (fast, no GPU needed)   ← default
    Option B: ESM-2 embeddings (better, needs GPU)    ← set USE_ESM2 = True
- Handles class imbalance with SMOTE
- Trains Random Forest classifier
- Evaluates with 5-fold cross-validation
- Saves model to models/bistro_model.pkl

Run: python 2_train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change USE_ESM2 to True if you have GPU + fair-esm installed
# ─────────────────────────────────────────────────────────────────────────────

USE_ESM2 = False   # set True for better accuracy (needs GPU)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — Option A: Physicochemical
# ─────────────────────────────────────────────────────────────────────────────

def amino_acid_composition_features(seq: str) -> list:
    """
    Extract detailed amino acid composition features.
    Analyzes all 20 standard amino acids.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(aa if aa in valid_aa else "A" for aa in seq.upper())
    
    if len(cleaned) < 5:
        return [0.0] * 20
    
    try:
        a = ProteinAnalysis(cleaned)
        aa_comp = a.get_amino_acids_percent()
        
        # All 20 amino acids in order
        amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                       "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
        return [aa_comp.get(aa, 0.0) for aa in amino_acids]
    except Exception:
        return [0.0] * 20


def physicochemical_features(seq: str) -> list:
    """
    Extract 13 physicochemical features from a protein sequence.
    Uses only standard amino acids; replaces unknowns with Ala.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    # Clean sequence — remove placeholders and non-standard AAs
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(aa if aa in valid_aa else "A" for aa in seq.upper())

    if len(cleaned) < 5:
        return [0.0] * 13

    try:
        a = ProteinAnalysis(cleaned)
        mw          = a.molecular_weight()
        pi          = a.isoelectric_point()
        instab      = a.instability_index()
        gravy       = a.gravy()
        helix, turn, sheet = a.secondary_structure_fraction()
        aa_comp     = a.get_amino_acids_percent()

        # key residues relevant to intein splicing (Cys, His, Asn, Ser, Thr)
        cys_frac  = aa_comp.get("C", 0.0)
        his_frac  = aa_comp.get("H", 0.0)
        asn_frac  = aa_comp.get("N", 0.0)
        ser_frac  = aa_comp.get("S", 0.0)
        thr_frac  = aa_comp.get("T", 0.0)
        length    = float(len(cleaned))

        return [mw, pi, instab, gravy, helix, turn, sheet,
                cys_frac, his_frac, asn_frac, ser_frac, thr_frac, length]
    except Exception:
        return [0.0] * 13


def motif_features(seq: str) -> list:
    """
    Check presence of key conserved intein motifs (Blocks A–G).
    Returns binary features [0 or 1] for each motif group.
    """
    seq = seq.upper()
    motif_groups = {
        "block_b_hnf":  ["HNF", "HNY", "HNL", "HNC"],   # Block F His-Asn
        "nc_dipeptide": ["NC", "SC", "TC"],               # Block G C-terminal
        "cfk_block_a":  ["CFK", "SFK", "TFK", "CFN"],    # Block A nucleophile
        "block_b_ser":  ["SH", "CH", "TH"],               # Block B
        "txxh_motif":   ["TXXH", "SXXH"],                 # conserved tetrapeptide
    }
    feats = []
    for motifs in motif_groups.values():
        found = 0
        for m in motifs:
            if "X" in m:
                # wildcard match — check every 4-mer
                for i in range(len(seq) - len(m) + 1):
                    window = seq[i:i+len(m)]
                    match = all(w == "X" or w == s for w, s in zip(m, window))
                    if match:
                        found = 1
                        break
            elif m in seq:
                found = 1
                break
        feats.append(float(found))
    return feats


def extract_features_A(n_seq: str, c_seq: str) -> np.ndarray:
    """Combine physicochemical + amino acid composition + motif features for N and C sequences."""
    n_phys  = physicochemical_features(n_seq)
    c_phys  = physicochemical_features(c_seq)
    n_aacomp = amino_acid_composition_features(n_seq)
    c_aacomp = amino_acid_composition_features(c_seq)
    n_motif = motif_features(n_seq)
    c_motif = motif_features(c_seq)
    # cross-features: length ratio, MW ratio
    n_len   = float(len(n_seq)) if n_seq else 1.0
    c_len   = float(len(c_seq)) if c_seq else 1.0
    cross   = [n_len / (c_len + 1e-6), abs(n_len - c_len)]
    return np.array(n_phys + c_phys + n_aacomp + c_aacomp + n_motif + c_motif + cross, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — Option B: ESM-2 Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_esm2():
    import torch
    import esm
    print("Loading ESM-2 model (this may take a minute)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter  = alphabet.get_batch_converter()
    model.eval()
    return model, batch_converter


def get_esm2_embedding(seq: str, name: str, model, batch_converter) -> np.ndarray:
    import torch
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned  = "".join(aa if aa in valid_aa else "A" for aa in seq.upper())
    if len(cleaned) < 5:
        return np.zeros(320, dtype=np.float32)

    data   = [(name, cleaned[:1022])]  # ESM-2 max length
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        out = model(tokens, repr_layers=[6])
    emb = out["representations"][6].mean(1).squeeze().numpy()
    return emb.astype(np.float32)


def extract_features_B(n_seq, c_seq, model, batch_converter):
    """ESM-2: concatenate mean-pooled N and C embeddings."""
    n_emb = get_esm2_embedding(n_seq, "intein_N", model, batch_converter)
    c_emb = get_esm2_embedding(c_seq, "intein_C", model, batch_converter)
    return np.concatenate([n_emb, c_emb])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("BISTRO — Model Training")
    print("=" * 55)

    # 1. Load data
    print("\n[1/5] Loading data...")
    if not os.path.exists("data/pairs.csv"):
        print("ERROR: data/pairs.csv not found. Run 1_prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv("data/pairs.csv")
    print(f"  {len(df)} pairs loaded")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Feature extraction
    print(f"\n[2/5] Extracting features (mode: {'ESM-2' if USE_ESM2 else 'Physicochemical'})...")

    if USE_ESM2:
        model_esm, bc = load_esm2()
        X = np.array([
            extract_features_B(row["n_seq"], row["c_seq"], model_esm, bc)
            for _, row in df.iterrows()
        ])
    else:
        X = np.array([
            extract_features_A(row["n_seq"], row["c_seq"])
            for _, row in df.iterrows()
        ])

    print(f"  Feature matrix shape: {X.shape}")

    # 3. Encode labels
    print("\n[3/5] Encoding labels...")
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    print(f"  Classes: {list(le.classes_)}")

    # 4. Handle class imbalance with SMOTE
    print("\n[4/5] Balancing classes with SMOTE...")
    counts = np.bincount(y)
    if min(counts) >= 2:
        sm      = SMOTE(random_state=42, k_neighbors=min(5, min(counts) - 1))
        X_bal, y_bal = sm.fit_resample(X, y)
        print(f"  Before: {dict(zip(le.classes_, counts))}")
        print(f"  After:  {dict(zip(le.classes_, np.bincount(y_bal)))}")
    else:
        print("  WARNING: Not enough samples per class for SMOTE. Using raw data.")
        X_bal, y_bal = X, y

    # 5. Train and evaluate with multiple models
    print("\n[5/5] Training and evaluating multiple algorithms...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model 1: Optimized Gradient Boosting (usually better than Random Forest)
    print("\n  ► Gradient Boosting with tuning...")
    gb_params = {
        'n_estimators': [150, 200, 250],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5, 6],
        'min_samples_split': [3, 5],
    }
    gb = GradientBoostingClassifier(random_state=42, subsample=0.8)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1_macro', n_jobs=-1)
    gb_grid.fit(X_bal, y_bal)
    gb_scores = cross_val_score(gb_grid.best_estimator_, X_bal, y_bal, cv=cv, scoring="f1_macro")
    print(f"    Best params: {gb_grid.best_params_}")
    print(f"    5-Fold CV F1 (macro): {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")
    
    # Model 2: Optimized Random Forest
    print("\n  ► Random Forest with tuning...")
    rf_params = {
        'n_estimators': [200, 300, 400],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1_macro', n_jobs=-1)
    rf_grid.fit(X_bal, y_bal)
    rf_scores = cross_val_score(rf_grid.best_estimator_, X_bal, y_bal, cv=cv, scoring="f1_macro")
    print(f"    Best params: {rf_grid.best_params_}")
    print(f"    5-Fold CV F1 (macro): {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # Select best model
    if gb_scores.mean() >= rf_scores.mean():
        print(f"\n  ✓ Gradient Boosting selected (F1: {gb_scores.mean():.3f})")
        clf = gb_grid.best_estimator_
        model_name = "Gradient Boosting"
    else:
        print(f"\n  ✓ Random Forest selected (F1: {rf_scores.mean():.3f})")
        clf = rf_grid.best_estimator_
        model_name = "Random Forest"

    # Full report on original data
    y_pred = clf.predict(X)
    print("\n  Classification report (on full original data):")
    print(classification_report(y, y_pred, target_names=le.classes_))

    print("  Confusion matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  Labels: {list(le.classes_)}")
    print(cm)

    # Feature importances (if available)
    if hasattr(clf, 'feature_importances_'):
        print("\n  Top 10 feature importances:")
        feat_names = (
            [f"N_phys_{i}" for i in range(13)] +
            [f"C_phys_{i}" for i in range(13)] +
            [f"N_motif_{i}" for i in range(5)] +
            [f"C_motif_{i}" for i in range(5)] +
            ["len_ratio", "len_diff"]
        ) if not USE_ESM2 else [f"esm_dim_{i}" for i in range(X.shape[1])]

        importances = clf.feature_importances_
        top10 = np.argsort(importances)[::-1][:10]
        for rank, idx in enumerate(top10):
            name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
            print(f"    {rank+1:2}. {name:<20} {importances[idx]:.4f}")
    
    best_f1 = max(gb_scores.mean(), rf_scores.mean())

    # 6. Save model
    joblib.dump(clf, "models/bistro_model.pkl")
    joblib.dump(le,  "models/bistro_label_encoder.pkl")

    # Save metadata
    meta = {
        "feature_mode":  "ESM2" if USE_ESM2 else "Physicochemical",
        "model_type":    model_name,
        "n_features":    int(X.shape[1]),
        "classes":       list(le.classes_),
        "cv_f1_mean":    float(best_f1),
        "n_train_pairs": len(df),
    }
    with open("models/bistro_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\n" + "=" * 55)
    print("Model saved to models/bistro_model.pkl")
    print("Metadata saved to models/bistro_model_meta.json")
    print("=" * 55)
    print("\nNext step: python 3_app.py")


if __name__ == "__main__":
    main()