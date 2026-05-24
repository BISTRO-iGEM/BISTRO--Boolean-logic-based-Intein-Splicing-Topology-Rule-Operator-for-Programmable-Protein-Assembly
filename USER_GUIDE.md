# BISTRO Enhanced Version - User Guide

## What's New?

Your BISTRO system has been enhanced with three major improvements:

### 1. 🧬 Advanced Amino Acid Analysis in Model Training
The model now analyzes all 20 standard amino acids in each sequence, providing better predictions based on the actual chemical composition of the protein. This allows the model to make accurate predictions even for novel sequences that weren't in the original training set.

### 2. 🔍 Custom Sequence Search Feature
You can now input any amino acid sequence (even if not in the database) and the system will:
- Find the most similar intein from the database
- Predict all compatible splicing partners
- Display results with similarity scores and confidence levels

### 3. 📊 Cleaner UI
- Removed F1 Score metric from the model information box
- Added intuitive "Search Custom Sequence" tab for easy access

---

## How to Use

### Running the System

1. **Train the model** (if not already done):
```bash
python train_model.py
```
This trains the ML model using the enhanced amino acid composition features.

2. **Start the web server**:
```bash
python app.py
```

3. **Open the website**:
Open `static/index.html` in your web browser (or visit `http://localhost:5000`)

---

## Using the New Features

### Feature 1: Predict Intein Compatibility (Existing)
- Go to the **"Predict"** tab
- Select N-terminal and C-terminal inteins from dropdowns
- Click **"Predict Compatibility"**
- See prediction results with confidence scores

### Feature 2: Search Custom Sequence (NEW!)
1. Click the **"🔍 Search Custom Sequence"** tab
2. **Paste your amino acid sequence** in the text area
   - Example: `MLGPGASTSAKNQAGVEGAGATVHSK...`
   - Minimum length: 5 amino acids
3. **Select sequence type**:
   - N-terminal Intein: if your sequence is an N-terminal half
   - C-terminal Intein: if your sequence is a C-terminal half
4. Click **"Find Best Match & Compatible Partners"**
5. View results showing:
   - **Best Match**: Most similar intein in database with similarity %
   - **Compatible Partners**: All inteins this sequence can splice with
   - **Predictions**: Circularise, Linearise, or Not compatible
   - **Confidence**: Percentage confidence for each prediction

### Feature 3: Find All Partners (Existing)
- **"Find N-term Partners"**: Select a C-terminal intein, find all compatible N-terminal partners
- **"Find C-term Partners"**: Select an N-terminal intein, find all compatible C-terminal partners

### Feature 4: View Database
- **"All Inteins"**: Browse all 20 inteins in the database with their details

---

## Example Workflow

**Scenario**: You have a novel protein sequence and want to find a suitable splicing partner.

1. Go to **"🔍 Search Custom Sequence"** tab
2. Paste your protein sequence
3. Select if it's N-terminal or C-terminal
4. Click search button
5. System shows:
   - The most similar intein: "Ssp DnaE (93% similarity)"
   - All compatible C-terminal partners (or N-terminal, depending on your selection)
   - Confidence scores for each prediction
6. Choose the best partner based on:
   - Prediction type (Circularise best for construct closure)
   - Confidence score (higher is better)
   - Organism compatibility with your application

---

## Key Features

✨ **What's Better:**
- Model makes predictions based on actual amino acid composition
- Can work with novel sequences not in the database
- Similarity matching finds closest database matches
- More accurate predictions for new intein variants

🎯 **Validation:**
- Only accepts standard amino acids (A-Y)
- Minimum sequence length of 5 amino acids
- Case-insensitive input
- Clear error messages if validation fails

---

## Understanding Results

### Prediction Categories:
- **Circularise**: The intein splices with itself (self-splicing), ideal for creating circular proteins
- **Linearise**: The intein splices with compatible partners in the same protein family
- **Not compatible**: These inteins cannot splice together

### Confidence Score:
- Ranges from 0 to 100%
- Higher confidence = more reliable prediction
- Based on machine learning model training

### Similarity Score (for custom sequences):
- Ranges from 0 to 100%
- Shows how similar your sequence is to the best match in database
- Higher = more similar chemical composition

---

## Technical Details

### Amino Acids Analyzed:
- All 20 standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

### Feature Extraction:
- Physicochemical properties (13 features per sequence)
- Complete amino acid composition (20 features per sequence)
- Conserved intein motifs (5 features per sequence)
- Total: 76 features per sequence pair

### Sequence Similarity Algorithm:
- Character-by-character alignment
- Length-normalized scoring
- Handles sequences of different lengths

---

## Tips for Best Results

1. **Ensure sequence quality**: Make sure your sequence is correct and contains only valid amino acids
2. **Check database matches**: Higher similarity scores mean more reliable partner predictions
3. **Review multiple options**: Look at top compatible partners with highest confidence scores
4. **Consider your application**: Choose "Circularise" for circular proteins, "Linearise" for linear constructs

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Sequence too short" | Input at least 5 amino acids |
| "Invalid amino acids" | Only use standard amino acids A-Y, remove numbers/special chars |
| "No matching inteins found" | Try checking the full database in "All Inteins" tab |
| Model not loaded | Run `python train_model.py` first, then restart the app |

---

## Files Modified

- `train_model.py` - Enhanced feature extraction with amino acid composition
- `app.py` - Added sequence similarity matching and custom sequence search endpoint
- `static/index.html` - Removed F1 Score, added custom sequence search UI

---

## Need Help?

Refer to the `CHANGES_SUMMARY.md` file for detailed technical documentation.

---

**Version**: Enhanced 2.0  
**Last Updated**: May 24, 2026
