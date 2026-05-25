# 🚀 BISTRO Streamlit Deployment Guide

## Quick Start (Local Testing)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app locally
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 📤 Deploy to Streamlit Cloud

### Step 1: Push to GitHub
1. Commit your changes:
```bash
git add streamlit_app.py requirements.txt .streamlit/
git commit -m "Add Streamlit app for deployment"
git push origin main
```

### Step 2: Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"** → **"From existing repo"**
4. Select your repository
5. Specify the main file: `streamlit_app.py`
6. Click **"Deploy"** ✨

### Step 3: Share Your App
Once deployed, your app will be available at:
```
https://your-username-streamlit-app.streamlit.app
```

---

## 📋 File Structure

```
BISTRO/
├── streamlit_app.py          ← Main Streamlit app (USE THIS FOR DEPLOYMENT)
├── app.py                     ← Original Flask app (optional)
├── requirements.txt           ← Python dependencies
├── .streamlit/
│   └── config.toml           ← Streamlit configuration
├── models/
│   ├── bistro_model.pkl      ← Trained ML model
│   ├── bistro_label_encoder.pkl
│   └── bistro_model_meta.json
└── data/
    └── pairs.csv             ← Training data
```

---

## ✨ Features Included

✅ **Pair Prediction** - Predict compatibility between any two inteins
✅ **Find Partners** - Discover all compatible partners for a given intein
✅ **Custom Sequence Search** - Search for best-matching intein and partners
✅ **Database Browser** - View all 20 inteins with metadata
✅ **ML + Boolean Fallback** - Gracefully handles missing trained model

---

## 🔧 Troubleshooting

### App won't run?
- Check that `requirements.txt` is in the root directory
- Ensure `streamlit_app.py` is in the root directory
- Run `streamlit --version` to verify installation

### Model not found?
- App automatically falls back to Boolean rules
- Place trained model files in `models/` folder
- Model paths must be relative: `models/bistro_model.pkl`

### Slow app?
- Streamlit caches the model with `@st.cache_resource`
- First load may take 10-15 seconds
- Subsequent interactions are instant

---

## 📝 Notes for Deployment

- **Secrets Management**: If you need API keys, use Streamlit Secrets
  - Create `.streamlit/secrets.toml` (not in GitHub)
  - Access via `st.secrets["key_name"]`

- **Large Files**: GitHub repo size limit is ~100MB
  - Consider uploading `models/` to cloud storage separately if needed

- **Performance**: Streamlit reruns the entire script on interaction
  - Use `@st.cache_resource` for heavy computations (already done ✓)

---

## 🎯 Next Steps

1. **Local Testing**: `streamlit run streamlit_app.py`
2. **Push to GitHub**: Commit and push all changes
3. **Deploy on Streamlit Cloud**: Follow Step 2 above
4. **Share URL**: Send your live app link to collaborators!

---

**Questions?** Check out:
- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/deploy/streamlit-cloud)
