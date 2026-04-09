# PSA 10 Profit Scanner — Deployment Guide

## File Structure

```
psa10_scanner/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── config.yaml         ← API keys & thresholds (do NOT commit to public repos)
├── .streamlit/
│   └── secrets.toml    ← Alternative to config.yaml for cloud deploy
└── README.md
```

---

## Step 1 — Get eBay API Keys (Free)

1. Go to https://developer.ebay.com and sign in with your eBay account
2. Click **My Account** → **Application Access Keys**
3. Click **Create a Keyset** → choose **Production**
4. Copy your **App ID (Client ID)**, **Cert ID (Client Secret)**, **Dev ID**
5. Paste them into `config.yaml` or `.streamlit/secrets.toml` (see below)

> The Browse API (used here) is free for up to 5,000 calls/day on the
> standard production tier. More than enough for daily scanning.

---

## Step 2 — Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser. The app works in demo mode
(mock listings) if no eBay credentials are configured.

---

## Step 3 — Deploy Free to Streamlit Cloud

### 3a. Push to GitHub

```bash
git init
git add app.py requirements.txt README.md
# Do NOT add config.yaml if it has real keys — use Secrets instead
git commit -m "PSA 10 Profit Scanner"
git remote add origin https://github.com/YOUR_USERNAME/psa10-scanner
git push -u origin main
```

### 3b. Connect on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your repo, branch `main`, and file path `app.py`
5. Click **Advanced settings** → **Secrets**

### 3c. Add Secrets (paste this, filled in with your keys)

```toml
[ebay]
app_id  = "YOUR-EBAY-APP-ID-HERE"
cert_id = "YOUR-EBAY-CERT-ID-HERE"
dev_id  = "YOUR-EBAY-DEV-ID-HERE"
```

6. Click **Deploy** — your app will be live in ~2 minutes at:
   `https://YOUR_USERNAME-psa10-scanner-app-XXXXX.streamlit.app`

7. Bookmark this URL on your phone. The dark-mode UI is mobile-optimised.

---

## CV Grading Logic Summary

| Check | Method | Threshold | Result on Fail |
|---|---|---|---|
| Resolution | Image dimension check | < 1000px | Hard Reject |
| Centering L/R | Canny edge → border ratio | > 55/45 | Hard Reject |
| Centering T/B | Canny edge → border ratio | > 55/45 | Hard Reject |
| Corner Whitening | White pixel cluster in corners | ≥ 30 px | Hard Reject |
| Scratch Detection | Laplacian in glint zones | > 0.3% density | Hard Reject |
| Glare/Glint | Brightness threshold in holo area | > 2% area | Warning |

**A card must pass ALL hard checks to receive a PASS verdict.**
Glint alone = Warning (may just be camera angle — use your eyes).

---

## Notes

- The PSA 10 value estimates are based on recent historical sales.
  Always verify current values on PSAcard.com or 130point.com before buying.
- This tool is for educational/research purposes. It does not guarantee
  grading outcomes. Always manually inspect cards before purchasing.
- For best results, only buy listings with 4+ high-resolution photos
  showing front, back, and all four corners clearly.
