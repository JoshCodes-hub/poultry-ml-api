# 🚀 Deploy Your Poultry ML Model to Render.com (FREE)

This guide takes your trained `.pkl` model from your laptop → live on the internet, so your Lovable web app can use it from anywhere.

**Time needed:** ~15 minutes. **Cost:** Free.

---

## 📋 What You'll Need

1. A **GitHub account** (free) — https://github.com/signup
2. A **Render.com account** (free, sign in with GitHub) — https://render.com
3. **Git installed** on your laptop — https://git-scm.com/downloads

---

## STEP 1 — Push your model to GitHub

Open a terminal **inside the `Train_Poultry_Model` folder** on your laptop and run these commands one by one:

```bash
git init
git add .
git commit -m "Poultry disease ML model API"
```

Now go to https://github.com/new and create a new repository called `poultry-ml-api` (keep it Public, don't add a README).

Then back in your terminal, copy the two commands GitHub shows you. They look like this:

```bash
git remote add origin https://github.com/YOUR-USERNAME/poultry-ml-api.git
git branch -M main
git push -u origin main
```

✅ Your code (and your trained `model.pkl`) is now on GitHub.

---

## STEP 2 — Deploy on Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect account"** under GitHub if first time, then pick your `poultry-ml-api` repo
4. Fill in the form:

| Field | Value |
|---|---|
| **Name** | `poultry-ml-api` |
| **Region** | Frankfurt (closest to Nigeria) |
| **Branch** | `main` |
| **Root Directory** | *(leave empty)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | **Free** |

5. Click **"Create Web Service"**.

Render will spend ~3-5 minutes installing Python, your packages, and starting the API. Watch the logs. When you see:

```
Uvicorn running on http://0.0.0.0:10000
```

✅ Your model is **LIVE on the internet**.

6. Copy your service URL from the top of the page. It looks like:
   `https://poultry-ml-api.onrender.com`

---

## STEP 3 — Connect it to your Lovable app

In the Lovable chat, tell me:

> "Add ML_API_URL secret with value `https://poultry-ml-api.onrender.com`"

I'll wire it in. After that, every symptom check on your web app will:

1. Send the data to YOUR trained Random Forest model on Render
2. Get the prediction back
3. Save it to the database tagged as `ml_model` (instead of `rule_engine`)

If your Render service is sleeping (free tier sleeps after 15 min idle), the app automatically falls back to the rule engine so users never see an error. Brilliant for a defense — show this resilience to your panel!

---

## ⚠️ About the Free Tier

- **Sleeps after 15 min** of inactivity — first request after sleep takes ~30 seconds to wake.
- **750 hours/month** free — more than enough for a final-year project.
- **512 MB RAM** — your model is ~5 MB, fits easily.

For your defense day: hit the URL in your browser once an hour before to keep it warm: `https://poultry-ml-api.onrender.com/health`

---

## 🆘 Troubleshooting

**Build fails with "model.pkl not found"** → Make sure you ran `git add .` and `git push` — the `model/` folder must be on GitHub. Check at `https://github.com/YOUR-USERNAME/poultry-ml-api/tree/main/model`.

**Build fails on `scikit-learn`** → Open `requirements.txt`, make sure it pins versions (e.g. `scikit-learn==1.5.0`). Train your model with the same version locally.

**"Application failed to respond"** → Check the Start Command exactly matches: `uvicorn api.main:app --host 0.0.0.0 --port $PORT` (note `$PORT`, not `8000`).

---

You now have a **real production ML system**. That's a 🔥 final-year project.
