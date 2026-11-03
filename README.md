# 🧵 Artisano — Production, Pricing & Zero-Waste Decision Support

AI-powered assistant for artisans: predicts a fair selling price (XGBoost) and plans the
most profitable daily production (PuLP linear programming). Includes user login / sign-up.

## Run in VS Code

```bash
# 1. open the Artisano folder in VS Code, then open a terminal (Ctrl+`)
python -m venv .venv
.venv\Scripts\activate          # Windows      (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. start the app
streamlit run app.py
```

The first time, click **Sign up** to create an account; after that use **Log in**.

### Only if you see a model-loading error
The included `models/price_model.pkl` was saved with specific scikit-learn / XGBoost versions.
If your versions differ, rebuild it (takes under a minute):

```bash
python src/data_generator.py
python src/train_model.py
```

## Folder layout

```
Artisano/
├── app.py                  # Streamlit app (login gate + 3 tabs)
├── requirements.txt
├── .streamlit/config.toml  # theme colours (Rosewood / Vanilla Cream / Blush)
├── data/
│   ├── artisan_market_data.csv
│   └── users.db            # created automatically on first run (do not commit)
├── models/price_model.pkl
└── src/
    ├── __init__.py
    ├── auth.py             # NEW: login / sign-up (SQLite, hashed passwords)
    ├── data_generator.py
    ├── optimizer.py
    └── train_model.py
```
