# InternCircle Project 05 — Streamlit Interactive ML App

This app wraps a Decision Tree classifier trained on the Seaborn Titanic dataset.

## Files
- `app.py`: interactive Streamlit interface
- `train_model.py`: trains the classifier and saves `model.joblib`
- `requirements.txt`: Python dependencies

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `train_model.py`, and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.
3. The app will train and save the model if `model.joblib` is absent (see `app.py` startup block).
4. Test the public URL on your phone before submitting.

The app is an educational example based on historical data, not advice.
