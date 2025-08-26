# 🥊 UFC Fight Predictor Dashboard

![UFC Dashboard GIF](https://github.com/lbransby1/lbransby1/blob/main/MMAMetrics.gif)

Predict UFC fight outcomes with machine learning! Compare fighters, analyze stats, and see visualized predictions in a sleek interactive dashboard.

---

## 🚀 Features
- Compare fighter stats side-by-side
- Predict fight winners using ensemble ML models
- Interactive radar charts & performance tables
- Fighter profiles with images, strengths, and weaknesses
- Swap-averaged predictions to reduce bias

---

## 🛠 Tech Stack
- Python 🐍
- Streamlit for UI
- Scikit-learn, XGBoost, LightGBM, CatBoost for ML
- Plotly for radar charts
- Pandas & NumPy for data manipulation
- BeautifulSoup for fighter images scraping

---


## ⚡ Quick Start  

1. **Clone the repo**  
```bash
git clone https://github.com/lbransby1/ufc-fight-predictor.git
cd ufc-fight-predictor
```
Install dependencies
```bash
pip install -r requirements.txt
```
Run the Streamlit app
```bash
streamlit run app.py
```
👉 The dashboard should now be live at http://localhost:8501.


## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the dashboard at `http://localhost:8501/`

---

## 🌍 Live Demo

👉 [View the Dashboard](https://ufc-fight-predictor.streamlit.app/)

👉 [View the GitHub Repo](https://github.com/lbransby1/UFC-Fight-Predictor)

---

## ⚔️ Challenges Faced

* Handling **messy fight data** (typos, missing values, inconsistent stats)
* Balancing ML models and avoiding **overfitting**
* Getting fighter **images dynamically** from scraping
* Optimizing ensemble voting & swap-averaging for **fairer predictions**
* Deploying a **smooth Streamlit app** with heavy models

---

## 📖 Future Improvements

* Add round-by-round simulation
* Track fighter ELO rankings over time
* Improve data scraping pipeline
* Add betting odds & comparison to model predictions

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License

MIT License © [Luke Bransby](https://github.com/lbransby1)

```

---

