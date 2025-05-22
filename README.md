# 🏔️ Uttarakhand Tourism Recommender (AI/ML Project)

A personalized travel recommendation system that suggests the best destinations in **Uttarakhand, India** based on your interests, budget, trip duration, and preferred season — powered by Machine Learning.

---

## 🚀 Features

- 🔍 Recommend destinations using KNN based on:
  - User travel interests (e.g., Hill Station, Adventure)
  - Preferred season
  - Budget
  - Trip duration
- 📍 Includes geolocation (latitude/longitude) for map-based integration
- ✨ TailwindCSS-based modern UI
- 🧠 ML-based filtering using:
  - MultiLabelBinarizer
  - OneHotEncoder
  - KNN (Nearest Neighbors)

---

## 🛠️ Tech Stack

- Python + Flask
- Scikit-learn
- Pandas + NumPy
- HTML5 + Tailwind CSS
- Jinja2 Templates
- Google Maps (optional)

---

## 🗂️ Folder Structure

uttarakhand-tourism-recommender/
├── app/ # Flask application logic
│ ├── recommender.py
│ ├── routes.py
│ └── utils.py
├── data/ # CSV dataset with all destinations
├── models/ # Trained ML models (Pickled)
├── static/ # CSS and assets
├── templates/ # HTML templates (index.html, recommendations.html)
├── app.py # Main Flask entry point
├── train_model.py # ML training script
├── requirements.txt # Dependencies
└── README.md
