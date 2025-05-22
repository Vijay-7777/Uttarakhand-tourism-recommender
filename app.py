from flask import Flask, render_template, request
import pandas as pd
import pickle
from scipy.sparse import hstack

app = Flask(__name__)

# Load dataset (for showing results info)
df = pd.read_csv('data/uttarakhand_places.csv')

# Load saved model and encoders once at startup
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tag_binarizer.pkl', 'rb') as f:
    tag_binarizer = pickle.load(f)
with open('models/season_encoder.pkl', 'rb') as f:
    season_encoder = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get and clean user inputs
    interests = request.form.getlist('interest')  # list from multi-select
    interests = [i.strip().lower() for i in interests]

    season = request.form['season'].strip().lower()
    budget = float(request.form['budget'])
    duration = float(request.form['duration'])

    # Transform interests using tag_binarizer
    interest_vec = tag_binarizer.transform([interests])

    # Transform season using season_encoder
    season_vec = season_encoder.transform([[season]])

    # Scale numeric features using scaler
    num_vec = scaler.transform([[budget, duration]])

    # Combine all features
    user_features = hstack([interest_vec, season_vec, num_vec])

    # Find nearest neighbors
    distances, indices = model.kneighbors(user_features, n_neighbors=5)

    # Get recommended places info
    recommended_places = df.iloc[indices[0]].to_dict(orient='records')

    return render_template('recommendations.html', recommendations=recommended_places)

if __name__ == '__main__':
    app.run(debug=True)
