from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load dataset and models
df = pd.read_csv('data/uttarakhand_places_50plus.csv')

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
    interests = request.form.getlist('interest')
    season = request.form['season']
    budget = float(request.form['budget'])
    duration = float(request.form['duration'])

    # Preprocess user input features
    interests_cleaned = [[i.strip().lower() for i in interests]]
    interest_features = tag_binarizer.transform(interests_cleaned)

    season_feature = season_encoder.transform([[season]])
    numeric_features = scaler.transform([[budget, duration]])

    # Combine features into one sparse matrix
    X_input = hstack([
        csr_matrix(interest_features),
        csr_matrix(season_feature),
        csr_matrix(numeric_features)
    ])

    # Find nearest neighbors using the trained model
    distances, indices = model.kneighbors(X_input)

    # indices is a 2D array, extract first row (first query)
    indices_list = indices[0] if indices.shape[0] > 0 else []

    # Defensive: check if indices_list is empty
    if len(indices_list) == 0:
        recommended_df = pd.DataFrame()  # no recommendations found
    else:
        # Select rows from df corresponding to recommended indices
        recommended_df = df.iloc[indices_list]

    # Apply dynamic filters on recommended_df
    # Adjust column names according to your CSV
    filtered_df = recommended_df[
        (recommended_df['Best_Season'].str.lower() == season.lower()) &
        (recommended_df['Avg_Cost'] <= budget) &
        (recommended_df['Typical_Duration'] <= duration)
    ]

    # If no filtered results, fallback to top 3 recommendations without filters
    if filtered_df.empty:
        filtered_df = recommended_df.head(3)

    # Convert final recommendations to dict for template rendering
    recommendations = filtered_df.to_dict(orient='records')

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

