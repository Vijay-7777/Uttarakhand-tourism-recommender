from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import os

app = Flask(__name__)

# Load dataset and models
df = pd.read_csv('data/uttarakhand_places.csv')

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

    # Debug: Print CSV columns
    print("CSV Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())

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

    # Fix column name mapping based on your actual CSV
    # Map different possible column names to standard names
    column_mapping = {}
    
    # Check for different variations of column names
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'name' in col_lower:
            column_mapping['Name'] = col
        elif 'type' in col_lower:
            column_mapping['Type'] = col
        elif 'season' in col_lower:
            column_mapping['Best_Season'] = col
        elif 'cost' in col_lower:
            column_mapping['Avg_Cost'] = col
        elif 'duration' in col_lower:
            column_mapping['Typical_Duration'] = col
        elif 'image' in col_lower or 'url' in col_lower:
            column_mapping['Image_URL'] = col
        elif 'description' in col_lower:
            column_mapping['Description'] = col
        elif 'latitude' in col_lower:
            column_mapping['Latitude'] = col
        elif 'longitude' in col_lower:
            column_mapping['Longitude'] = col

    print("Column mapping:", column_mapping)

    # Apply dynamic filters using the correct column names
    try:
        season_col = column_mapping.get('Best_Season', 'season')
        cost_col = column_mapping.get('Avg_Cost', 'cost')
        duration_col = column_mapping.get('Typical_Duration', 'duration')
        
        filtered_df = recommended_df[
            (recommended_df[season_col].str.lower() == season.lower()) &
            (recommended_df[cost_col] <= budget) &
            (recommended_df[duration_col] <= duration)
        ]
    except Exception as e:
        print(f"Filtering error: {e}")
        filtered_df = pd.DataFrame()

    # If no filtered results, fallback to top 3 recommendations without filters
    if filtered_df.empty:
        filtered_df = recommended_df.head(3)

    # Process recommendations and standardize column names
    recommendations = []
    for _, row in filtered_df.iterrows():
        # Get image filename
        image_col = column_mapping.get('Image_URL', 'Image_URL')
        image_url = row.get(image_col, '')
        
        if pd.isna(image_url) or not image_url:
            image_filename = f"{row[column_mapping.get('Name', 'Name')].lower().replace(' ', '_').replace(',', '')}.jpg"
        else:
            image_filename = image_url
            
        # Check if image file exists
        image_path = os.path.join('static', 'images', image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            # Try alternative naming
            alt_filename = f"{row[column_mapping.get('Name', 'Name')].lower().replace(' ', '_').replace(',', '').replace('_national_park', '').replace('_lake', '')}.jpg"
            alt_path = os.path.join('static', 'images', alt_filename)
            if os.path.exists(alt_path):
                image_filename = alt_filename
            else:
                print(f"Alternative image also not found: {alt_path}")
        
        recommendation = {
            'Name': row[column_mapping.get('Name', 'Name')],
            'Type': row[column_mapping.get('Type', 'type')],
            'Best_Season': row[column_mapping.get('Best_Season', 'season')],
            'Avg_Cost': row[column_mapping.get('Avg_Cost', 'cost')],
            'Typical_Duration': row[column_mapping.get('Typical_Duration', 'duration')],
            'Description': row[column_mapping.get('Description', 'Description')],
            'Image_URL': image_filename,
            'Latitude': row[column_mapping.get('Latitude', 'Latitude')],
            'Longitude': row[column_mapping.get('Longitude', 'Longitude')]
        }
        recommendations.append(recommendation)
        print(f"Recommendation: {recommendation['Name']}, Image: {recommendation['Image_URL']}")

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)