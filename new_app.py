from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import os
import sys

# Import the recommendation algorithms
sys.path.append('.')  # Add current directory to path
from clustering_recommender import ClusteringRecommender
from collaborative_filtering_based import CollaborativeFilteringRecommender
from content_based_recommender import ContentBasedRecommender

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data/uttarakhand_places.csv')

# Initialize recommenders
clustering_recommender = None
collaborative_recommender = None
content_recommender = None

# Load or train models
def load_models():
    global clustering_recommender, collaborative_recommender, content_recommender

    try:
        # Clustering Recommender
        clustering_recommender = ClusteringRecommender(n_clusters=5, algorithm='kmeans')
        if os.path.exists('models/clustering_model.pkl'):
            clustering_recommender.load_model('models/clustering_model.pkl')
        clustering_recommender.preprocess_data(df)  # Always set df & fit

        # Collaborative Filtering
        collaborative_recommender = CollaborativeFilteringRecommender(n_components=8, algorithm='nmf')
        if os.path.exists('models/collaborative_filtering_model.pkl'):
            collaborative_recommender.load_model()
        collaborative_recommender.preprocess_data(df)

        # Content-Based Recommender
        content_recommender = ContentBasedRecommender()
        if os.path.exists('models/content_based_model.pkl'):
            content_recommender.load_model()
        content_recommender.preprocess_data(df)

        print("✅ All recommenders initialized and ready.")
    
    except Exception as e:
        print(f"❌ Error initializing recommenders: {e}")
        try:
            with open('models/model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('models/tag_binarizer.pkl', 'rb') as f:
                tag_binarizer = pickle.load(f)
            with open('models/season_encoder.pkl', 'rb') as f:
                season_encoder = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, tag_binarizer, season_encoder, scaler
        except Exception as fallback_error:
            print("❌ Fallback models also failed.")
            return None, None, None, None

# Load models on startup
fallback_models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        interests = request.form.getlist('interest')
        season = request.form['season']
        budget = float(request.form['budget'])
        duration = float(request.form['duration'])
        algorithm = request.form.get('algorithm', 'hybrid')  # Default to hybrid

        print(f"User input: {interests}, {season}, ₹{budget}, {duration} days, {algorithm}")

        column_mapping = get_column_mapping()
        recommendations = []

        if algorithm == 'clustering' and clustering_recommender:
            recommended_indices = clustering_recommender.get_recommendations(
                interests, season, budget, duration, top_k=5
            )
            recommendations = process_recommendations(recommended_indices, column_mapping, "Clustering-based")

        elif algorithm == 'collaborative' and collaborative_recommender:
            recommended_indices = collaborative_recommender.get_recommendations(
                interests, season, budget, duration, top_k=5
            )
            recommendations = process_recommendations(recommended_indices, column_mapping, "Collaborative Filtering")

        elif algorithm == 'content' and content_recommender:
            recommended_indices = content_recommender.get_recommendations(
                interests, season, budget, duration, top_k=5
            )
            recommendations = process_recommendations(recommended_indices, column_mapping, "Content-based")

        elif algorithm == 'hybrid':
            recommendations = get_hybrid_recommendations(interests, season, budget, duration, column_mapping)

        else:
            recommendations = get_fallback_recommendations(interests, season, budget, duration, column_mapping)

        if recommendations is None:
            recommendations = []

        return render_template("recommendations.html", recommendations=recommendations, algorithm_used=algorithm.title())

    except Exception as e:
        print(f"Unhandled error in /recommend route: {e}")
        return render_template("recommendations.html", recommendations=[], algorithm_used="Unknown")



def get_column_mapping():
    """Get column mapping for the dataset"""
    column_mapping = {}
    
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
    
    return column_mapping

def process_recommendations(indices, column_mapping, algorithm_name):
    """Process recommendation indices into final format"""
    recommendations = []
    
    for idx in indices:
        try:
            if isinstance(idx, (int, np.integer)):
                row = df.iloc[idx]
            else:
                row = df.loc[idx]
            
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
                alt_filename = f"{row[column_mapping.get('Name', 'Name')].lower().replace(' ', '_').replace(',', '').replace('_national_park', '').replace('_lake', '')}.jpg"
                alt_path = os.path.join('static', 'images', alt_filename)
                if os.path.exists(alt_path):
                    image_filename = alt_filename
            
            recommendation = {
                'Name': row[column_mapping.get('Name', 'Name')],
                'Type': row[column_mapping.get('Type', 'Type')],
                'Best_Season': row[column_mapping.get('Best_Season', 'Best_Season')],
                'Avg_Cost': row[column_mapping.get('Avg_Cost', 'Avg_Cost')],
                'Typical_Duration': row[column_mapping.get('Typical_Duration', 'Typical_Duration')],
                'Description': row[column_mapping.get('Description', 'Description')],
                'Image_URL': image_filename,
                'Latitude': row[column_mapping.get('Latitude', 'Latitude')],
                'Longitude': row[column_mapping.get('Longitude', 'Longitude')],
                'Algorithm': algorithm_name
            }
            recommendations.append(recommendation)
            
        except Exception as e:
            print(f"Error processing recommendation {idx}: {e}")
            continue
    
    return recommendations

def get_hybrid_recommendations(interests, season, budget, duration, column_mapping):
    """Get hybrid recommendations by combining all three algorithms"""
    all_recommendations = {}
    
    try:
        # Get recommendations from each algorithm
        if clustering_recommender:
            cluster_indices = clustering_recommender.get_recommendations(interests, season, budget, duration, top_k=3)
            for idx in cluster_indices:
                if idx not in all_recommendations:
                    all_recommendations[idx] = {'score': 0, 'algorithms': []}
                all_recommendations[idx]['score'] += 3  # Higher weight for clustering
                all_recommendations[idx]['algorithms'].append('Clustering')
        
        if collaborative_recommender:
            collab_indices = collaborative_recommender.get_recommendations(interests, season, budget, duration, top_k=3)
            for idx in collab_indices:
                if idx not in all_recommendations:
                    all_recommendations[idx] = {'score': 0, 'algorithms': []}
                all_recommendations[idx]['score'] += 2  # Medium weight for collaborative
                all_recommendations[idx]['algorithms'].append('Collaborative')
        
        if content_recommender:
            content_indices = content_recommender.get_recommendations(interests, season, budget, duration, top_k=3)
            for idx in content_indices:
                if idx not in all_recommendations:
                    all_recommendations[idx] = {'score': 0, 'algorithms': []}
                all_recommendations[idx]['score'] += 1  # Lower weight for content-based
                all_recommendations[idx]['algorithms'].append('Content-based')
        
        # Sort by score and get top 5
        sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
        top_indices = [idx for idx, _ in sorted_recommendations]
        
        # Process recommendations
        recommendations = []
        for idx in top_indices:
            try:
                row = df.iloc[idx] if isinstance(idx, (int, np.integer)) else df.loc[idx]
                
                image_col = column_mapping.get('Image_URL', 'Image_URL')
                image_url = row.get(image_col, '')
                
                if pd.isna(image_url) or not image_url:
                    image_filename = f"{row[column_mapping.get('Name', 'Name')].lower().replace(' ', '_').replace(',', '')}.jpg"
                else:
                    image_filename = image_url
                
                recommendation = {
                    'Name': row[column_mapping.get('Name', 'Name')],
                    'Type': row[column_mapping.get('Type', 'Type')],
                    'Best_Season': row[column_mapping.get('Best_Season', 'Best_Season')],
                    'Avg_Cost': row[column_mapping.get('Avg_Cost', 'Avg_Cost')],
                    'Typical_Duration': row[column_mapping.get('Typical_Duration', 'Typical_Duration')],
                    'Description': row[column_mapping.get('Description', 'Description')],
                    'Image_URL': image_filename,
                    'Latitude': row[column_mapping.get('Latitude', 'Latitude')],
                    'Longitude': row[column_mapping.get('Longitude', 'Longitude')],
                    'Algorithm': f"Hybrid ({', '.join(all_recommendations[idx]['algorithms'])})",
                    'Score': all_recommendations[idx]['score']
                }
                recommendations.append(recommendation)
            except Exception as e:
                print(f"Error in hybrid processing for {idx}: {e}")
                continue
        
        return recommendations
        
    except Exception as e:
        print(f"Error in hybrid recommendations: {e}")
        return get_fallback_recommendations(interests, season, budget, duration, column_mapping)

def get_fallback_recommendations(interests, season, budget, duration, column_mapping):
    """Fallback recommendation using original KNN model or simple filtering"""
    try:
        if fallback_models[0] is not None:  # Use original KNN model
            model, tag_binarizer, season_encoder, scaler = fallback_models
            
            interests_cleaned = [[i.strip().lower() for i in interests]]
            interest_features = tag_binarizer.transform(interests_cleaned)
            season_feature = season_encoder.transform([[season]])
            numeric_features = scaler.transform([[budget, duration]])

            X_input = hstack([
                csr_matrix(interest_features),
                csr_matrix(season_feature),
                csr_matrix(numeric_features)
            ])

            distances, indices = model.kneighbors(X_input)
            indices_list = indices[0] if indices.shape[0] > 0 else []
            
            if len(indices_list) > 0:
                return process_recommendations(indices_list[:5], column_mapping, "KNN Fallback")
        
        # Simple filtering fallback
        filtered_df = df[
            (df[column_mapping.get('Avg_Cost', 'Avg_Cost')] <= budget * 1.2) &
            (df[column_mapping.get('Typical_Duration', 'Typical_Duration')] <= duration + 1)
        ]
        
        if filtered_df.empty:
            filtered_df = df.head(5)
        else:
            filtered_df = filtered_df.head(5)
        
        return process_recommendations(filtered_df.index.tolist(), column_mapping, "Simple Filter")
        
    except Exception as e:
        print(f"Error in fallback: {e}")
        # Ultimate fallback - return first 5 destinations
        return process_recommendations(list(range(5)), column_mapping, "Random")

@app.route('/api/algorithms')
def get_algorithms():
    """API endpoint to get available algorithms"""
    algorithms = []
    if clustering_recommender:
        algorithms.append({'value': 'clustering', 'name': 'Clustering-based'})
    if collaborative_recommender:
        algorithms.append({'value': 'collaborative', 'name': 'Collaborative Filtering'})
    if content_recommender:
        algorithms.append({'value': 'content', 'name': 'Content-based'})
    algorithms.append({'value': 'hybrid', 'name': 'Hybrid (All Combined)'})
    
    return jsonify(algorithms)

if __name__ == '__main__':
    app.run(debug=True)