import os
import pandas as pd
import pickle

from clustering_recommender import ClusteringRecommender
from collaborative_filtering_based import CollaborativeFilteringRecommender
from content_based_recommender import ContentBasedRecommender

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix

DATA_PATH = 'data/uttarakhand_places.csv'
MODELS_DIR = 'models'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_clustering(df):
    print("\n🚀 Training Clustering Recommender...")
    model = ClusteringRecommender(n_clusters=5, algorithm='kmeans')
    model.preprocess_data(df)
    model.save_model(os.path.join(MODELS_DIR, 'clustering_model.pkl'))
    print("✅ Clustering model saved.")

def train_collaborative(df):
    print("\n🚀 Training Collaborative Filtering Recommender...")
    model = CollaborativeFilteringRecommender(n_components=8, algorithm='nmf')
    model.preprocess_data(df)
    model.save_model()
    print("✅ Collaborative filtering model saved.")

def train_content_based(df):
    print("\n🚀 Training Content-Based Recommender...")
    model = ContentBasedRecommender()
    model.preprocess_data(df)
    model.save_model()
    print("✅ Content-based model saved.")

def train_fallback_knn(df):
    print("\n🚀 Training Original KNN Fallback Model...")

    # Clean and encode inputs
    tag_binarizer = MultiLabelBinarizer()
    df['Type'] = df['Type'].fillna('').apply(lambda x: [t.strip().lower() for t in str(x).split(',') if t.strip()])
    tag_features = tag_binarizer.fit_transform(df['Type'])

    season_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df['Best_Season'] = df['Best_Season'].fillna('summer')
    season_features = season_encoder.fit_transform(df[['Best_Season']])

    numeric_features_raw = df[['Avg_Cost', 'Typical_Duration']].copy()
    numeric_features_raw['Avg_Cost'] = pd.to_numeric(numeric_features_raw['Avg_Cost'], errors='coerce').fillna(df['Avg_Cost'].median())
    numeric_features_raw['Typical_Duration'] = pd.to_numeric(numeric_features_raw['Typical_Duration'], errors='coerce').fillna(df['Typical_Duration'].median())
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric_features_raw)

    # Combine all
    X = hstack([
        csr_matrix(tag_features),
        csr_matrix(season_features),
        csr_matrix(numeric_features)
    ])

    # Train NearestNeighbors model
    knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn_model.fit(X)

    # Save model and encoders
    with open(os.path.join(MODELS_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(knn_model, f)
    with open(os.path.join(MODELS_DIR, 'tag_binarizer.pkl'), 'wb') as f:
        pickle.dump(tag_binarizer, f)
    with open(os.path.join(MODELS_DIR, 'season_encoder.pkl'), 'wb') as f:
        pickle.dump(season_encoder, f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ KNN fallback model and encoders saved.")

def train_all():
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    ensure_dir(MODELS_DIR)

    train_clustering(df)
    train_collaborative(df)
    train_content_based(df)
    train_fallback_knn(df)

    print("\n🎉 All models trained and saved successfully!")

if __name__ == "__main__":
    train_all()
