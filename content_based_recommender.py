import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import os

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        self.tag_binarizer = MultiLabelBinarizer()
        self.season_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.similarity_matrix = None
        self.df = None
        
    def preprocess_data(self, df):
        """Preprocess the dataset for content-based filtering"""
        self.df = df.copy()
        
        # Clean and preprocess text features
        self.df['Description'] = self.df['Description'].fillna('')
        self.df['combined_text'] = (
            self.df['Name'] + ' ' + 
            self.df['Type'].astype(str) + ' ' + 
            self.df['Description']
        )
        
        # TF-IDF features from combined text
        tfidf_features = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])
        
        # Process categorical features
        self.df['Type_processed'] = self.df['Type'].fillna('').apply(
            lambda x: [i.strip().lower() for i in str(x).split(',') if i.strip()]
        )
        interest_features = self.tag_binarizer.fit_transform(self.df['Type_processed'])
        
        season_features = self.season_encoder.fit_transform(self.df[['Best_Season']])
        
        # Normalize numeric features
        numeric_features = self.scaler.fit_transform(self.df[['Avg_Cost', 'Typical_Duration']])
        
        # Combine all features
        combined_features = hstack([
            tfidf_features,
            csr_matrix(interest_features),
            csr_matrix(season_features),
            csr_matrix(numeric_features)
        ])
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(combined_features)
        
        return combined_features
    
    def get_recommendations(self, user_interests, season, budget, duration, top_k=5):
        """Get recommendations based on user preferences"""
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call preprocess_data first.")
        
        # Create user profile vector
        user_text = ' '.join(user_interests) + ' ' + season
        user_tfidf = self.tfidf_vectorizer.transform([user_text])
        
        user_interests_processed = [[i.strip().lower() for i in user_interests]]
        user_interest_features = self.tag_binarizer.transform(user_interests_processed)
        
        user_season_features = self.season_encoder.transform([[season]])
        user_numeric_features = self.scaler.transform([[budget, duration]])
        
        user_profile = hstack([
            user_tfidf,
            csr_matrix(user_interest_features),
            csr_matrix(user_season_features),
            csr_matrix(user_numeric_features)
        ])
        
        # Calculate similarity with all destinations
        user_similarities = cosine_similarity(user_profile, 
                                            hstack([
                                                self.tfidf_vectorizer.transform(self.df['combined_text']),
                                                csr_matrix(self.tag_binarizer.transform(self.df['Type_processed'])),
                                                csr_matrix(self.season_encoder.transform(self.df[['Best_Season']])),
                                                csr_matrix(self.scaler.transform(self.df[['Avg_Cost', 'Typical_Duration']]))
                                            ])).flatten()
        
        # Apply budget and duration filters
        budget_filter = self.df['Avg_Cost'] <= budget * 1.2  # 20% flexibility
        duration_filter = self.df['Typical_Duration'] <= duration + 1  # 1 day flexibility
        season_filter = self.df['Best_Season'].str.lower() == season.lower()
        
        # Combine filters
        valid_indices = np.where(budget_filter & duration_filter)[0]
        
        if len(valid_indices) == 0:
            # Fallback to budget filter only
            valid_indices = np.where(budget_filter)[0]
        
        if len(valid_indices) == 0:
            # Final fallback - top similar regardless of constraints
            valid_indices = np.arange(len(self.df))
        
        # Get similarities for valid indices
        valid_similarities = user_similarities[valid_indices]
        
        # Sort by similarity and get top k
        top_indices = valid_indices[np.argsort(valid_similarities)[::-1][:top_k]]
        
        return top_indices
    
    def save_model(self, model_dir='models'):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f'{model_dir}/content_based_model.pkl', 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tag_binarizer': self.tag_binarizer,
                'season_encoder': self.season_encoder,
                'scaler': self.scaler,
                'similarity_matrix': self.similarity_matrix
            }, f)
    
    def load_model(self, model_dir='models'):
        """Load the trained model"""
        with open(f'{model_dir}/content_based_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tag_binarizer = model_data['tag_binarizer']
        self.season_encoder = model_data['season_encoder']
        self.scaler = model_data['scaler']
        self.similarity_matrix = model_data['similarity_matrix']

def train_content_based_model():
    """Training script for content-based recommender"""
    # Load data
    df = pd.read_csv('data/uttarakhand_places.csv')
    
    # Initialize and train model
    recommender = ContentBasedRecommender()
    recommender.preprocess_data(df)
    
    # Save model
    recommender.save_model()
    
    print("Content-based recommender trained and saved successfully!")
    
    # Test the model
    test_interests = ['hill station', 'nature']
    test_season = 'summer'
    test_budget = 10000
    test_duration = 3
    
    recommendations = recommender.get_recommendations(
        test_interests, test_season, test_budget, test_duration
    )
    
    print(f"\nTest recommendations for {test_interests}, {test_season}, ₹{test_budget}, {test_duration} days:")
    for idx in recommendations:
        place = df.iloc[idx]
        print(f"- {place['Name']} ({place['Type']}) - ₹{place['Avg_Cost']}")

if __name__ == "__main__":
    train_content_based_model()