import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

class CollaborativeFilteringRecommender:
    def __init__(self, n_components=10, algorithm='nmf'):
        self.n_components = n_components
        self.algorithm = algorithm
        self.model = None
        self.scaler = MinMaxScaler()
        self.item_similarity = None
        self.df = None
        self.user_item_matrix = None
        
    def create_synthetic_user_data(self, df):
        """Create synthetic user-item interaction data for demonstration"""
        np.random.seed(42)
        
        # Create synthetic users with different preference profiles
        user_profiles = {
            'adventure_lover': {'adventure': 0.9, 'trekking': 0.8, 'ski resort': 0.7},
            'spiritual_seeker': {'spiritual': 0.9, 'pilgrimage': 0.8, 'cultural': 0.6},
            'nature_enthusiast': {'nature': 0.9, 'scenic': 0.8, 'hill station': 0.7},
            'budget_traveler': {'hill station': 0.6, 'cultural': 0.7, 'nature': 0.8},
            'luxury_traveler': {'ski resort': 0.8, 'scenic': 0.9, 'hill station': 0.7},
            'family_traveler': {'hill station': 0.8, 'nature': 0.7, 'cultural': 0.6},
            'solo_backpacker': {'trekking': 0.9, 'adventure': 0.8, 'nature': 0.7}
        }
        
        users = []
        items = []
        ratings = []
        
        # Generate user-item interactions
        for user_type, preferences in user_profiles.items():
            for idx, row in df.iterrows():
                place_type = str(row['Type']).lower()
                
                # Calculate preference score based on place type
                base_score = 0.3  # baseline interest
                for pref_type, pref_score in preferences.items():
                    if pref_type in place_type:
                        base_score = max(base_score, pref_score)
                
                # Add some randomness and noise
                rating = base_score + np.random.normal(0, 0.2)
                rating = np.clip(rating, 0.1, 1.0)  # Keep ratings between 0.1 and 1.0
                
                # Only add interactions above a threshold to make data sparse
                if rating > 0.4:
                    users.append(user_type)
                    items.append(idx)
                    ratings.append(rating)
        
        # Create DataFrame
        interactions_df = pd.DataFrame({
            'user': users,
            'item': items,
            'rating': ratings
        })
        
        return interactions_df
    
    def preprocess_data(self, df):
        """Preprocess data and create user-item matrix"""
        self.df = df.copy()
        
        # Create synthetic user interactions
        interactions_df = self.create_synthetic_user_data(df)
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user', 
            columns='item', 
            values='rating', 
            fill_value=0
        )
        
        # Scale the ratings
        scaled_matrix = self.scaler.fit_transform(self.user_item_matrix.values)
        
        # Apply matrix factorization
        if self.algorithm == 'nmf':
            self.model = NMF(n_components=self.n_components, random_state=42, max_iter=1000)
        else:  # SVD
            self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        
        # Fit the model
        self.model.fit(scaled_matrix)
        
        # Calculate item-item similarity for content-based backup
        item_features = self.model.components_.T  # Items x Components
        self.item_similarity = cosine_similarity(item_features)
        
        return scaled_matrix
    
    def get_user_profile(self, interests, season, budget, duration):
        """Create a user profile based on preferences"""
        # Create a preference vector based on the synthetic user profiles
        profile_scores = {
            'adventure_lover': 0,
            'spiritual_seeker': 0,
            'nature_enthusiast': 0,
            'budget_traveler': 0,
            'luxury_traveler': 0,
            'family_traveler': 0,
            'solo_backpacker': 0
        }
        
        # Score based on interests
        interest_mapping = {
            'adventure': ['adventure_lover', 'solo_backpacker'],
            'trekking': ['adventure_lover', 'solo_backpacker'],
            'spiritual': ['spiritual_seeker'],
            'pilgrimage': ['spiritual_seeker'],
            'nature': ['nature_enthusiast', 'family_traveler'],
            'scenic': ['nature_enthusiast', 'luxury_traveler'],
            'hill station': ['nature_enthusiast', 'family_traveler', 'luxury_traveler'],
            'cultural': ['spiritual_seeker', 'budget_traveler', 'family_traveler'],
            'ski resort': ['luxury_traveler'],
            'wildlife': ['nature_enthusiast', 'adventure_lover']
        }
        
        for interest in interests:
            interest_lower = interest.lower()
            if interest_lower in interest_mapping:
                for user_type in interest_mapping[interest_lower]:
                    profile_scores[user_type] += 1
        
        # Adjust based on budget
        if budget < 10000:
            profile_scores['budget_traveler'] += 2
        elif budget > 15000:
            profile_scores['luxury_traveler'] += 2
        else:
            profile_scores['family_traveler'] += 1
        
        # Adjust based on duration
        if duration <= 2:
            profile_scores['budget_traveler'] += 1
        elif duration >= 5:
            profile_scores['solo_backpacker'] += 1
            profile_scores['adventure_lover'] += 1
        
        # Normalize scores
        total_score = sum(profile_scores.values())
        if total_score > 0:
            profile_scores = {k: v/total_score for k, v in profile_scores.items()}
        
        return profile_scores
    
    def get_recommendations(self, user_interests, season, budget, duration, top_k=5):
        """Get collaborative filtering recommendations"""
        if self.model is None:
            raise ValueError("Model not trained. Call preprocess_data first.")
        
        # Get user profile
        user_profile = self.get_user_profile(user_interests, season, budget, duration)
        
        # Calculate weighted recommendations based on similar users
        recommendations_scores = np.zeros(len(self.df))
        
        for user_type, weight in user_profile.items():
            if user_type in self.user_item_matrix.index and weight > 0:
                user_ratings = self.user_item_matrix.loc[user_type].values
                recommendations_scores += weight * user_ratings
        
        # Apply filters
        budget_filter = self.df['Avg_Cost'] <= budget * 1.2
        duration_filter = self.df['Typical_Duration'] <= duration + 1
        season_filter = self.df['Best_Season'].str.lower() == season.lower()
        
        # Create combined filter
        valid_indices = np.where(budget_filter & duration_filter)[0]
        
        if len(valid_indices) == 0:
            valid_indices = np.where(budget_filter)[0]
        
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(self.df))
        
        # Get scores for valid items only
        valid_scores = recommendations_scores[valid_indices]
        
        # Sort and get top k
        top_local_indices = np.argsort(valid_scores)[::-1][:top_k]
        top_indices = valid_indices[top_local_indices]
        
        return top_indices
    
    def get_item_based_recommendations(self, item_idx, top_k=5):
        """Get item-based collaborative filtering recommendations"""
        if self.item_similarity is None:
            raise ValueError("Model not trained.")
        
        # Get similar items
        similar_items = np.argsort(self.item_similarity[item_idx])[::-1][1:top_k+1]
        return similar_items
    
    def save_model(self, model_dir='models'):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f'{model_dir}/collaborative_filtering_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'user_item_matrix': self.user_item_matrix,
                'item_similarity': self.item_similarity,
                'algorithm': self.algorithm,
                'n_components': self.n_components
            }, f)
    
    def load_model(self, model_dir='models'):
        """Load the trained model"""
        with open(f'{model_dir}/collaborative_filtering_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.user_item_matrix = model_data['user_item_matrix']
        self.item_similarity = model_data['item_similarity']
        self.algorithm = model_data['algorithm']
        self.n_components = model_data['n_components']

def train_collaborative_filtering_model():
    """Training script for collaborative filtering recommender"""
    # Load data
    df = pd.read_csv('data/uttarakhand_places.csv')
    
    # Train NMF model
    print("Training NMF-based collaborative filtering...")
    cf_nmf = CollaborativeFilteringRecommender(n_components=8, algorithm='nmf')
    cf_nmf.preprocess_data(df)
    cf_nmf.save_model()
    
    # Train SVD model
    print("Training SVD-based collaborative filtering...")
    cf_svd = CollaborativeFilteringRecommender(n_components=8, algorithm='svd')
    cf_svd.preprocess_data(df)
    
    # Save SVD model with different name
    os.makedirs('models', exist_ok=True)
    with open('models/collaborative_filtering_svd_model.pkl', 'wb') as f:
        pickle.dump({
            'model': cf_svd.model,
            'scaler': cf_svd.scaler,
            'user_item_matrix': cf_svd.user_item_matrix,
            'item_similarity': cf_svd.item_similarity,
            'algorithm': cf_svd.algorithm,
            'n_components': cf_svd.n_components
        }, f)
    
    print("Collaborative filtering models trained and saved successfully!")
    
    # Test the model
    test_interests = ['adventure', 'trekking']
    test_season = 'winter'
    test_budget = 12000
    test_duration = 4
    
    recommendations = cf_nmf.get_recommendations(
        test_interests, test_season, test_budget, test_duration
    )
    
    print(f"\nTest recommendations for {test_interests}, {test_season}, ₹{test_budget}, {test_duration} days:")
    for idx in recommendations:
        place = df.iloc[idx]
        print(f"- {place['Name']} ({place['Type']}) - ₹{place['Avg_Cost']}")

if __name__ == "__main__":
    train_collaborative_filtering_model()