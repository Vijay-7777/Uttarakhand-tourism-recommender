import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.sparse import hstack, csr_matrix
import os
import warnings
warnings.filterwarnings('ignore')

class ClusteringRecommender:
    def __init__(self, n_clusters=5, algorithm='kmeans'):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.tag_binarizer = MultiLabelBinarizer()
        self.season_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.pca = PCA(n_components=10)  # Reduce dimensionality
        self.df = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.features_scaled = None
        self.coord_scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """Preprocess data for clustering"""
        self.df = df.copy()
        
        # Process type/interests - handle various formats
        self.df['Type_processed'] = self.df['Type'].fillna('').apply(
            lambda x: [i.strip().lower() for i in str(x).split(',') if i.strip()]
        )
        
        # Fit and transform interest features
        interest_features = self.tag_binarizer.fit_transform(self.df['Type_processed'])
        
        # Process season - handle missing values
        self.df['Best_Season'] = self.df['Best_Season'].fillna('summer')
        season_features = self.season_encoder.fit_transform(self.df[['Best_Season']])
        
        # Process numeric features - handle missing values
        self.df['Avg_Cost'] = pd.to_numeric(self.df['Avg_Cost'], errors='coerce').fillna(self.df['Avg_Cost'].median())
        self.df['Typical_Duration'] = pd.to_numeric(self.df['Typical_Duration'], errors='coerce').fillna(self.df['Typical_Duration'].median())
        
        numeric_features = self.scaler.fit_transform(self.df[['Avg_Cost', 'Typical_Duration']])
        
        # Add coordinate features if available
        if 'Latitude' in self.df.columns and 'Longitude' in self.df.columns:
            # Handle missing coordinates
            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce').fillna(self.df['Latitude'].median())
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce').fillna(self.df['Longitude'].median())
            coord_features = self.coord_scaler.fit_transform(self.df[['Latitude', 'Longitude']])
        else:
            coord_features = np.zeros((len(self.df), 2))
        
        # Combine all features
        combined_features = np.hstack([
            interest_features,
            season_features,
            numeric_features,
            coord_features
        ])
        
        # Apply PCA to reduce dimensionality
        n_components = min(10, combined_features.shape[1] - 1, combined_features.shape[0] - 1)
        self.pca = PCA(n_components=max(1, n_components))
        self.features_scaled = self.pca.fit_transform(combined_features)
        
        # Apply clustering
        if self.algorithm == 'kmeans':
            # Adjust n_clusters if necessary
            actual_clusters = min(self.n_clusters, len(self.df) - 1)
            self.clustering_model = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        elif self.algorithm == 'dbscan':
            self.clustering_model = DBSCAN(eps=0.5, min_samples=2)
        elif self.algorithm == 'hierarchical':
            actual_clusters = min(self.n_clusters, len(self.df) - 1)
            self.clustering_model = AgglomerativeClustering(n_clusters=actual_clusters)
        
        # Fit clustering model
        self.cluster_labels = self.clustering_model.fit_predict(self.features_scaled)
        
        # Store cluster centers for KMeans
        if hasattr(self.clustering_model, 'cluster_centers_'):
            self.cluster_centers = self.clustering_model.cluster_centers_
        else:
            # Calculate cluster centers manually for other algorithms
            unique_labels = np.unique(self.cluster_labels)
            if -1 in unique_labels:  # Remove noise points for DBSCAN
                unique_labels = unique_labels[unique_labels != -1]
            
            self.cluster_centers = []
            for label in unique_labels:
                mask = self.cluster_labels == label
                if np.sum(mask) > 0:
                    center = np.mean(self.features_scaled[mask], axis=0)
                    self.cluster_centers.append(center)
            self.cluster_centers = np.array(self.cluster_centers)
        
        # Add cluster information to dataframe
        self.df['cluster'] = self.cluster_labels
        
        return self.features_scaled
    
    def get_user_cluster(self, user_interests, season, budget, duration):
        """Determine which cluster the user belongs to"""
        try:
            # Create user feature vector
            user_interests_processed = [[i.strip().lower() for i in user_interests]]
            user_interest_features = self.tag_binarizer.transform(user_interests_processed)
            
            user_season_features = self.season_encoder.transform([[season]])
            user_numeric_features = self.scaler.transform([[budget, duration]])
            
            # Use average coordinates as placeholder
            if hasattr(self.coord_scaler, 'scale_'):
                avg_coords = np.array([[self.df['Latitude'].mean(), self.df['Longitude'].mean()]])
                user_coord_features = self.coord_scaler.transform(avg_coords)
            else:
                user_coord_features = np.array([[0, 0]])
            
            # Combine user features
            user_features = np.hstack([
                user_interest_features[0],
                user_season_features[0],
                user_numeric_features[0],
                user_coord_features[0]
            ]).reshape(1, -1)
            
            # Apply PCA
            user_features_scaled = self.pca.transform(user_features)
            
            # Predict cluster
            if hasattr(self.clustering_model, 'predict'):
                user_cluster = self.clustering_model.predict(user_features_scaled)[0]
            else:
                # For algorithms without predict method, find closest cluster center
                if len(self.cluster_centers) > 0:
                    distances = np.linalg.norm(self.cluster_centers - user_features_scaled, axis=1)
                    user_cluster = np.argmin(distances)
                else:
                    # Fallback: find most similar existing point
                    similarities = cosine_similarity(user_features_scaled, self.features_scaled)
                    most_similar_idx = np.argmax(similarities)
                    user_cluster = self.cluster_labels[most_similar_idx]
            
            return user_cluster
        except Exception as e:
            print(f"Error in get_user_cluster: {e}")
            # Fallback to most common cluster
            return np.bincount(self.cluster_labels[self.cluster_labels >= 0]).argmax()
    
    def get_recommendations(self, user_interests, season, budget, duration, top_k=5):
        """Get cluster-based recommendations"""
        if self.clustering_model is None:
            raise ValueError("Model not trained. Call preprocess_data first.")
        
        try:
            # Get user's cluster
            user_cluster = self.get_user_cluster(user_interests, season, budget, duration)
            
            # Get all destinations in the same cluster
            cluster_destinations = self.df[self.df['cluster'] == user_cluster].copy()
            
            # Apply additional filters with flexibility
            budget_filter = cluster_destinations['Avg_Cost'] <= budget * 1.3  # 30% flexibility
            duration_filter = cluster_destinations['Typical_Duration'] <= duration + 2  # 2 days flexibility
            season_filter = cluster_destinations['Best_Season'].str.lower() == season.lower()
            
            # Apply filters progressively
            filtered_destinations = cluster_destinations[budget_filter & duration_filter & season_filter]
            
            if len(filtered_destinations) < top_k:
                # Relax season filter
                filtered_destinations = cluster_destinations[budget_filter & duration_filter]
            
            if len(filtered_destinations) < top_k:
                # Relax duration filter
                filtered_destinations = cluster_destinations[budget_filter]
            
            if len(filtered_destinations) < top_k:
                # Use all cluster destinations
                filtered_destinations = cluster_destinations
            
            # If still not enough, expand to similar clusters
            if len(filtered_destinations) < top_k:
                filtered_destinations = self.get_from_similar_clusters(
                    user_cluster, user_interests, season, budget, duration, top_k
                )
            
            # Calculate similarity scores within cluster
            if len(filtered_destinations) > 0:
                scores = self.calculate_similarity_scores(
                    filtered_destinations, user_interests, season, budget, duration
                )
                
                # Sort by similarity and get top k
                top_indices = scores.argsort()[::-1][:top_k]
                recommended_destinations = filtered_destinations.iloc[top_indices]
                recommended_indices = recommended_destinations.index.tolist()
            else:
                recommended_indices = []
            
            return recommended_indices
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            # Fallback: return random destinations
            return self.df.sample(min(top_k, len(self.df))).index.tolist()
    
    def get_from_similar_clusters(self, user_cluster, user_interests, season, budget, duration, top_k):
        """Get recommendations from similar clusters when current cluster has insufficient data"""
        try:
            if len(self.cluster_centers) == 0:
                return self.df.sample(min(top_k, len(self.df)))
            
            # Ensure user_cluster is valid
            if user_cluster >= len(self.cluster_centers):
                user_cluster = 0
            
            # Find similar clusters
            user_center = self.cluster_centers[user_cluster]
            cluster_distances = np.linalg.norm(self.cluster_centers - user_center, axis=1)
            similar_clusters = np.argsort(cluster_distances)[1:min(4, len(self.cluster_centers))]  # Top 3 similar clusters
            
            all_candidates = self.df[self.df['cluster'].isin(similar_clusters)]
            
            if len(all_candidates) == 0:
                return self.df.sample(min(top_k, len(self.df)))
            
            # Apply filters
            budget_filter = all_candidates['Avg_Cost'] <= budget * 1.3
            duration_filter = all_candidates['Typical_Duration'] <= duration + 2
            
            filtered_candidates = all_candidates[budget_filter & duration_filter]
            
            if len(filtered_candidates) == 0:
                filtered_candidates = all_candidates
            
            return filtered_candidates.head(top_k)
        except Exception as e:
            print(f"Error in get_from_similar_clusters: {e}")
            return self.df.sample(min(top_k, len(self.df)))
    
    def calculate_similarity_scores(self, destinations, user_interests, season, budget, duration):
        """Calculate similarity scores for destinations"""
        scores = np.zeros(len(destinations))
        
        for i, (_, destination) in enumerate(destinations.iterrows()):
            score = 0
            
            # Interest matching
            dest_types_str = str(destination['Type']).lower()
            dest_types = [t.strip() for t in dest_types_str.split(',')]
            
            for interest in user_interests:
                for dest_type in dest_types:
                    if interest.lower().strip() in dest_type:
                        score += 1
            
            # Season matching
            if str(destination['Best_Season']).lower() == season.lower():
                score += 2
            
            # Budget scoring (closer to budget gets higher score)
            try:
                budget_diff = abs(destination['Avg_Cost'] - budget) / max(budget, 1)
                score += max(0, 1 - budget_diff)
            except:
                score += 0.5  # Default score if budget calculation fails
            
            # Duration scoring
            try:
                duration_diff = abs(destination['Typical_Duration'] - duration) / max(duration, 1)
                score += max(0, 1 - duration_diff)
            except:
                score += 0.5  # Default score if duration calculation fails
            
            scores[i] = score
        
        return scores
    
    def get_cluster_summary(self):
        """Get summary of each cluster"""
        if self.df is None or 'cluster' not in self.df.columns:
            return None
        
        cluster_summary = {}
        unique_clusters = sorted([c for c in self.df['cluster'].unique() if c != -1])  # Skip noise points
        
        for cluster_id in unique_clusters:
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            # Get most common types and seasons
            common_types = {}
            common_seasons = {}
            
            try:
                # Process types
                all_types = []
                for types_list in cluster_data['Type_processed']:
                    all_types.extend(types_list)
                
                type_counts = pd.Series(all_types).value_counts().head(3).to_dict()
                common_types = type_counts
                
                # Process seasons
                season_counts = cluster_data['Best_Season'].value_counts().head(3).to_dict()
                common_seasons = season_counts
                
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
            
            summary = {
                'size': len(cluster_data),
                'avg_cost': float(cluster_data['Avg_Cost'].mean()),
                'avg_duration': float(cluster_data['Typical_Duration'].mean()),
                'cost_range': [float(cluster_data['Avg_Cost'].min()), float(cluster_data['Avg_Cost'].max())],
                'duration_range': [float(cluster_data['Typical_Duration'].min()), float(cluster_data['Typical_Duration'].max())],
                'common_types': common_types,
                'common_seasons': common_seasons,
                'sample_destinations': cluster_data['Name'].head(5).tolist()
            }
            cluster_summary[cluster_id] = summary
        
        return cluster_summary
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'clustering_model': self.clustering_model,
            'scaler': self.scaler,
            'tag_binarizer': self.tag_binarizer,
            'season_encoder': self.season_encoder,
            'pca': self.pca,
            'coord_scaler': self.coord_scaler,
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
            'algorithm': self.algorithm
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.clustering_model = model_data['clustering_model']
        self.scaler = model_data['scaler']
        self.tag_binarizer = model_data['tag_binarizer']
        self.season_encoder = model_data['season_encoder']
        self.pca = model_data['pca']
        self.coord_scaler = model_data['coord_scaler']
        self.cluster_centers = model_data['cluster_centers']
        self.n_clusters = model_data['n_clusters']
        self.algorithm = model_data['algorithm']
        print(f"Model loaded from {filepath}")
    
    def fit_predict(self, df, user_interests, season, budget, duration, top_k=5):
        """Convenience method to fit and get recommendations in one step"""
        self.preprocess_data(df)
        return self.get_recommendations(user_interests, season, budget, duration, top_k)
    
    def get_cluster_visualization_data(self):
        """Get data for cluster visualization"""
        if self.features_scaled is None:
            return None
        
        # Use first 2 PCA components for 2D visualization
        if self.features_scaled.shape[1] >= 2:
            viz_data = {
                'x': self.features_scaled[:, 0].tolist(),
                'y': self.features_scaled[:, 1].tolist(),
                'clusters': self.cluster_labels.tolist(),
                'names': self.df['Name'].tolist()
            }
        else:
            # If only 1 dimension, create dummy y-axis
            viz_data = {
                'x': self.features_scaled[:, 0].tolist(),
                'y': [0] * len(self.features_scaled),
                'clusters': self.cluster_labels.tolist(),
                'names': self.df['Name'].tolist()
            }
        
        return viz_data


# Example usage and testing
if __name__ == "__main__":
    # Test the clustering recommender
    try:
        # Load sample data
        df = pd.read_csv('data/uttarakhand_places.csv')
        
        # Initialize recommender
        recommender = ClusteringRecommender(n_clusters=5, algorithm='kmeans')
        
        # Fit the model
        print("Training clustering model...")
        recommender.preprocess_data(df)
        
        # Get cluster summary
        print("\n=== CLUSTER SUMMARY ===")
        summary = recommender.get_cluster_summary()
        for cluster_id, info in summary.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {info['size']} destinations")
            print(f"  Avg Cost: ₹{info['avg_cost']:.0f}")
            print(f"  Avg Duration: {info['avg_duration']:.1f} days")
            print(f"  Common Types: {list(info['common_types'].keys())[:3]}")
            print(f"  Sample Destinations: {info['sample_destinations'][:3]}")
        
        # Test recommendations
        print("\n=== TESTING RECOMMENDATIONS ===")
        user_interests = ['spiritual', 'nature']
        season = 'summer'
        budget = 10000
        duration = 3
        
        print(f"User preferences: {user_interests}, {season}, ₹{budget}, {duration} days")
        
        recommended_indices = recommender.get_recommendations(
            user_interests, season, budget, duration, top_k=5
        )
        
        print(f"\nRecommended destinations:")
        for idx in recommended_indices:
            place = df.loc[idx]
            print(f"- {place['Name']} ({place['Type']}) - ₹{place['Avg_Cost']}")
        
        # Save the model
        recommender.save_model('models/clustering_model.pkl')
        
        print("\n✅ Clustering recommender completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()