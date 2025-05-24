import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix

# Load data
df = pd.read_csv('data/uttarakhand_places.csv')

# Ensure required columns exist
required_columns = ['Type', 'Best_Season', 'Avg_Cost', 'Typical_Duration']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: '{col}' in dataset")

# Preprocess type (interests)
df['Type'] = df['Type'].fillna('').apply(lambda x: [i.strip().lower() for i in x.split(',') if i.strip()])
tag_binarizer = MultiLabelBinarizer()
interest_features = tag_binarizer.fit_transform(df['Type'])

# Preprocess season
season_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
season_features = season_encoder.fit_transform(df[['Best_Season']])

# Preprocess numeric values
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[['Avg_Cost', 'Typical_Duration']])

# Convert to sparse
interest_features = csr_matrix(interest_features)
season_features = csr_matrix(season_features)
numeric_features = csr_matrix(numeric_features)

# Combine all features
X = hstack([interest_features, season_features, numeric_features])

# Fit Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(X)

# Save model and encoders
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/tag_binarizer.pkl', 'wb') as f:
    pickle.dump(tag_binarizer, f)
with open('models/season_encoder.pkl', 'wb') as f:
    pickle.dump(season_encoder, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and preprocessing assets saved successfully.")
