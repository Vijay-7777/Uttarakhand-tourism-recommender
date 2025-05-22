import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
from scipy.sparse import hstack

# Load dataset
df = pd.read_csv('data/uttarakhand_places.csv')

# Preprocessing
# 1. Interests (Type) - note: some entries have multiple types separated by '/'
df['Type'] = df['Type'].str.lower()
df['Type_list'] = df['Type'].apply(lambda x: [t.strip() for t in x.split('/')])

tag_binarizer = MultiLabelBinarizer()
type_features = tag_binarizer.fit_transform(df['Type_list'])

# 2. Best_Season - clean and OneHotEncode
df['Best_Season'] = df['Best_Season'].str.strip().str.lower()  # <-- FIX: normalize case and strip spaces

season_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
season_features = season_encoder.fit_transform(df[['Best_Season']])

# 3. Numeric features: Avg_Cost, Typical_Duration
scaler = StandardScaler()
num_features = scaler.fit_transform(df[['Avg_Cost', 'Typical_Duration']])

# Combine all features (sparse hstack)
X = hstack([type_features, season_features, num_features])

# Train KNN model
model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
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

print("Model and encoders saved successfully.")
