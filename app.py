from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load data and embeddings once at startup
try:
    df = pd.read_csv("mock_data_field_specific_boolean_capped.csv")
    embeddings = np.load("embeddings.npy")
    print("Data and embeddings loaded:", df.shape, embeddings.shape)
except FileNotFoundError:
    print("Error: Ensure 'mock_data_field_specific_boolean_capped.csv' and 'embeddings.npy' are in the project directory")
    exit(1)

def recommend_profiles(user_id, df, embeddings, top_k=5, min_sim=0.7):
    """
    Recommend alumni profiles to both students and alumni based on cosine similarity.
    - Students: Recommend alumni.
    - Alumni: Recommend other alumni, excluding self.
    """
    try:
        user_row = df[df["id"] == user_id]
        if user_row.empty:
            return {"error": f"User ID {user_id} not found"}
        
        user_idx = user_row.index[0]
        is_user_alumni = user_row["is_alumni"].iloc[0]
        
        # Determine candidates
        alumni_mask = df["is_alumni"] == True
        if is_user_alumni:
            candidates_mask = alumni_mask & (df.index != user_idx)
        else:
            candidates_mask = alumni_mask
        
        candidates_df = df[candidates_mask]
        candidates_idx = candidates_df.index
        
        if len(candidates_idx) == 0:
            return {"error": "No alumni candidates available"}
        
        # Compute similarity
        sim_scores = cosine_similarity([embeddings[user_idx]], embeddings[candidates_idx]).flatten()
        
        # Filter by threshold
        valid_candidates = np.where(sim_scores >= min_sim)[0]
        if len(valid_candidates) == 0:
            print(f"Warning: No matches above {min_sim} for user {user_id}; using fallback")
            valid_candidates = np.arange(len(candidates_idx))
        
        # Select top_k
        sorted_valid = np.argsort(sim_scores[valid_candidates])[::-1][:min(top_k, len(valid_candidates))]
        top_candidates_local_idx = valid_candidates[sorted_valid]
        top_global_idx = candidates_idx[top_candidates_local_idx]
        
        # Prepare results
        required_cols = ['id', 'major', 'interests', 'skills', 'is_alumni', 'is_mentor']
        available_cols = [col for col in required_cols if col in df.columns]
        results_df = df.loc[top_global_idx, available_cols].copy()
        results_df['similarity_score'] = sim_scores[top_candidates_local_idx]
        
        return results_df.to_dict('records')
    
    except (IndexError, KeyError) as e:
        return {"error": f"Error processing user ID {user_id}: {str(e)}"}


@app.route('/')
@app.route('/index')
def index():
    return jsonify({"message": "Welcome to the Recommendation API"})

@app.route('/recommend_profiles/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    top_k = request.args.get('top_k', 5, type=int)
    min_sim = request.args.get('min_sim', 0.7, type=float)
    if top_k < 1:
        return jsonify({"error": "top_k must be at least 1"}), 400
    results = recommend_profiles(user_id, df, embeddings, top_k=top_k, min_sim=min_sim)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)