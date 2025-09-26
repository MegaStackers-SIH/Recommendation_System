from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import time
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Custom JSON encoder to handle numpy/pandas data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

# Load data and embeddings once at startup
try:
    df = pd.read_csv("mock_data_field_specific_boolean_capped.csv")
    df_user_info = pd.read_csv("user_additional_info.csv")
    embeddings = np.load("embeddings.npy")
    print("Data and embeddings loaded:", df.shape, embeddings.shape)
    print("Additional user info loaded:", df_user_info.shape)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Ensure 'mock_data_field_specific_boolean_capped.csv', 'user_additional_info.csv' and 'embeddings.npy' are in the project directory")
    exit(1)

def convert_to_json_serializable(obj):
    """Convert numpy/pandas data types to JSON serializable types"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

def recommend_profiles(user_id, df, embeddings, top_k=5, min_sim=0.7, diversify=True, random_seed=None, strategy='mixed'):
    """
    Recommend alumni profiles to both students and alumni based on cosine similarity.
    - Students: Recommend alumni.
    - Alumni: Recommend other alumni, excluding self.
    
    Args:
        user_id: User ID to get recommendations for
        df: User dataframe
        embeddings: User embeddings
        top_k: Number of recommendations to return
        min_sim: Minimum similarity threshold
        diversify: Whether to add diversity to recommendations
        random_seed: Random seed for reproducible results (if None, uses current time)
        strategy: Recommendation strategy ('mixed', 'similarity', 'random', 'fresh')
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
        
        # Set random seed for consistent but time-varied results
        if random_seed is None:
            # Change recommendations every 5 seconds for dynamic behavior
            random_seed = int(time.time() // 5) + user_id
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Filter by threshold
        valid_candidates = np.where(sim_scores >= min_sim)[0]
        if len(valid_candidates) == 0:
            print(f"Warning: No matches above {min_sim} for user {user_id}; using fallback")
            valid_candidates = np.arange(len(candidates_idx))
        
        # Apply different recommendation strategies
        if strategy == 'similarity':
            # Pure similarity-based (deterministic)
            sorted_valid = np.argsort(sim_scores[valid_candidates])[::-1][:min(top_k, len(valid_candidates))]
            top_candidates_local_idx = valid_candidates[sorted_valid]
            
        elif strategy == 'random':
            # Completely random from valid candidates
            if len(valid_candidates) >= top_k:
                random_picks = np.random.choice(valid_candidates, size=top_k, replace=False)
            else:
                random_picks = valid_candidates
            top_candidates_local_idx = random_picks
            
        elif strategy == 'fresh':
            # Focus on less similar but still relevant users for discovery
            sorted_indices = np.argsort(sim_scores[valid_candidates])[::-1]
            # Skip the most similar ones and pick from middle tier
            start_idx = min(top_k, len(sorted_indices) // 4)
            end_idx = min(len(sorted_indices), start_idx + top_k * 3)
            fresh_pool = sorted_indices[start_idx:end_idx]
            
            if len(fresh_pool) >= top_k:
                fresh_picks = np.random.choice(fresh_pool, size=top_k, replace=False)
            else:
                fresh_picks = fresh_pool
            top_candidates_local_idx = valid_candidates[fresh_picks]
            
        elif strategy == 'mixed' and diversify and len(valid_candidates) > top_k:
            # Enhanced mixed strategy for diversity (default)
            sorted_indices = np.argsort(sim_scores[valid_candidates])[::-1]
            
            # Take top performers (60% of recommendations)
            top_count = max(1, int(top_k * 0.6))
            top_performers = sorted_indices[:min(top_count, len(sorted_indices))]
            
            # Take good performers with some randomness (40% of recommendations)
            remaining_count = top_k - len(top_performers)
            if remaining_count > 0:
                # Select from next tier with weighted randomness
                extended_pool_size = min(len(sorted_indices), max(top_k * 3, len(top_performers) + remaining_count * 2))
                extended_pool = sorted_indices[len(top_performers):extended_pool_size]
                
                # Weighted selection - higher similarity = higher chance
                if len(extended_pool) > 0:
                    weights = sim_scores[valid_candidates[extended_pool]]
                    weights = np.maximum(weights, 0.1)  # Ensure no zero weights
                    
                    # Randomly select from extended pool
                    if len(extended_pool) >= remaining_count:
                        diverse_picks = np.random.choice(
                            extended_pool, 
                            size=remaining_count, 
                            replace=False, 
                            p=weights/weights.sum()
                        )
                    else:
                        diverse_picks = extended_pool
                else:
                    diverse_picks = []
            else:
                diverse_picks = []
            
            # Combine selections
            if len(diverse_picks) > 0:
                final_selection = np.concatenate([top_performers, diverse_picks])
            else:
                final_selection = top_performers
            
            # Shuffle the final results to add more variety
            np.random.shuffle(final_selection)
            top_candidates_local_idx = valid_candidates[final_selection[:top_k]]
        else:
            # Fallback to standard deterministic selection
            sorted_valid = np.argsort(sim_scores[valid_candidates])[::-1][:min(top_k, len(valid_candidates))]
            top_candidates_local_idx = valid_candidates[sorted_valid]
        
        top_global_idx = candidates_idx[top_candidates_local_idx]
        
        # Prepare results
        required_cols = ['id', 'major', 'interests', 'skills', 'is_alumni', 'is_mentor']
        available_cols = [col for col in required_cols if col in df.columns]
        results_df = df.loc[top_global_idx, available_cols].copy()
        results_df['similarity_score'] = sim_scores[top_candidates_local_idx]
        
        # Convert to dict and ensure JSON serializable types
        results = results_df.to_dict('records')
        return convert_to_json_serializable(results)
    
    except (IndexError, KeyError) as e:
        return {"error": f"Error processing user ID {user_id}: {str(e)}"}


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
@app.route('/api')
def api_info():
    """API information and health check endpoint"""
    return jsonify({
        "name": "Alumni Recommendation System API",
        "version": "1.0.0",
        "status": "healthy",
        "description": "RESTful API for alumni-student networking and recommendations",
        "total_users": len(df),
        "total_alumni": len(df[df['is_alumni'] == True]),
        "total_students": len(df[df['is_alumni'] == False]),
        "endpoints": {
            "users": "/api/users",
            "recommendations": "/api/users/{id}/recommendations",
            "search": "/api/users/search",
            "analytics": "/api/analytics",
            "recommendation_strategies": "/api/recommendations/strategies",
            "time_debug": "/api/recommendations/time-debug",
            "documentation": "See API_DOCUMENTATION.md"
        },
        "recommendation_features": {
            "dynamic_refresh": "every 5 seconds",
            "strategies": ["mixed", "similarity", "random", "fresh"],
            "time_based_variation": "automatic without seed parameter",
            "current_time_window": int(time.time() // 5)
        }
    })

# ============================================================================
# USER MANAGEMENT ROUTES
# ============================================================================

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user_by_id(user_id):
    """
    Fetch a user's complete profile information by their ID from both CSV files.
    Returns merged data from academic/professional info and personal details.
    """
    try:
        # Get user data from main CSV
        user_row = df[df["id"] == user_id]
        if user_row.empty:
            return jsonify({"error": f"User ID {user_id} not found"}), 404
        
        # Get additional user info
        user_info_row = df_user_info[df_user_info["id"] == user_id]
        if user_info_row.empty:
            return jsonify({"error": f"Additional user info for ID {user_id} not found"}), 404
        
        # Convert both to dictionaries
        user_data = user_row.iloc[0].to_dict()
        user_info_data = user_info_row.iloc[0].to_dict()
        
        # Merge the data (additional info takes precedence for common keys except 'id')
        merged_data = {**user_data, **user_info_data}
        
        # Organize the response in a more structured format
        structured_response = {
            "id": merged_data["id"],
            "personal_info": {
                "full_name": merged_data.get("full_name"),
                "first_name": merged_data.get("first_name"),
                "last_name": merged_data.get("last_name"),
                "email": merged_data.get("email"),
                "phone": merged_data.get("phone"),
                "location": merged_data.get("location")
            },
            "academic_info": {
                "major": merged_data.get("major"),
                "degree": merged_data.get("degree"),
                "graduation_year": merged_data.get("graduation_year"),
                "is_alumni": merged_data.get("is_alumni")
            },
            "professional_info": {
                "current_company": merged_data.get("current_company"),
                "job_title": merged_data.get("job_title"),
                "years_experience": merged_data.get("years_experience"),
                "salary_range": merged_data.get("salary_range"),
                "is_mentor": merged_data.get("is_mentor")
            },
            "profile_details": {
                "bio": merged_data.get("bio"),
                "skills": merged_data.get("skills"),
                "interests": merged_data.get("interests")
            },
            "social_profiles": {
                "linkedin_profile": merged_data.get("linkedin_profile"),
                "github_profile": merged_data.get("github_profile")
            }
        }
        
        # Convert to JSON serializable format
        structured_response = convert_to_json_serializable(structured_response)
        return jsonify(structured_response)
    
    except Exception as e:
        return jsonify({"error": f"Error fetching user data: {str(e)}"}), 500

@app.route('/api/users', methods=['GET'])
def get_all_users():
    """Get all users with pagination and filtering options"""
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        user_type = request.args.get('type')  # 'alumni' or 'student'
        major = request.args.get('major')
        is_mentor = request.args.get('is_mentor')
        
        # Validate parameters
        if page < 1:
            return jsonify({"error": "Page must be >= 1"}), 400
        if limit < 1 or limit > 100:
            return jsonify({"error": "Limit must be between 1 and 100"}), 400
        
        # Apply filters
        filtered_df = df.copy()
        
        if user_type:
            if user_type.lower() == 'alumni':
                filtered_df = filtered_df[filtered_df['is_alumni'] == True]
            elif user_type.lower() == 'student':
                filtered_df = filtered_df[filtered_df['is_alumni'] == False]
            else:
                return jsonify({"error": "Invalid user type. Use 'alumni' or 'student'"}), 400
        
        if major:
            filtered_df = filtered_df[filtered_df['major'].str.contains(major, case=False, na=False)]
        
        if is_mentor is not None:
            mentor_bool = is_mentor.lower() in ['true', '1', 'yes']
            filtered_df = filtered_df[filtered_df['is_mentor'] == mentor_bool]
        
        # Calculate pagination
        total_count = len(filtered_df)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        # Get paginated results
        paginated_df = filtered_df.iloc[start_idx:end_idx]
        
        # Basic user info (not full profile)
        users = []
        for _, user in paginated_df.iterrows():
            user_info = df_user_info[df_user_info['id'] == user['id']].iloc[0] if not df_user_info[df_user_info['id'] == user['id']].empty else {}
            users.append({
                "id": user['id'],
                "full_name": user_info.get('full_name', 'N/A'),
                "major": user['major'],
                "degree": user['degree'],
                "graduation_year": user['graduation_year'],
                "is_alumni": user['is_alumni'],
                "is_mentor": user['is_mentor'],
                "current_company": user_info.get('current_company', 'N/A'),
                "job_title": user_info.get('job_title', 'N/A'),
                "location": user_info.get('location', 'N/A')
            })
        
        response_data = {
            "users": users,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_count": total_count,
                "total_pages": (total_count + limit - 1) // limit,
                "has_next": end_idx < total_count,
                "has_prev": page > 1
            },
            "filters_applied": {
                "user_type": user_type,
                "major": major,
                "is_mentor": is_mentor
            }
        }
        
        return jsonify(convert_to_json_serializable(response_data))
    
    except Exception as e:
        return jsonify({"error": f"Error fetching users: {str(e)}"}), 500

@app.route('/api/users/search', methods=['GET'])
def search_users():
    """Search users by name, skills, interests, or company"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({"error": "Search query 'q' is required"}), 400
        
        limit = request.args.get('limit', 20, type=int)
        if limit < 1 or limit > 100:
            limit = 20
        
        # Search in multiple fields
        search_results = []
        query_lower = query.lower()
        
        for _, user in df.iterrows():
            user_info = df_user_info[df_user_info['id'] == user['id']].iloc[0] if not df_user_info[df_user_info['id'] == user['id']].empty else {}
            
            # Search in relevant fields
            searchable_text = f"{user_info.get('full_name', '')} {user['major']} {user['skills']} {user['interests']} {user_info.get('current_company', '')} {user_info.get('job_title', '')}".lower()
            
            if query_lower in searchable_text:
                search_results.append({
                    "id": user['id'],
                    "full_name": user_info.get('full_name', 'N/A'),
                    "major": user['major'],
                    "degree": user['degree'],
                    "is_alumni": user['is_alumni'],
                    "is_mentor": user['is_mentor'],
                    "current_company": user_info.get('current_company', 'N/A'),
                    "job_title": user_info.get('job_title', 'N/A'),
                    "skills": user['skills'][:100] + "..." if len(user['skills']) > 100 else user['skills'],
                    "interests": user['interests'][:100] + "..." if len(user['interests']) > 100 else user['interests']
                })
        
        # Limit results
        search_results = search_results[:limit]
        
        response_data = {
            "query": query,
            "results_count": len(search_results),
            "results": search_results
        }
        
        return jsonify(convert_to_json_serializable(response_data))
    
    except Exception as e:
        return jsonify({"error": f"Error searching users: {str(e)}"}), 500

# ============================================================================
# RECOMMENDATION ROUTES
# ============================================================================

@app.route('/api/users/<int:user_id>/recommendations', methods=['GET'])
def get_user_recommendations(user_id):
    """Get personalized recommendations for a specific user"""
    try:
        top_k = request.args.get('top_k', 5, type=int)
        min_sim = request.args.get('min_sim', 0.7, type=float)
        diversify = request.args.get('diversify', 'true').lower() in ['true', '1', 'yes']
        seed = request.args.get('seed', type=int)  # Optional seed for reproducible results
        strategy = request.args.get('strategy', 'mixed')  # mixed, similarity, random, fresh
        
        if top_k < 1 or top_k > 50:
            return jsonify({"error": "top_k must be between 1 and 50"}), 400
        if min_sim < 0 or min_sim > 1:
            return jsonify({"error": "min_sim must be between 0 and 1"}), 400
        
        # Validate strategy parameter
        valid_strategies = ['mixed', 'similarity', 'random', 'fresh']
        if strategy not in valid_strategies:
            return jsonify({"error": f"Invalid strategy. Must be one of: {valid_strategies}"}), 400
        
        results = recommend_profiles(
            user_id, df, embeddings, 
            top_k=top_k, 
            min_sim=min_sim, 
            diversify=diversify, 
            random_seed=seed,
            strategy=strategy
        )
        
        if isinstance(results, dict) and "error" in results:
            return jsonify(results), 404
        
        # Enhance results with additional user info
        enhanced_results = []
        for result in results:
            user_info = df_user_info[df_user_info['id'] == result['id']].iloc[0] if not df_user_info[df_user_info['id'] == result['id']].empty else {}
            enhanced_result = {**result}
            enhanced_result.update({
                "full_name": user_info.get('full_name', 'N/A'),
                "current_company": user_info.get('current_company', 'N/A'),
                "job_title": user_info.get('job_title', 'N/A'),
                "location": user_info.get('location', 'N/A'),
                "linkedin_profile": user_info.get('linkedin_profile', ''),
                "years_experience": user_info.get('years_experience', 0)
            })
            enhanced_results.append(enhanced_result)
        
        response_data = {
            "user_id": user_id,
            "recommendations": enhanced_results,
            "parameters": {
                "top_k": top_k,
                "min_similarity_threshold": min_sim,
                "diversify": diversify,
                "seed_used": seed if seed else "time-based",
                "strategy": strategy,
                "recommendation_approach": strategy
            },
            "count": len(enhanced_results),
            "timestamp": datetime.now().isoformat(),
            "refresh_info": {
                "recommendations_refresh": "every 5 seconds" if not seed else "static with seed",
                "next_refresh_tip": "Remove 'seed' parameter or wait 5 seconds for different results",
                "current_time_window": int(time.time() // 5)
            }
        }
        
        return jsonify(convert_to_json_serializable(response_data))
    
    except Exception as e:
        return jsonify({"error": f"Error getting recommendations: {str(e)}"}), 500

@app.route('/api/recommendations/batch', methods=['POST'])
def get_batch_recommendations():
    """Get recommendations for multiple users at once"""
    try:
        data = request.get_json()
        if not data or 'user_ids' not in data:
            return jsonify({"error": "user_ids array is required in request body"}), 400
        
        user_ids = data['user_ids']
        top_k = data.get('top_k', 5)
        min_sim = data.get('min_sim', 0.7)
        
        if not isinstance(user_ids, list) or len(user_ids) > 10:
            return jsonify({"error": "user_ids must be an array with max 10 users"}), 400
        
        batch_results = {}
        for user_id in user_ids:
            results = recommend_profiles(user_id, df, embeddings, top_k=top_k, min_sim=min_sim)
            batch_results[str(user_id)] = results
        
        response_data = {
            "batch_recommendations": batch_results,
            "parameters": {"top_k": top_k, "min_sim": min_sim}
        }
        return jsonify(convert_to_json_serializable(response_data))
    
    except Exception as e:
        return jsonify({"error": f"Error processing batch recommendations: {str(e)}"}), 500

@app.route('/api/recommendations/strategies', methods=['GET'])
def get_recommendation_strategies():
    """Get information about available recommendation strategies"""
    strategies_info = {
        "available_strategies": {
            "mixed": {
                "description": "Balanced approach combining top matches with diverse options",
                "behavior": "60% top similar users + 40% diverse weighted random selection",
                "best_for": "General discovery and networking",
                "variety": "High"
            },
            "similarity": {
                "description": "Pure similarity-based matching",
                "behavior": "Returns most similar users in descending order",
                "best_for": "Finding highly compatible matches",
                "variety": "Low (deterministic)"
            },
            "random": {
                "description": "Random selection from qualified candidates",
                "behavior": "Random selection from users above similarity threshold", 
                "best_for": "Exploring completely different profiles",
                "variety": "Very High"
            },
            "fresh": {
                "description": "Discovery-focused recommendations",
                "behavior": "Avoids most similar users, focuses on middle-tier matches",
                "best_for": "Finding new and unexpected connections",
                "variety": "High"
            }
        },
        "parameters": {
            "strategy": "One of: mixed, similarity, random, fresh (default: mixed)",
            "diversify": "Enable/disable diversification (default: true)",
            "seed": "Optional integer for reproducible results",
            "top_k": "Number of recommendations (1-50, default: 5)",
            "min_sim": "Minimum similarity threshold (0-1, default: 0.7)"
        },
        "tips": {
            "dynamic_results": "Remove 'seed' parameter for time-based variation (changes hourly)",
            "static_results": "Use 'seed' parameter for consistent results across requests",
            "variety": "Try different strategies: 'fresh' for discovery, 'mixed' for balance",
            "quality": "Use 'similarity' for highest quality matches"
        }
    }
    
    return jsonify(strategies_info)

@app.route('/api/recommendations/time-debug', methods=['GET'])
def get_time_debug_info():
    """Get current time window information for debugging dynamic recommendations"""
    current_time = time.time()
    time_window = int(current_time // 5)
    
    debug_info = {
        "current_timestamp": current_time,
        "current_datetime": datetime.now().isoformat(),
        "time_window_5_sec": time_window,
        "seconds_until_next_change": 5 - (current_time % 5),
        "explanation": "Recommendations change every 5 seconds based on time_window_5_sec value"
    }
    
    return jsonify(debug_info)

# ============================================================================
# ANALYTICS ROUTES
# ============================================================================

@app.route('/api/analytics/overview', methods=['GET'])
def get_analytics_overview():
    """Get system-wide analytics and statistics"""
    try:
        # User statistics
        total_users = len(df)
        total_alumni = len(df[df['is_alumni'] == True])
        total_students = len(df[df['is_alumni'] == False])
        total_mentors = len(df[df['is_mentor'] == True])
        
        # Major distribution
        major_counts = df['major'].value_counts().head(10).to_dict()
        
        # Degree distribution
        degree_counts = df['degree'].value_counts().to_dict()
        
        # Graduation year stats for alumni
        alumni_df = df[df['is_alumni'] == True]
        if not alumni_df.empty:
            graduation_years = alumni_df['graduation_year'].describe().to_dict()
        else:
            graduation_years = {}
        
        # Company distribution (top 10)
        company_counts = df_user_info['current_company'].value_counts().head(10).to_dict()
        
        # Experience distribution
        experience_stats = df_user_info['years_experience'].describe().to_dict()
        
        analytics_data = {
            "user_statistics": {
                "total_users": total_users,
                "total_alumni": total_alumni,
                "total_students": total_students,
                "total_mentors": total_mentors,
                "alumni_percentage": round((total_alumni / total_users) * 100, 2),
                "mentor_percentage": round((total_mentors / total_users) * 100, 2)
            },
            "academic_distribution": {
                "top_majors": major_counts,
                "degree_types": degree_counts
            },
            "professional_insights": {
                "top_companies": company_counts,
                "experience_statistics": experience_stats
            },
            "graduation_year_stats": graduation_years
        }
        
        return jsonify(convert_to_json_serializable(analytics_data))
    
    except Exception as e:
        return jsonify({"error": f"Error generating analytics: {str(e)}"}), 500

@app.route('/api/analytics/majors', methods=['GET'])
def get_major_analytics():
    """Get detailed analytics by major"""
    try:
        major_analytics = {}
        
        for major in df['major'].unique():
            major_df = df[df['major'] == major]
            major_user_info = df_user_info[df_user_info['id'].isin(major_df['id'])]
            
            alumni_count = len(major_df[major_df['is_alumni'] == True])
            student_count = len(major_df[major_df['is_alumni'] == False])
            mentor_count = len(major_df[major_df['is_mentor'] == True])
            
            # Average experience for this major
            avg_experience = major_user_info['years_experience'].mean() if not major_user_info.empty else 0
            
            # Top companies for this major
            top_companies = major_user_info['current_company'].value_counts().head(5).to_dict()
            
            major_analytics[major] = {
                "total_users": len(major_df),
                "alumni_count": alumni_count,
                "student_count": student_count,
                "mentor_count": mentor_count,
                "average_experience_years": round(avg_experience, 1),
                "top_companies": top_companies
            }
        
        return jsonify(convert_to_json_serializable({"major_analytics": major_analytics}))
    
    except Exception as e:
        return jsonify({"error": f"Error generating major analytics: {str(e)}"}), 500

# ============================================================================
# UTILITY ROUTES
# ============================================================================

@app.route('/api/majors', methods=['GET'])
def get_all_majors():
    """Get list of all available majors"""
    try:
        majors = sorted(df['major'].unique().tolist())
        return jsonify(convert_to_json_serializable({
            "majors": majors,
            "count": len(majors)
        }))
    except Exception as e:
        return jsonify({"error": f"Error fetching majors: {str(e)}"}), 500

@app.route('/api/companies', methods=['GET'])
def get_all_companies():
    """Get list of all companies"""
    try:
        companies = sorted(df_user_info['current_company'].unique().tolist())
        return jsonify(convert_to_json_serializable({
            "companies": companies,
            "count": len(companies)
        }))
    except Exception as e:
        return jsonify({"error": f"Error fetching companies: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_loaded": {
            "main_dataset": len(df) > 0,
            "user_info": len(df_user_info) > 0,
            "embeddings": embeddings is not None
        }
    }
    return jsonify(convert_to_json_serializable(health_data))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)