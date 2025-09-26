# Alumni Recommendation System API Documentation

## Overview

The Alumni Recommendation System API is a RESTful service designed to facilitate networking between students and alumni. It provides personalized recommendations, user management, search capabilities, and comprehensive analytics.

**Base URL**: `http://localhost:5000`  
**API Version**: 1.0.0  
**Content-Type**: `application/json`

---

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [User Management](#user-management)
3. [Recommendations](#recommendations)
4. [Search & Discovery](#search--discovery)
5. [Analytics](#analytics)
6. [Utility Endpoints](#utility-endpoints)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)

---

## 🏠 API Overview

### Get API Information

**GET** `/` or `/api`

Returns basic API information and health status.

**Response:**

```json
{
    "name": "Alumni Recommendation System API",
    "version": "1.0.0",
    "status": "healthy",
    "description": "RESTful API for alumni-student networking and recommendations",
    "total_users": 5000,
    "total_alumni": 3200,
    "total_students": 1800,
    "endpoints": {
        "users": "/api/users",
        "recommendations": "/api/users/{id}/recommendations",
        "search": "/api/users/search",
        "analytics": "/api/analytics",
        "documentation": "See API_DOCUMENTATION.md"
    }
}
```

---

## 👥 User Management

### Get User by ID

**GET** `/api/users/{user_id}`

Retrieve complete profile information for a specific user.

**Parameters:**

-   `user_id` (path, integer, required): User's unique identifier

**Response:**

```json
{
    "id": 1,
    "personal_info": {
        "full_name": "Mark Johnson",
        "first_name": "Mark",
        "last_name": "Johnson",
        "email": "mark.johnson@gmail.com",
        "phone": "+91-9429946048",
        "location": "Bhopal, Louisiana, India"
    },
    "academic_info": {
        "major": "Nursing",
        "degree": "Ph.D.",
        "graduation_year": 2015,
        "is_alumni": true
    },
    "professional_info": {
        "current_company": "Zomato",
        "job_title": "Manager",
        "years_experience": 13,
        "salary_range": "15-40 LPA",
        "is_mentor": false
    },
    "profile_details": {
        "bio": "A Nursing alumna with a Ph.D. degree...",
        "skills": "counseling, data visualization, empathy training...",
        "interests": "patient care, behavioral economics, neuropsychology..."
    },
    "social_profiles": {
        "linkedin_profile": "https://linkedin.com/in/mark-johnson-328",
        "github_profile": "https://github.com/markjohnson23"
    }
}
```

### Get All Users (Paginated)

**GET** `/api/users`

Retrieve a paginated list of users with optional filtering.

**Query Parameters:**

-   `page` (integer, default: 1): Page number
-   `limit` (integer, default: 20, max: 100): Items per page
-   `type` (string, optional): Filter by user type ('alumni' or 'student')
-   `major` (string, optional): Filter by major (partial match)
-   `is_mentor` (boolean, optional): Filter by mentor status

**Example Request:**

```
GET /api/users?page=1&limit=10&type=alumni&major=Computer&is_mentor=true
```

**Response:**

```json
{
    "users": [
        {
            "id": 1,
            "full_name": "Mark Johnson",
            "major": "Computer Science",
            "degree": "Ph.D.",
            "graduation_year": 2015,
            "is_alumni": true,
            "is_mentor": true,
            "current_company": "Google",
            "job_title": "Senior Engineer",
            "location": "Mumbai, Maharashtra, India"
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 10,
        "total_count": 150,
        "total_pages": 15,
        "has_next": true,
        "has_prev": false
    },
    "filters_applied": {
        "user_type": "alumni",
        "major": "Computer",
        "is_mentor": "true"
    }
}
```

---

## 🎯 Recommendations

### Get User Recommendations

**GET** `/api/users/{user_id}/recommendations`

Get personalized recommendations for a specific user based on cosine similarity.

**Parameters:**

-   `user_id` (path, integer, required): User's unique identifier
-   `top_k` (query, integer, default: 5, max: 50): Number of recommendations
-   `min_sim` (query, float, default: 0.7, range: 0-1): Minimum similarity threshold

**Example Request:**

```
GET /api/users/123/recommendations?top_k=10&min_sim=0.8
```

**Response:**

```json
{
    "user_id": 123,
    "recommendations": [
        {
            "id": 456,
            "major": "Computer Science",
            "interests": "AI, Machine Learning, Data Science",
            "skills": "Python, TensorFlow, AWS",
            "is_alumni": true,
            "is_mentor": true,
            "similarity_score": 0.85,
            "full_name": "Sarah Wilson",
            "current_company": "Microsoft",
            "job_title": "Senior Data Scientist",
            "location": "Bangalore, Karnataka, India",
            "linkedin_profile": "https://linkedin.com/in/sarah-wilson-123",
            "years_experience": 8
        }
    ],
    "parameters": {
        "top_k": 10,
        "min_similarity_threshold": 0.8
    },
    "count": 5
}
```

### Batch Recommendations

**POST** `/api/recommendations/batch`

Get recommendations for multiple users simultaneously.

**Request Body:**

```json
{
    "user_ids": [123, 456, 789],
    "top_k": 5,
    "min_sim": 0.7
}
```

**Response:**

```json
{
  "batch_recommendations": {
    "123": [
      {
        "id": 456,
        "similarity_score": 0.85,
        "major": "Computer Science"
      }
    ],
    "456": [...],
    "789": [...]
  },
  "parameters": {
    "top_k": 5,
    "min_sim": 0.7
  }
}
```

---

## 🔍 Search & Discovery

### Search Users

**GET** `/api/users/search`

Search users by name, skills, interests, company, or other profile information.

**Query Parameters:**

-   `q` (string, required): Search query
-   `limit` (integer, default: 20, max: 100): Maximum results

**Example Request:**

```
GET /api/users/search?q=machine%20learning&limit=15
```

**Response:**

```json
{
    "query": "machine learning",
    "results_count": 8,
    "results": [
        {
            "id": 123,
            "full_name": "John Doe",
            "major": "Data Science",
            "degree": "M.S.",
            "is_alumni": true,
            "is_mentor": true,
            "current_company": "Google",
            "job_title": "ML Engineer",
            "skills": "Python, TensorFlow, Machine Learning, Deep Learning...",
            "interests": "AI, Machine Learning, Computer Vision..."
        }
    ]
}
```

---

## 📊 Analytics

### System Overview Analytics

**GET** `/api/analytics/overview`

Get comprehensive system-wide statistics and insights.

**Response:**

```json
{
    "user_statistics": {
        "total_users": 5000,
        "total_alumni": 3200,
        "total_students": 1800,
        "total_mentors": 450,
        "alumni_percentage": 64.0,
        "mentor_percentage": 9.0
    },
    "academic_distribution": {
        "top_majors": {
            "Computer Science": 850,
            "Data Science": 620,
            "Business Administration": 480,
            "Engineering": 720
        },
        "degree_types": {
            "B.Tech": 1200,
            "M.S.": 800,
            "Ph.D.": 300,
            "MBA": 450
        }
    },
    "professional_insights": {
        "top_companies": {
            "Google": 120,
            "Microsoft": 95,
            "Amazon": 88,
            "TCS": 150
        },
        "experience_statistics": {
            "mean": 8.5,
            "std": 6.2,
            "min": 0.0,
            "max": 35.0
        }
    },
    "graduation_year_stats": {
        "mean": 2010.5,
        "min": 1970.0,
        "max": 2025.0
    }
}
```

### Major-wise Analytics

**GET** `/api/analytics/majors`

Get detailed analytics broken down by academic major.

**Response:**

```json
{
    "major_analytics": {
        "Computer Science": {
            "total_users": 850,
            "alumni_count": 680,
            "student_count": 170,
            "mentor_count": 95,
            "average_experience_years": 9.2,
            "top_companies": {
                "Google": 45,
                "Microsoft": 38,
                "Amazon": 32,
                "Apple": 28,
                "Meta": 25
            }
        },
        "Data Science": {
            "total_users": 620,
            "alumni_count": 480,
            "student_count": 140,
            "mentor_count": 65,
            "average_experience_years": 7.8,
            "top_companies": {
                "Netflix": 25,
                "Uber": 22,
                "Airbnb": 18
            }
        }
    }
}
```

---

## 🛠️ Utility Endpoints

### Get All Majors

**GET** `/api/majors`

Retrieve a list of all available academic majors.

**Response:**

```json
{
    "majors": [
        "Aerospace Engineering",
        "Biology",
        "Computer Science",
        "Data Science",
        "Electrical Engineering"
    ],
    "count": 25
}
```

### Get All Companies

**GET** `/api/companies`

Retrieve a list of all companies where users work.

**Response:**

```json
{
    "companies": ["Accenture", "Amazon", "Google", "Microsoft", "TCS"],
    "count": 150
}
```

### Health Check

**GET** `/api/health`

Check API health and data availability.

**Response:**

```json
{
    "status": "healthy",
    "timestamp": "2025-09-26T10:30:00.000Z",
    "data_loaded": {
        "main_dataset": true,
        "user_info": true,
        "embeddings": true
    }
}
```

---

## ⚠️ Error Handling

The API uses standard HTTP status codes and returns consistent error responses:

### Error Response Format

```json
{
    "error": "Descriptive error message"
}
```

### Common HTTP Status Codes

-   **200 OK**: Successful request
-   **400 Bad Request**: Invalid parameters or request format
-   **404 Not Found**: User or resource not found
-   **500 Internal Server Error**: Server-side error

### Example Error Responses

**User Not Found (404):**

```json
{
    "error": "User ID 9999 not found"
}
```

**Invalid Parameters (400):**

```json
{
    "error": "top_k must be between 1 and 50"
}
```

**Missing Required Parameter (400):**

```json
{
    "error": "Search query 'q' is required"
}
```

---

## 🚦 Rate Limiting

Currently, no rate limiting is implemented, but consider implementing rate limits in production:

-   **Recommendation endpoints**: 60 requests per minute per user
-   **Search endpoints**: 100 requests per minute per user
-   **Analytics endpoints**: 30 requests per minute per user
-   **Other endpoints**: 200 requests per minute per user

---

## 🚀 Usage Examples

### Python Example

```python
import requests

# Get user profile
response = requests.get('http://localhost:5000/api/users/123')
user = response.json()

# Get recommendations
rec_response = requests.get(
    'http://localhost:5000/api/users/123/recommendations',
    params={'top_k': 10, 'min_sim': 0.8}
)
recommendations = rec_response.json()

# Search users
search_response = requests.get(
    'http://localhost:5000/api/users/search',
    params={'q': 'machine learning', 'limit': 20}
)
search_results = search_response.json()
```

### JavaScript/Node.js Example

```javascript
const axios = require("axios");

const baseURL = "http://localhost:5000/api";

// Get user profile
async function getUser(userId) {
    try {
        const response = await axios.get(`${baseURL}/users/${userId}`);
        return response.data;
    } catch (error) {
        console.error("Error:", error.response.data);
    }
}

// Get recommendations
async function getRecommendations(userId, topK = 5) {
    try {
        const response = await axios.get(
            `${baseURL}/users/${userId}/recommendations`,
            {
                params: { top_k: topK, min_sim: 0.7 },
            }
        );
        return response.data;
    } catch (error) {
        console.error("Error:", error.response.data);
    }
}
```

### cURL Examples

```bash
# Get API info
curl -X GET http://localhost:5000/api

# Get user by ID
curl -X GET http://localhost:5000/api/users/123

# Get recommendations
curl -X GET "http://localhost:5000/api/users/123/recommendations?top_k=10&min_sim=0.8"

# Search users
curl -X GET "http://localhost:5000/api/users/search?q=machine%20learning&limit=20"

# Get analytics
curl -X GET http://localhost:5000/api/analytics/overview

# Batch recommendations
curl -X POST http://localhost:5000/api/recommendations/batch \
  -H "Content-Type: application/json" \
  -d '{"user_ids": [123, 456, 789], "top_k": 5, "min_sim": 0.7}'
```

---

## 📝 Notes

1. **Performance**: Large datasets may require pagination for optimal performance
2. **Caching**: Consider implementing caching for frequently accessed data
3. **Authentication**: This API currently has no authentication - implement as needed
4. **CORS**: CORS is enabled for all origins - restrict in production
5. **Logging**: Add comprehensive logging for monitoring and debugging
6. **Validation**: Input validation is basic - enhance for production use

---

## 🔗 Related Resources

-   **Model Pipeline**: See `model_pipeline.ipynb` for recommendation algorithm details
-   **Data Sources**:
    -   `mock_data_field_specific_boolean_capped.csv` - Academic and professional data
    -   `user_additional_info.csv` - Personal and contact information
    -   `embeddings.npy` - Pre-computed user embeddings for similarity calculations

---

_This documentation is current as of September 26, 2025. For the latest updates, check the repository._
