# Movie Recommendation System

This project implements a movie recommendation system using various recommendation techniques: collaborative filtering (user-based and item-based) and content-based filtering. It includes data preprocessing, dimensionality reduction for visualization, and recommendation generation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Data Preprocessing](#data-preprocessing)
4. [Visualization](#visualization)
5. [Collaborative Filtering](#collaborative-filtering)
    - [User-Based Filtering](#user-based-filtering)
    - [Item-Based Filtering](#item-based-filtering)
6. [Content-Based Filtering](#content-based-filtering)
7. [Usage](#usage)
8. [Dependencies](#dependencies)

---

## Project Overview

This system processes movie data and ratings to recommend movies based on:
- **User-Based Collaborative Filtering**: Recommends movies by finding similar users.
- **Item-Based Collaborative Filtering**: Recommends movies by finding similar items (movies).
- **Content-Based Filtering**: Recommends movies based on their features (e.g., genres).

---

## Setup and Installation

Place your datasets in the `data/raw/` directory:
   - `ratings.csv`: User ratings for movies.
   - `movies.csv`: Movie metadata.
   - `users.csv`: User metadata (optional).

---

## Data Preprocessing

The preprocessing steps include:
1. **Genre Extraction**: Extract genres into a single column.
2. **Normalization**: Normalize numerical features using `StandardScaler`.
3. **Dimensionality Reduction**: Reduce movie features to 3 dimensions using UMAP for visualization.

Output:
- Processed genres saved in `data/processed/genres.csv`.
- Dimensionality-reduced data visualized as a 3D scatter plot saved in `artifacts/imgs/3D_movies.png`.

---

## Visualization

The UMAP-reduced features are visualized as a 3D scatter plot to understand the distribution of movies based on their features. The scatter plot is saved as `artifacts/imgs/3D_movies.png`.

---

## Collaborative Filtering

### User-Based Filtering

This approach uses cosine similarity to find users with similar preferences. Based on their ratings, it predicts movie recommendations for a target user.

- **Implementation**:
   - Train the model using Surprise's `KNNBasic`.
   - Evaluate using RMSE.

- **Function**:
   ```
   get_ubased_predictions(user_id)
   ```
   Recommends top 10 movies for a given user.

---

### Item-Based Filtering

This approach uses cosine similarity to find items (movies) similar to those rated by the target user.

- **Implementation**:
   - Train the model using Surprise's `KNNBasic`.
   - Evaluate using RMSE.

- **Function**:
   ```
   get_ibased_predictions(user_id)
   ```
   Recommends top 10 movies for a given user.

---

## Content-Based Filtering

This approach recommends movies based on their content similarity (e.g., genres).

- **Movie-to-Movie Recommendations**:
   ```
   recommend_movies(title)
   ```
   Recommends 10 movies similar to the given movie title.

- **User-Specific Recommendations**:
   ```
   recommend_for_user(user_id)
   ```
   Aggregates recommendations for movies liked by the user.

---

## Usage

1. Run the preprocessing script to generate visualizations and processed data:
   ```
   python preprocess.py
   ```

2. Use the provided functions in the script to generate recommendations:
   - **User-Based**: `get_ubased_predictions(user_id)`
   - **Item-Based**: `get_ibased_predictions(user_id)`
   - **Content-Based**:
     - `recommend_movies(title)`
     - `recommend_for_user(user_id)`

---

## Dependencies

- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `umap-learn`
  - `Surprise`

Install dependencies via:
   ```
   pip install -r requirements.txt
   ```

---

## Future Improvements

- Add hybrid recommendation methods combining collaborative and content-based approaches.
- Enhance scalability for large datasets.
- Implement a web-based interface for user interaction.