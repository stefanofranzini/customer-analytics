from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pandas as pd
from surprise import KNNBasic
from sklearn.preprocessing import StandardScaler
from umap import umap_
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.metrics.pairwise import cosine_similarity

#--------------------------------------------------------------------------------------
# CLEANING
#--------------------------------------------------------------------------------------

ratings = pd.read_csv('data/raw/ratings.csv')
movies  = pd.read_csv('data/raw/movies.csv')
users   = pd.read_csv('data/raw/users.csv')

movies['release_year'] = movies['release_date'].apply(lambda x: x.split('-')[-1] if not pd.isna(x) else pd.NA)

def get_genres(row):
    row_dict = dict(row)

    genres = []

    for key in row_dict:
        if key != 'movie_id' and isinstance(row_dict[key],int):
            if row_dict[key] == 1:
                genres += [key]
    
    return ','.join(genres)

movies['genres'] = movies.apply(get_genres,axis=1)

movie_genres = movies.groupby('genres').count()[['movie_id']].rename({'movie_id':'count'},axis=1).sort_values(by='count',ascending=False)

movie_genres.to_csv('data/processed/genres.csv')

id2movie = movies.set_index('movie_id')['movie_title'].to_dict()

features = movies.select_dtypes(include=['number']).drop(['movie_id'],axis=1)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)

umap_reducer = umap_.UMAP(n_components=3, random_state=42)
umap_data = umap_reducer.fit_transform(normalized_data)

movies['UMAP1'] = umap_data[:,0]
movies['UMAP2'] = umap_data[:,1]
movies['UMAP3'] = umap_data[:,2]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = movies['UMAP1']  
y = movies['UMAP2']  
z = movies['UMAP3']  

# Create the scatter plot
# scatter = ax.scatter(x, y, z, c=labels, cmap='tab20', s=20, alpha=0.8)
scatter = ax.scatter(x, y, z, s=20, alpha=0.8)

# Label axes
ax.set_title('3D Scatter Plot of Movies')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')  

# Add white outlines to all text elements (axes labels, ticks, and titles)
ax = plt.gca()
texts = [ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]  # Title, xlabel, ylabel
texts += ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels() # Tick labels

# Apply white outlines to each text object
for text in texts:
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
        path_effects.Normal()  # Normal text rendering
    ])

plt.tight_layout()

plt.savefig('artifacts/imgs/3D_movies.png', transparent=True)

plt.show()

#--------------------------------------------------------------------------------------
# USER BASED FILTERING
#--------------------------------------------------------------------------------------

reader = Reader(rating_scale=(1, 5))  # Assuming ratings range from 1 to 5
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

sim_options = {
    "name": "cosine",      # Similarity metric: cosine similarity
    "user_based": True     # Set to False for item-based filtering
}

umodel = KNNBasic(sim_options=sim_options)
umodel.fit(trainset)

predictions = umodel.test(testset)
rmse = accuracy.rmse(predictions)

def get_ubased_predictions(user_id):
    # List of all movie IDs
    all_movie_ids = ratings['movie_id'].unique()

    # Get seen movies for the user
    seen_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()

    # Predict ratings for unseen movies
    unseen_movies = [movie for movie in all_movie_ids if movie not in seen_movies]
    predictions = [umodel.predict(user_id, movie) for movie in unseen_movies]

    # Sort by predicted rating
    top_movies = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

    # Display top recommendations
    print("Top 10 Recommended Movies:")
    for pred in top_movies:
        print(f"Movie: {id2movie[pred.iid]}, Predicted Rating: {pred.est:.2f}")

#--------------------------------------------------------------------------------------
# ITEM BASED FILTERING
#--------------------------------------------------------------------------------------


sim_options = {
    "name": "cosine",      # Similarity metric: cosine similarity
    "user_based": False    # Set to False for item-based filtering
}

imodel = KNNBasic(sim_options=sim_options)
imodel.fit(trainset)

predictions = imodel.test(testset)
rmse = accuracy.rmse(predictions)

def get_ibased_predictions(user_id):
    # List of all movie IDs
    all_movie_ids = ratings['movie_id'].unique()

    # Get seen movies for the user
    user_id = 1
    seen_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()

    # Predict ratings for unseen movies
    unseen_movies = [movie for movie in all_movie_ids if movie not in seen_movies]
    predictions = [imodel.predict(user_id, movie) for movie in unseen_movies]

    # Sort by predicted rating
    top_movies = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

    # Display top recommendations
    print("Top 10 Recommended Movies:")
    for pred in top_movies:
        print(f"Movie ID: {id2movie[pred.iid]}, Predicted Rating: {pred.est:.2f}")

#--------------------------------------------------------------------------------------
# CONTENT BASED
#--------------------------------------------------------------------------------------

# compute the simalirities between movies based on genres

cosine_sim = cosine_similarity(features, features)

movie_indices = pd.Series(movies.index, index=movies['movie_title']).drop_duplicates()

def recommend_movies(title, cosine_sim=cosine_sim, movies_df=movies, movie_indices=movie_indices):
    # Get the index of the movie
    idx = movie_indices[title]

    # Get similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_df['movie_title'].iloc[movie_indices]

def recommend_for_user(user_id, cosine_sim=cosine_sim, movies_df=movies, ratings_df=ratings):
   
    # Get the movies the user has rated highly
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['movie_id']

    # Aggregate recommendations from liked movies
    recommendations = []
    for movie_id in liked_movies:
        movie_title = movies_df[movies_df['movie_id'] == movie_id]['movie_title'].values[0]
        recommendations.extend(recommend_movies(movie_title).values)

    # Return unique recommendations
    return list(set(recommendations))