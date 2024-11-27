from pyspark.sql.functions import avg, desc

def generate_recommendations(model, movies_df, user_id):
    user_recs = model.recommendForAllUsers(10).filter(f"user_id = {user_id}")

    if user_recs.count() == 0:
        print(f"No available recommendations for User ID {user_id}.")
        generate_default_recommendations(ratings_df, movies_df)
        return

    recommendations = user_recs.collect()[0]["recommendations"]
    movie_ids = [row["movie_id"] for row in recommendations]

    recommended_movies = movies_df.filter(movies_df.movie_id.isin(movie_ids)).collect()

    print(f"\nRecommended movies for User {user_id}:")
    for movie in recommended_movies:
        print(f"- {movie['title']}")

def generate_default_recommendations(ratings_df, movies_df, num_recommendations=10):
    avg_ratings = ratings_df.groupBy("movie_id").agg(avg("rating").alias("avg_rating"))
    popular_movies = avg_ratings.join(movies_df, on="movie_id") \
        .orderBy(desc("avg_rating")) \
        .limit(num_recommendations)

    print("\nHere are some popular movies for you:")
    for row in popular_movies.collect():
        print(f"- {row['title']} (Average Rating: {row['avg_rating']:.2f})")
