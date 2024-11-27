from data_processing import load_and_preprocess_data
from model_training import train_model, evaluate_model
from recommendations import generate_recommendations

def get_valid_user_id(ratings_df):
    valid_user_ids = set(ratings_df.select('user_id').distinct().rdd.flatMap(lambda x: x).collect())
    while True:
        user_input = input("Please enter a User ID to get movie recommendations (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Program exited.")
            exit()
        try:
            user_id = int(user_input)
            if user_id in valid_user_ids:
                return user_id
            else:
                print("User ID does not exist. Please enter a valid User ID.")
        except ValueError:
            print("Invalid input. Please enter an integer User ID.")

def main():
    # Load data
    ratings_df, movies_df = load_and_preprocess_data()
    ratings_df.cache()
    movies_df.cache()

    # Train model
    model, test_data = train_model(ratings_df)
    # Evaluate model
    rmse = evaluate_model(model, test_data)
    # Get a valid User ID from the user
    user_id = get_valid_user_id(ratings_df)
    # Generate recommendations
    generate_recommendations(model, movies_df, user_id)

if __name__ == "__main__":
    main()
