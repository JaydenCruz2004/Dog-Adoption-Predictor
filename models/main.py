## Dog Adoption Project
## Belmont University
## Jayden Cruz
## DSC-4900-01: Data Science Project/Portfolio (Spring 2026)

# main.py
# Entry point — run this file to execute the full pipeline
# Imports from config.py, features.py, and model.py

from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, TEST_PATH, OUTPUT_PATH, FEATURES, TARGET
from features import load_data, add_adoption_score, add_sentiment_features
from model import run_grid_search, train_model, evaluate_model, print_feature_importance, save_submission


def main():
    # 1. Load data — filters to dogs only
    train, test = load_data(TRAIN_PATH, TEST_PATH)

    # 2. Convert AdoptionSpeed (0-4) to a 0-1 adoption score
    train = add_adoption_score(train)

    # 3. Add sentiment features from the Description column to both train and test
    print("\nRunning sentiment analysis...")
    train = add_sentiment_features(train)
    test = add_sentiment_features(test)

    # 4. Split into training and validation sets
    X = train[FEATURES]
    y = train[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTrain size:", len(X_train), "| Validation size:", len(X_val))

    # 5. Grid search — finds the best LightGBM hyperparameters
    best_params = run_grid_search(X_train, y_train)

    # 6. Train the final model using those best parameters
    model = train_model(X_train, y_train, X_val, y_val, best_params)

    # 7. Evaluate — prints MAE, RMSE, R2, ROC AUC, and sample predictions
    evaluate_model(model, X_val, y_val)

    # 8. Feature importance — shows which features mattered most
    print_feature_importance(model)

    # 9. Generate and save final predictions on the test set
    save_submission(model, test, OUTPUT_PATH)


if __name__ == "__main__":
    main()
