## Dog Adoption Project
## Belmont University
## Jayden Cruz
## DSC-4900-01: Data Science Project/Portfolio (Spring 2026)

# model.py
# Grid search, model training, evaluation, and saving predictions
# Called by main.py after features have been built

import numpy as np
import pandas as pd
import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV  # uncomment to re-run grid search
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, roc_auc_score
from config import FEATURES
# from config import FEATURES, PARAM_GRID           # uncomment to re-run grid search

# ==============================================================================
# BEST PARAMETERS
# Found via full grid search (324 combinations x 3-fold CV = 972 fits)
# Best cross-val R2: 0.1874
# To re-run the search, uncomment run_grid_search() below and in main.py
# ==============================================================================

BEST_PARAMS = {
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "feature_fraction":  0.8,
    "learning_rate":     0.01,
    "min_child_samples": 10,
    "num_leaves":        50,
}

# ==============================================================================
# GRID SEARCH — commented out after best parameters were found
# Uncomment this function and the call in main.py to re-run
# ==============================================================================

# def run_grid_search(X_train, y_train):
#     total_combinations = 1
#     for v in PARAM_GRID.values():
#         total_combinations *= len(v)
#
#     print("\nRunning full grid search...")
#     print("Total combinations to try:", total_combinations)
#     print("(Each combination is tested with 3-fold CV — this will take several minutes)")
#
#     lgb_estimator = lgb.LGBMRegressor(
#         objective="regression",
#         n_estimators=500,
#         verbose=-1
#     )
#
#     grid_search = GridSearchCV(
#         lgb_estimator,
#         param_grid=PARAM_GRID,
#         scoring="r2",
#         cv=3,
#         n_jobs=-1,
#         verbose=1
#     )
#
#     grid_search.fit(X_train, y_train)
#
#     print("\nBest parameters found:")
#     for param, value in grid_search.best_params_.items():
#         print(" ", param, ":", value)
#     print("Best cross-val R2:", round(grid_search.best_score_, 4))
#
#     return grid_search.best_params_

# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the final LightGBM regression model using the best parameters
    found during grid search. Stops early if validation RMSE stops improving.
    Returns the trained model.
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val)

    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "verbose":           -1,
        "learning_rate":     BEST_PARAMS["learning_rate"],
        "num_leaves":        BEST_PARAMS["num_leaves"],
        "min_child_samples": BEST_PARAMS["min_child_samples"],
        "feature_fraction":  BEST_PARAMS["feature_fraction"],
        "bagging_fraction":  BEST_PARAMS["bagging_fraction"],
        "bagging_freq":      BEST_PARAMS["bagging_freq"],
    }

    print("\nTraining final model with best parameters...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1200,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    return model

# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model on the validation set.
    Reports MAE, RMSE, R2, and ROC AUC.
    Prints a sample of predictions alongside actual scores.
    Returns the validation predictions.
    """
    val_preds = np.clip(model.predict(X_val), 0, 1)

    # MAE — average absolute error between predicted and actual score
    mae = mean_absolute_error(y_val, val_preds)

    # RMSE — same as MAE but penalizes larger errors more heavily
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # R2 — proportion of variation in the score that the model explains
    r2 = r2_score(y_val, val_preds)

    # ROC AUC — converts to binary (score >= 0.5 = adopted well)
    # measures how cleanly the model separates fast vs slow adoptions
    binary_actual = (y_val >= 0.5).astype(int)
    roc_auc = roc_auc_score(binary_actual, val_preds)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print("MAE   (avg error, lower is better):           ", round(mae, 4))
    print("RMSE  (penalizes big errors, lower is better):", round(rmse, 4))
    print("R2    (variation explained, higher is better):", round(r2, 4))
    print("ROC AUC (fast vs slow separation, 1.0=best): ", round(roc_auc, 4))

    print("\nSample predictions vs actual:")
    comparison = pd.DataFrame({
        "Actual Score":    y_val.values,
        "Predicted Score": val_preds.round(3)
    })
    print(comparison.head(10).to_string(index=False))

    return val_preds

# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================

def print_feature_importance(model):
    """
    Print a ranked table of feature importances by information gain.
    Higher gain means the feature contributed more to reducing prediction error.
    """
    importance = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": model.feature_importance(importance_type="gain")
    }).sort_values("Importance", ascending=False)

    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    print(importance.to_string(index=False))

# ==============================================================================
# PREDICTION AND SUBMISSION
# ==============================================================================

def save_submission(model, test, output_path):
    """
    Generate predictions on the test set, clip to 0-1 range,
    and save a CSV with PetID and AdoptionScore columns.
    """
    X_test = test[FEATURES]
    test_scores = np.clip(model.predict(X_test), 0, 1)

    submission = pd.DataFrame({
        "PetID":         test["PetID"],
        "AdoptionScore": test_scores.round(3)
    })

    submission.to_csv(output_path, index=False)
    print("\nSubmission saved to", output_path, "-", len(submission), "rows")
    print(submission.head(10).to_string(index=False))