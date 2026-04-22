# Dog Adoption Predictor

**Belmont University | DSC-4900-01: Data Science Project/Portfolio (Spring 2026)**
**Jayden Cruz**

A LightGBM regression model that scores each dog's likelihood of being adopted quickly. Scores run from 0 to 1, where higher means faster adoption. The model combines tabular pet attributes with VADER sentiment analysis on rescuer-written descriptions from the PetFinder dataset.

---

## Table of Contents

1. [Objectives](#objectives)
2. [Advanced Topics](#advanced-topics)
3. [Data](#data)
4. [Variables](#variables)
5. [Model](#model)
6. [NLP and Sentiment Analysis](#nlp-and-sentiment-analysis)
7. [Results](#results)
8. [Figures](#figures)
9. [Project Structure](#project-structure)
10. [Code Overview](#code-overview)
11. [How to Run](#how-to-run)
12. [Pipeline](#pipeline)
13. [Adoption Score](#adoption-score)
14. [Grid Search](#grid-search)
15. [Dependencies](#dependencies)

---

## Objectives

This project predicts how quickly a dog listed on PetFinder will be adopted. Rather than predicting the raw category (0 to 4), the model outputs a continuous score between 0 and 1 that is easier to interpret and compare across dogs. A score near 1.0 means the dog is likely to be adopted same-day or within the week. A score near 0.0 means the dog is unlikely to be adopted within 100 days.

The two main techniques applied are gradient boosted trees (LightGBM) and natural language processing (VADER sentiment analysis). Both are covered as advanced topics below.

---

## Advanced Topics

This project covers 5 points worth of advanced topics from the approved list.

| Topic | Points |
|-------|--------|
| Gradient Boosting (LightGBM) | 1 |
| Natural Language Processing (VADER) | 2 |
| ROC / AUC Curves | 1 |
| Cross Validation (3-fold CV in grid search) | 0.5 |
| Feature Selection / Engineering | 0.5 |
| **Total** | **5.0** |

### 1. Gradient Boosting (1 point)

LightGBM is a gradient boosting framework that builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous one. It was chosen because it trains faster than standard gradient boosting on larger datasets using histogram-based learning and leaf-wise tree growth.

The model was configured as a regressor with `objective = regression` and `metric = rmse`. Early stopping was applied with a patience of 50 rounds on a held-out validation set. The final model trained for 634 rounds.

Key parameters after tuning:

| Parameter | Value |
|-----------|-------|
| num_leaves | 50 |
| learning_rate | 0.01 |
| min_child_samples | 10 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| bagging_freq | 5 |

### 2. Natural Language Processing (2 points)

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool built for short-form text. It scores words and phrases against a pre-built lexicon that accounts for punctuation, capitalization, and modifiers like "very" or "not". No training data is required.

Each dog's `Description` field was passed through VADER to produce four scores: negativity, neutrality, positivity, and a compound score summarizing overall tone on a -1 to +1 scale. These four columns were added as features alongside the tabular data.

All four sentiment features ranked in the top seven by information gain, with `sentiment_compound` ranking second overall behind Age.

```
Feature              Importance
Age                  2305.21
sentiment_compound   1393.77
Breed1               1271.29
sentiment_neu        1207.73
sentiment_pos        1134.73
PhotoAmt              856.35
sentiment_neg         782.38
```

The tone of a rescuer's description turned out to be more predictive than most structured fields including sterilization status, fur length, and color.

### 3. ROC / AUC Curves (1 point)

ROC (Receiver Operating Characteristic) curves measure how well a model separates two classes at different decision thresholds. The AUC summarizes this into a single number: 1.0 is perfect, 0.5 is random.

The problem was framed as binary: dogs with an AdoptionScore of 0.5 or higher (adopted within a month) vs dogs below 0.5. The ROC curve was plotted on the validation set and the AUC came out to 0.7404. This means the model correctly ranks a fast-adoption dog above a slow-adoption dog about 74% of the time.

The ROC curve plot is in `analysis/roc_curve.py` and the figure is saved to `outputs/roc_curve.png`.

### 4. Cross Validation (0.5 points)

3-fold cross validation was used inside the grid search to evaluate each of the 324 parameter combinations. Each combination was trained on two thirds of the training data and evaluated on the remaining third, three times with different splits. The R2 score was averaged across the three folds to select the best combination. This prevents selecting parameters that work well on one particular split by chance.

Best cross-validation R2: 0.1874 across 972 total model fits.

### 5. Feature Selection / Engineering (0.5 points)

Two types of feature work were done. Four new features were engineered from raw text using VADER, turning an unstructured description string into four numeric columns the model can use directly. Feature selection was then applied after an initial run: `Health` and `VideoAmt` were dropped after showing near-zero importance scores, reducing noise without affecting performance.

---

## Data

The dataset comes from the [PetFinder Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction/overview) competition on Kaggle. The data is subject to Kaggle competition rules and cannot be redistributed, so it is not included in this repository.

To get the data:

1. Go to https://www.kaggle.com/competitions/petfinder-adoption-prediction/overview
2. Accept the competition rules
3. Download `train.csv` and `test.csv` from the Data tab
4. Place both files in the `data/` folder

The training set contains 8,132 dog listings after filtering to dogs only (Type == 1). The test set contains 2,100 dogs.

---

## Variables

These are the features used in the final model. `Health` and `VideoAmt` were dropped after showing near-zero feature importance in early experiments.

| Feature | Description |
|---------|-------------|
| Age | Age of the dog in months when listed |
| Breed1 | Primary breed ID |
| Breed2 | Secondary breed ID (mixed breeds) |
| Gender | 1 = Male, 2 = Female, 3 = Mixed |
| Color1 | Primary color ID |
| Color2 | Secondary color ID |
| MaturitySize | Size at maturity: 1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large |
| FurLength | 1 = Short, 2 = Medium, 3 = Long |
| Vaccinated | 1 = Yes, 2 = No, 3 = Not Sure |
| Dewormed | 1 = Yes, 2 = No, 3 = Not Sure |
| Sterilized | 1 = Yes, 2 = No, 3 = Not Sure |
| Quantity | Number of pets in the listing |
| Fee | Adoption fee in local currency (0 = free) |
| PhotoAmt | Number of photos uploaded for the listing |
| sentiment_neg | VADER negativity score from the description (0 to 1) |
| sentiment_neu | VADER neutrality score from the description (0 to 1) |
| sentiment_pos | VADER positivity score from the description (0 to 1) |
| sentiment_compound | Overall VADER sentiment score (-1 to +1) |

**Target variable:** `AdoptionScore` — a continuous 0 to 1 score derived from the original `AdoptionSpeed` column (see Adoption Score section).

**Dropped features:**

| Feature | Reason |
|---------|--------|
| Health | Near-zero importance (3.39 gain vs 2305 for Age) |
| VideoAmt | Near-zero importance (4.46 gain vs 2305 for Age) |
| Type | Filtered out — only dogs (Type == 1) are included |

---

## Model

The model is a LightGBM regressor trained to predict a continuous adoption score. Regression was chosen over classification because the score is an ordered, continuous value and regression better captures the relative distance between outcomes.

Early stopping was used with a patience of 50 rounds on a held-out validation set (20% of training data, 1,627 dogs). The final model used 634 trees, compared to 97 trees before hyperparameter tuning.

---

## NLP and Sentiment Analysis

VADER was applied to the `Description` column for every dog in both the training and test sets. Missing descriptions were filled with empty strings before scoring so no rows were dropped.

The four scores produced per description:

- `sentiment_neg` — proportion of text that is negative in tone
- `sentiment_neu` — proportion of text that is neutral in tone
- `sentiment_pos` — proportion of text that is positive in tone
- `sentiment_compound` — normalized overall score from -1 (most negative) to +1 (most positive)

Example from the training data:

> "Good guard dog, very alert, active, obedience waiting for her good master, plz call or sms for more details if you really get interested, thanks!!"
> compound: 0.9538 | pos: 0.517 | neu: 0.483 | neg: 0.000

All four scores were appended to the dataframe as new columns and included in the feature set passed to LightGBM.

---

## Results

| Metric | Value |
|--------|-------|
| MAE | 0.2083 |
| RMSE | 0.2521 |
| R2 | 0.2187 |
| ROC AUC | 0.7404 |

The R2 of 0.22 is modest. Adoption speed depends on things the data does not capture, like how a dog behaves during a visit or an adopter's personal preferences on a given day. The ROC AUC of 0.74 is more useful here: when the task is framed as separating fast adoptions from slow ones, the model does a decent job.

### What actually predicts adoption speed

Age is the strongest signal by a wide margin. Younger dogs get adopted faster. Breed matters too, though it is hard to know if that is driven by the breed itself or by how common certain breeds are in the dataset.

The sentiment results are the most interesting finding. All four VADER scores rank above most tabular features. Dogs whose rescuers wrote more positive, expressive descriptions tended to get adopted faster. Whether that reflects the dog's personality, the rescuer's level of care, or just correlation with other factors is hard to separate, but the pattern held consistently.

Photo count showed up in the top six. More photos, faster adoption.

At the bottom: health status and video count barely moved the needle and were removed from the final model.

---

## Figures

All figures are saved to the `outputs/` folder and generated by code in `analysis/` folder.

**Feature Importance Plot** (`outputs/feature_importance.png`) — horizontal bar chart ranking all 18 features by LightGBM information gain. Age and sentiment_compound are the clear leaders. Shows the NLP features outperforming most tabular columns.

**AdoptionScore Distribution** (`outputs/score_distribution.png`) — histogram of the target variable across the training set. The distribution is skewed toward 0.0 (not adopted), which explains some of the model's difficulty predicting extreme values.

**ROC Curve** (`outputs/roc_curve.png`) — plots the true positive rate against the false positive rate at every decision threshold. The AUC of 0.74 is labeled on the plot. Generated by `analysis/roc_curve.py`.

**Predicted vs Actual Scores** (`outputs/predicted_vs_actual.png`) — scatter plot of validation set predictions against actual scores. The cluster toward the middle reflects the model's tendency to predict conservatively rather than committing to extreme values.

**Sentiment Score Distribution by Adoption Class** (`outputs/sentiment_by_class.png`) — box plots comparing compound sentiment scores across adoption speed categories. Faster-adopted dogs show slightly higher median compound scores.

---

## Project Structure

```
DogProject/
├── data/                  # place train.csv and test.csv here (not pushed to GitHub)
├── models/
│   ├── config.py          # all constants: paths, features, parameters
│   ├── features.py        # data loading and feature engineering
│   ├── model.py           # training, evaluation, predictions
│   └── main.py            # entry point, run this
├── analysis/
│   ├── figures.py         # generates all figures
│   └── roc_curve.py       # generates the ROC curve plot
├── outputs/               # figures and submission.csv saved here (not pushed to GitHub)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Code Overview

The project is split into four files so each one has a single job.

**`config.py`** holds every constant used across the project: file paths, the feature list, the speed-to-score mapping, and the grid search parameter grid. If you need to change a path or add a feature, this is the only file you touch.

**`features.py`** handles all data preparation. `load_data()` reads the CSVs and filters to dogs. `add_adoption_score()` maps the integer AdoptionSpeed to the 0 to 1 scale. `add_sentiment_features()` runs VADER on the Description column and adds four new columns to the dataframe.

**`model.py`** contains the ML logic. The grid search is commented out with the best parameters hardcoded in `BEST_PARAMS` at the top so the model runs immediately without waiting 45 minutes. `train_model()` fits LightGBM with early stopping. `evaluate_model()` computes MAE, RMSE, R2, and ROC AUC. `print_feature_importance()` ranks features by information gain. `save_submission()` generates test set predictions and writes the output CSV.

**`main.py`** is the only file you run. It calls each function in order and acts as a readable summary of the full pipeline.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/JaydenCruz2004/Dog-Adoption-Predictor.git
cd Dog-Adoption-Predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the data**

Follow the steps in the Data section above and place `train.csv` and `test.csv` in the `data/` folder.

**4. Run the pipeline**
```bash
python models/main.py
```

**5. Generate figures**
```bash
python analysis/figures.py
python analysis/roc_curve.py
```

All figures are saved to `outputs/`.

---

## Pipeline

`main.py` runs these steps in order:

1. Load train and test CSVs, filter to dogs only
2. Convert `AdoptionSpeed` (0 to 4) to a continuous 0 to 1 score
3. Run VADER sentiment analysis on each dog's description
4. Split into 80/20 train/validation sets
5. Train LightGBM with the best parameters from grid search
6. Evaluate on the validation set (MAE, RMSE, R2, ROC AUC)
7. Print feature importances
8. Save predictions to `submission.csv`

---

## Adoption Score

The original `AdoptionSpeed` column is an integer from 0 to 4. This project converts it to a 0 to 1 score so predictions are easier to interpret:

| AdoptionSpeed | Meaning | Score |
|---------------|---------|-------|
| 0 | Same day | 1.00 |
| 1 | Within a week | 0.75 |
| 2 | Within a month | 0.50 |
| 3 | Within 3 months | 0.25 |
| 4 | Not adopted in 100 days | 0.00 |

---

## Grid Search

The best parameters were found by running a full grid search (324 combinations, 3-fold CV, 972 total fits). That takes roughly 45 minutes so it is commented out in `model.py` and `main.py`. The winning parameters are hardcoded in `BEST_PARAMS` in `model.py`. To re-run the search, uncomment the relevant lines in both files.

---

## Dependencies

```
pandas
numpy
lightgbm
nltk
scikit-learn
matplotlib
seaborn
```