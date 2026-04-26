## Dog Adoption Project
## Belmont University
## Jayden Cruz
## DSC-4900-01: Data Science Project/Portfolio (Spring 2026)

# config.py
# All constants and settings for the project live here
# If you need to change a path, parameter, or feature list — this is the only file you touch

# FILE PATHS


TRAIN_PATH = "/Users/jaydencruz/PycharmProjects/DogProject/data/train.csv"
TEST_PATH = "/Users/jaydencruz/PycharmProjects/DogProject/data/test.csv"
OUTPUT_PATH = "submission.csv"

# DATA SETTINGS

DOG_TYPE = 1  # Type == 1 means dog in the dataset

# Maps the original AdoptionSpeed (0-4) to a continuous 0-1 score
# Higher score = adopted faster
SPEED_TO_SCORE = {
    0: 1.00,  # same day
    1: 0.75,  # within a week
    2: 0.50,  # within a month
    3: 0.25,  # within 3 months
    4: 0.00  # not adopted within 100 days
}

TARGET = "AdoptionScore"

# FEATURES: 18 features

FEATURES = [
    "Age",  # age in months
    "Breed1",  # primary breed ID
    "Breed2",  # secondary breed ID
    "Gender",  # 1=Male, 2=Female, 3=Mixed
    "Color1",  # primary color
    "Color2",  # secondary color
    "MaturitySize",  # 1=Small to 4=Extra Large
    "FurLength",  # 1=Short, 2=Medium, 3=Long
    "Vaccinated",  # 1=Yes, 2=No, 3=Not Sure
    "Dewormed",  # 1=Yes, 2=No, 3=Not Sure
    "Sterilized",  # 1=Yes, 2=No, 3=Not Sure
    "Quantity",  # number of pets in listing
    "Fee",  # adoption fee (0 = free)
    "PhotoAmt",  # number of photos uploaded
    "sentiment_neg",  # negativity score from description
    "sentiment_neu",  # neutrality score from description
    "sentiment_pos",  # positivity score from description
    "sentiment_compound"  # overall tone (-1 very negative to +1 very positive)
]

# GRID SEARCH PARAMETERS

# GridSearchCV will try every single combination of these values
# Current grid = 324 total combinations x 3 folds = 972 model fits
PARAM_GRID = {
    "num_leaves": [20, 31, 50],  # tree complexity
    "learning_rate": [0.01, 0.03, 0.05],  # step size per boosting round
    "min_child_samples": [10, 20, 30],  # min samples per leaf (prevents overfitting)
    "feature_fraction": [0.7, 0.8, 0.9],  # % of features used per tree
    "bagging_fraction": [0.8, 0.9],  # % of rows used per tree
    "bagging_freq": [3, 5],  # how often bagging is applied
}
