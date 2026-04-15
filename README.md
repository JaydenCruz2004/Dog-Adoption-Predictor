# Dog Adoption Predictor

**Belmont University | DSC-4900-01: Data Science Project/Portfolio (Spring 2026)**
**Jayden Cruz**

A LightGBM regression model that scores each dog's likelihood of being adopted quickly. Scores run from 0 to 1, where higher means faster adoption. The model trains on the PetFinder dataset using tabular pet attributes and VADER sentiment analysis on the description text written by the rescuer.

---

## Data

The dataset comes from the [PetFinder Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction/overview) competition on Kaggle. The data is subject to Kaggle competition rules and cannot be redistributed, so it is not included in this repository.

To get the data:

1. Go to https://www.kaggle.com/competitions/petfinder-adoption-prediction/overview
2. Accept the competition rules
3. Download `train.csv` and `test.csv` from the Data tab
4. Place both files in the `data/` folder

---

## Results

| Metric | Score |
|--------|-------|
| MAE | 0.2083 |
| RMSE | 0.2521 |
| R2 | 0.2187 |
| ROC AUC | 0.7404 |

The R2 of 0.22 is modest, which makes sense. Adoption speed depends on things the data doesn't capture, like adopter preferences or how a dog behaves in person. The ROC AUC of 0.74 is more telling: the model does a decent job separating dogs that got adopted quickly from dogs that didn't.

### What actually predicts adoption speed

Age is the strongest signal by a wide margin. Younger dogs get adopted faster. Breed matters too, though it's hard to know if that's driven by the breed itself or just by how common certain breeds are in the dataset.

The most interesting finding is how much the description text matters. All four sentiment scores (compound, neutral, positive, negative) rank in the top seven features, above most of the tabular columns. Dogs whose rescuers wrote more positive, expressive descriptions tended to get adopted faster. Whether that reflects the dog's personality, the rescuer's effort, or just correlation with other factors is hard to say, but the signal is real.

Photo count also shows up in the top six. More photos, faster adoption.

At the bottom: health status and video count barely move the needle.

---

## Project Structure

```
DogProject/
├── data/               # place train.csv and test.csv here (not pushed to GitHub)
├── models/
│   ├── config.py       # all constants: paths, features, parameters
│   ├── features.py     # data loading and feature engineering
│   ├── model.py        # training, evaluation, predictions
│   └── main.py         # entry point, run this
├── notebooks/
│   └── sample.ipynb    # exploratory analysis and sentiment testing
├── outputs/            # submission.csv saved here (not pushed to GitHub)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/jaydencruz/dog-adoption-predictor.git
cd dog-adoption-predictor
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

`main.py` runs the full pipeline and saves predictions to `outputs/submission.csv`.

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

The best parameters were found by running a full grid search (324 combinations, 3-fold CV). That takes roughly 45 minutes so it is commented out in `model.py` and `main.py`. The winning parameters are hardcoded in `BEST_PARAMS` in `model.py`. To re-run the search, uncomment the relevant lines in both files.

---

## Dependencies

```
pandas
numpy
lightgbm
nltk
scikit-learn
```
