import pandas as pd
import numpy as np
import lightgbm as lgb
import nltk
import json
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

model = lgb.Booster(model_file='/Users/jaydencruz/PycharmProjects/DogProject/outputs/lgb_model.txt')
sia = SentimentIntensityAnalyzer()

FEATURES = ['Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'MaturitySize',
            'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Quantity', 'Fee',
            'PhotoAmt', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound']

PET_IDS = ['5eb7443c7', 'fb5c27aa4', '7dd97ecfe', '21d7f07ba', '9c5a4b145', '6b357800a',
           'cf62b409e', '81d10dff7', '03193ad4e', '138228197', 'd3281a835', '5069fa551',
           'b21940173', 'f743529bf', '57a0b3def', '9ef159820', '603a1c912', 'db6c077d0',
           '0f225be32', '542af6f78', 'a059a19d9', 'a2c642a9e', 'b08bccfc9', '23c260cc9']

gender_map = {1: 'Male', 2: 'Female', 3: 'Mixed'}
size_map = {0: 'Unknown', 1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large'}
vax_map = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
fur_map = {0: 'Unknown', 1: 'Short', 2: 'Medium', 3: 'Long'}
steril_map = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
deworm_map = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
health_map = {0: 'Not Specified', 1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury'}

test = pd.read_csv('/Users/jaydencruz/PycharmProjects/DogProject/data/test.csv')
test = test[test['PetID'].isin(PET_IDS)].copy()

scores = test['Description'].fillna('').apply(lambda t: sia.polarity_scores(t))
test['sentiment_neg'] = scores.apply(lambda s: s['neg'])
test['sentiment_neu'] = scores.apply(lambda s: s['neu'])
test['sentiment_pos'] = scores.apply(lambda s: s['pos'])
test['sentiment_compound'] = scores.apply(lambda s: s['compound'])

preds = np.clip(model.predict(test[FEATURES]), 0, 1)
test['AdoptionScore'] = preds.round(3)

dogs = []
for _, row in test.iterrows():
    dogs.append({
        'id': row['PetID'],
        'name': row['Name'] if pd.notna(row['Name']) else 'No Name',
        'age': int(row['Age']),
        'gender': gender_map.get(row['Gender'], 'Unknown'),
        'size': size_map.get(row['MaturitySize'], 'Unknown'),
        'fur': fur_map.get(row['FurLength'], 'Unknown'),
        'vax': vax_map.get(row['Vaccinated'], 'Unknown'),
        'sterilized': steril_map.get(row['Sterilized'], 'Unknown'),
        'dewormed': deworm_map.get(row['Dewormed'], 'Unknown'),
        'health': health_map.get(row['Health'], 'Unknown'),
        'fee': int(row['Fee']),
        'photos': int(row['PhotoAmt']),
        'description': row['Description'] if pd.notna(row['Description']) else 'No description available.',
        'score': round(float(row['AdoptionScore']), 3)
    })

output_path = '/Users/jaydencruz/PycharmProjects/DogProject/web/dogs.json'
with open(output_path, 'w') as f:
    json.dump(dogs, f, indent=2)

print('Saved', len(dogs), 'dogs to', output_path)
for d in dogs:
    print(d['id'], d['name'], d['score'])
