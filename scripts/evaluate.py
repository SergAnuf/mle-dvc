# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

def evaluate_model():
    # Read the params.yaml file
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # Load the fitted_model.pkl result
    with open('models/fitted_model.pkl', 'rb') as fd:
        pipeline = joblib.load(fd)

    # Load data
    data = pd.read_csv(r'data/initial_data.csv')

        
    Y = data[params['target_col']]
    X = data.drop([params['target_col']],axis=1)

    # Check quality on cross-validation
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        pipeline,
        X,
        Y,
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics']
    )

    # Проведите кросс-валидацию #
    cv_dict = {}
    for key, value in cv_res.items():
         cv_dict[key] = value.mean().round(2)

    # Save cross-validation results in cv_res.json
    os.makedirs('cv_results', exist_ok=True)

    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(cv_dict, fd)


if __name__ == '__main__':
	evaluate_model()
