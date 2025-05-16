'''
Goal: For each row in training_pairs.csv, well compute: -fuzz_ratio -partial_ratio -token_sort_ratio -token_set_ratio -len_diff
'''
from config import FEATURE_COLUMNS

# features.py
from thefuzz import fuzz
from utils import normalize
import pandas as pd

def extract_features(row):
    q = normalize(row['query'])
    p = normalize(row['product'])

    features = {
        'fuzz_ratio': fuzz.ratio(q, p),
        'partial_ratio': fuzz.partial_ratio(q, p),
        'token_sort_ratio': fuzz.token_sort_ratio(q, p),
        'token_set_ratio': fuzz.token_set_ratio(q, p),
        'len_diff': abs(len(q) - len(p))
    }
    # Optional sanity check
    assert set(FEATURE_COLUMNS) == set(features.keys()), "Feature mismatch!"
    
    return features

def build_feature_df(df_pairs):
    features = df_pairs.apply(extract_features, axis=1, result_type='expand')
    return pd.concat([features, df_pairs['label']], axis=1)
