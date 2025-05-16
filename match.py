'''
Essential core logic for matching
'''

import pandas as pd
import joblib
from rapidfuzz import fuzz
from features import extract_features
from utils import normalize

class FuzzyMatcher:
    def __init__(self, model_path="lr_fuzzy_model.pkl", product_csv="product_clean.csv", top_k=5):
        self.model = joblib.load(model_path)
        self.products = pd.read_csv(product_csv)['concat'].dropna().unique().tolist()
        self.top_k = top_k

    def match(self, user_query, min_score_threshold=0.005):  # 👈 configurable threshold
        user_query = normalize(user_query)
        candidates = []

        for product in self.products:
            features = extract_features({"query": user_query, "product": product})
            features_df = pd.DataFrame([features])
            prob = self.model.predict_proba(features_df)[0][1]

            # Only consider candidates with decent probability
            if prob >= min_score_threshold:
                candidates.append((product, prob))

        # Sort by probability
        ranked = sorted(candidates, key=lambda x: -x[1])

        if ranked:
            return ranked[:self.top_k]

        # 🔁 Fallback: Return top-1 raw match even if it failed threshold
        # Try all products again, this time with no threshold
        raw_candidates = []

        for product in self.products:
            features = extract_features({"query": user_query, "product": product})
            features_df = pd.DataFrame([features])
            prob = self.model.predict_proba(features_df)[0][1]
            raw_candidates.append((product, prob))

        fallback_ranked = sorted(raw_candidates, key=lambda x: -x[1])
        print("⚠️ No match passed the threshold — showing fallback:")
        return fallback_ranked[:1]

