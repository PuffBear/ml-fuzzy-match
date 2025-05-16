'''
Goal: Generate labeled pairs
We'll:
- Create positive examples: identical pairs
- Create negative examples: randomly mismatched pairs
'''

import pandas as pd
import random
from rapidfuzz import fuzz
from utils import normalize
from config import PRODUCT_CSV_PATH, NUM_NEGATIVE_PAIRS, FUZZY_SCORE_LOWER_BOUND, RANDOM_SEED

def create_training_pairs(csv_path=PRODUCT_CSV_PATH, num_neg=NUM_NEGATIVE_PAIRS, seed=RANDOM_SEED):
    """
    Creates labeled (query, product, label) pairs from a product list:
    - Positive = identical product strings
    - Negative = similar (but incorrect) product strings using fuzzy ratio

    Args:
        csv_path (str): Path to cleaned product CSV
        num_neg (int): Number of hard negatives to generate per product
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Labeled pairs
    """
    random.seed(seed)
    df = pd.read_csv(csv_path)
    products = df['concat'].dropna().unique().tolist()

    # Already normalized, but extra safety
    products = [normalize(p) for p in products if isinstance(p, str) and p.strip()]

    pairs = []

    for i in range(len(products)):
        anchor = products[i]

        # Positive pair
        pairs.append({
            'query': anchor,
            'product': anchor,
            'label': 1
        })

        # Hard negative pairs
        # Step 1: Get all other products ≠ anchor
        candidates = [p for p in products if p != anchor]

        # Step 2: Score all candidates with token_set_ratio
        scored = sorted(
            [(p, fuzz.token_set_ratio(anchor, p)) for p in candidates],
            key=lambda x: -x[1]
        )

        # Step 3: Take top `num_neg` hard but wrong matches (e.g., fuzz > 60, < 100)
        j = 0
        count = 0
        while count < num_neg and j < len(scored):
            candidate, score = scored[j]
            if score < 100 and score > FUZZY_SCORE_LOWER_BOUND:  # "close but wrong"
                pairs.append({
                    'query': anchor,
                    'product': candidate,
                    'label': 0
                })
                count += 1
            j += 1

    return pd.DataFrame(pairs)
