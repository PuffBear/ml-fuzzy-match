# config.py

# File Paths
PRODUCT_CSV_PATH = "product_clean.csv"
MODEL_PATH = "lr_fuzzy_model.pkl"

# Fuzzy Matching Inference
TOP_K_RESULTS = 5
MIN_SCORE_THRESHOLD = 0.005  # minimum probability to accept a match

# Training Pair Generation
NUM_NEGATIVE_PAIRS = 2
FUZZY_SCORE_LOWER_BOUND = 60  # hard negative threshold (0–100 fuzz score)

# Training Settings
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Feature setting
FEATURE_COLUMNS = [
    "fuzz_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "len_diff"
]
