## Supervised Learning Fuzzy Matching Model:

ml-fuzzy-match/
|
├── data.py           # Load and prepare training data (positive + negative pairs)
├── features.py       # Extract similarity features from string pairs
├── model.py          # Train and evaluate ML model
├── match.py          # Inference: take a user query and return best match
├── utils.py          # Common utilities like text normalization
├── config.py         # Centralized constants (thresholds, paths, etc.)
├── demo.ipynb        # notebook to test everything end-to-end and essentially serves as a demo for others to see how to work with this git repo.
└── README.md         # Project guide