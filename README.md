<h1 align="center" id="title">ml-fuzzy-matching</h1>

  
  
<h2>📂 Files</h2> 

*   eda_cleaning.ipynb: Data loading, eda, cleaning, finalizing column.
*   utils.py
*   data.py: Load and prepare training data (positive + negative pairs).
*   features.py: Extract similarity features from string pairs.
*   model.py: Train and evaluate LR and XGBoost Model. and depending on performances save the better model.
*   evaluate.py: The cli driver for comparison
*   match.py: Inference: take a user query and return best match
*   config.py: Centralized constants (thresholds, paths, etc.).
*   demo.ipynb: notebook to test everything end-to-end and essentially serves as a demo for others to see how to work with this git repo.

  
  
<h2>💻 Built with</h2>

*   Python
*   Scikit-Learn
*   XGBoost

Correct Working Pipeline:
1. evaluate.py
2. demo.ipynb