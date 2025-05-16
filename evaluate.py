# evaluate.py

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data import create_training_pairs
from features import build_feature_df

from config import PRODUCT_CSV_PATH, MODEL_PATH

def evaluate_models(product_csv="product_clean.csv"):
    # Step 1: Generate training data
    df_pairs = create_training_pairs(product_csv, num_neg=2)
    df_features = build_feature_df(df_pairs)

    X = df_features.drop(columns='label')
    y = df_features['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }

    for name, model in models.items():
        print(f"\n🚀 {name}")

        model.fit(X_train, y_train)

        if name == "Logistic Regression":
            import joblib
            joblib.dump(model, MODEL_PATH)
            print("💾 Saved Logistic Regression model as lr_fuzzy_model.pkl")
            y_prob = model.predict_proba(X_test)
            losses = []
            for i in range(1, len(y_prob)+1, max(1, len(y_prob)//50)):
                partial = y_prob[:i, 1]
                losses.append(log_loss(y_test[:i], partial, labels=[0, 1]))

            plt.plot(losses)
            plt.title(f"{name} – Log Loss")
            plt.xlabel("Samples")
            plt.ylabel("Log Loss")
            plt.grid(True)
            plt.show()

        # Skip log loss plot for XGBoost unless you upgrade
        y_pred = model.predict(X_test)
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        print("🧱 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_models()
