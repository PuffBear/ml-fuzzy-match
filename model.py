# model.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_and_evaluate_models(df_features):
    X = df_features.drop(columns='label')
    y = df_features['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --------------------------
    # Logistic Regression
    # --------------------------
    print("\n🚀 Training: Logistic Regression")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)

    print("\n📊 Classification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, digits=3))

    print("\n🧱 Confusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test, y_pred_lr))

    # Manual loss curve
    losses = []
    for i in range(1, len(y_prob_lr)+1, max(1, len(y_prob_lr)//50)):
        partial = y_prob_lr[:i, 1]
        losses.append(log_loss(y_test[:i], partial))

    plt.plot(losses)
    plt.title("Logistic Regression – Log Loss over partial test set")
    plt.xlabel("Samples")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.show()

    # --------------------------
    # XGBoost with Early Stopping
    # --------------------------
    print("\n🚀 Training: XGBoost")

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=10,
        verbose=False
    )

    y_pred_xgb = xgb_model.predict(X_test)

    print("\n📊 Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred_xgb, digits=3))

    print("\n🧱 Confusion Matrix (XGBoost):")
    print(confusion_matrix(y_test, y_pred_xgb))

    # Plot training vs validation loss
    results = xgb_model.evals_result()
    plt.plot(results['validation_0']['logloss'], label='Train')
    plt.plot(results['validation_1']['logloss'], label='Validation')
    plt.title("XGBoost – Log Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
