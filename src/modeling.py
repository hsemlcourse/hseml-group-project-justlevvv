# src/modeling.py
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report
)

# ------------------- Загрузка данных -------------------
def load_processed_data(data_dir='data/processed/'):
    """Загружает обработанные CSV-файлы из data/processed/."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_data_dir = os.path.join(base_dir, data_dir)

    X_train = pd.read_csv(os.path.join(full_data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(full_data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(full_data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(full_data_dir, 'y_test.csv')).values.ravel()
    print(f"Данные загружены: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test

# ------------------- Обучение базовых моделей (старая функция) -------------------
def train_baseline_models(X_train, y_train):
    """Обучает Dummy и LogisticRegression (baseline из CP1)."""
    models = {}
    models['Dummy (most_frequent)'] = DummyClassifier(strategy='most_frequent', random_state=42).fit(X_train, y_train)
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced').fit(X_train, y_train)
    print("Baseline-модели обучены.")
    return models

# ------------------- Набор моделей для экспериментов -------------------
def get_models():
    """Возвращает словарь моделей, которые будем тестировать в CP2."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=4, eval_metric='logloss'),
        'Linear SVC': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    }
    return models

# ------------------- Оценка моделей (расширенная) -------------------
def evaluate_models_extended(models, X_test, y_test):
    """
    Оценивает модели по Accuracy, Precision, Recall, F1 (для Yes),
    ROC-AUC, PR-AUC. Возвращает DataFrame с результатами.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Yes)': prec,
            'Recall (Yes)': rec,
            'F1 (Yes)': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
        print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    return pd.DataFrame(results)

# ------------------- Подбор гиперпараметров -------------------
def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=3, scoring='f1'):
    """
    Выполняет GridSearchCV для переданной модели.
    Возвращает лучшую модель и результаты поиска.
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"Лучший F1 (CV): {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.cv_results_

# ------------------- Важность признаков -------------------
def plot_feature_importance(model, feature_names, top_n=15, title='Feature Importance'):
    """
    Рисует важность признаков для древовидных моделей (RandomForest, XGBoost).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        print("Модель не поддерживает извлечение важности.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# ------------------- Сохранение модели -------------------
def save_model(model, filename, models_dir='models/'):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, filename))
    print(f"Модель сохранена как {filename}")

# ------------------- Старый run_baseline (для совместимости) -------------------
def run_baseline():
    X_train, X_test, y_train, y_test = load_processed_data()
    models = train_baseline_models(X_train, y_train)
    results_df = evaluate_models_extended(models, X_test, y_test)
    print("\nСравнение моделей:")
    print(results_df.to_string(index=False))
    save_model(models['Logistic Regression'], 'logistic_regression_baseline.pkl')
    return results_df

if __name__ == "__main__":
    run_baseline()