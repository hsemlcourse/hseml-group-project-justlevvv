# src/modeling.py
import pandas as pd
import os
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def load_processed_data(data_dir='data/processed/'):
    """
    Загружает обработанные CSV-файлы из data/processed/.
    Путь вычисляется относительно корня проекта (на один уровень выше src/).
    """
    # base_dir = корень проекта (папка, содержащая src/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_data_dir = os.path.join(base_dir, data_dir)

    X_train = pd.read_csv(os.path.join(full_data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(full_data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(full_data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(full_data_dir, 'y_test.csv')).values.ravel()
    print(f"Данные загружены: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_baseline_models(X_train, y_train):
    """
    Обучает DummyClassifier и LogisticRegression с балансировкой классов.
    Возвращает словарь с моделями.
    """
    models = {}

    # Заглушка: всегда предсказывает самый частый класс
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    models['Dummy (most_frequent)'] = dummy

    # Логистическая регрессия с балансировкой классов
    logreg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    logreg.fit(X_train, y_train)
    models['Logistic Regression'] = logreg

    print("Модели обучены.")
    return models

def evaluate_models(models, X_test, y_test):
    """
    Оценивает модели на тестовой выборке, выводит метрики.
    Возвращает DataFrame с результатами.
    """
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({'Модель': name, 'Accuracy': acc, 'F1-score': f1})
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
        print("Матрица ошибок:")
        print(confusion_matrix(y_test, y_pred))

    return pd.DataFrame(results)

def save_model(model, filename, models_dir='models/'):
    """Сохраняет модель в папку models/."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, filename))
    print(f"Модель сохранена как {filename}")

def run_baseline():
    """Полный цикл baseline: загрузка, обучение, оценка, сохранение."""
    X_train, X_test, y_train, y_test = load_processed_data()
    models = train_baseline_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)
    print("\nСравнение моделей:")
    print(results_df.to_string(index=False))

    # Сохраняем логистическую регрессию
    save_model(models['Logistic Regression'], 'logistic_regression_baseline.pkl')
    return results_df

if __name__ == "__main__":
    run_baseline()