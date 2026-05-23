# src/preprocessing.py
import pandas as pd
import numpy as np
import os
import joblib                          # <-- добавили
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_raw_data(filepath):
    """Загружает сырые данные из CSV."""
    df = pd.read_csv(filepath)
    print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df

def clean_data(df):
    """Очистка: удаление ID, бинаризация Attrition, проверка пропусков и дубликатов."""
    if 'Employee_ID' in df.columns:
        df = df.drop('Employee_ID', axis=1)
        print("Столбец Employee_ID удалён")
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    assert df['Attrition'].isin([0,1]).all(), "Обнаружены неожиданные значения в Attrition!"
    if df.isnull().sum().sum() > 0:
        print("Внимание: есть пропуски! Требуется ручная обработка.")
    else:
        print("Пропусков нет.")
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"Удалено дубликатов: {n_dup}")
    else:
        print("Дубликатов нет.")
    return df

def split_and_preprocess(df, target_col='Attrition', test_size=0.2, random_state=42,
                         output_dir='data/processed/', models_dir=None):
    """
    Разделяет данные, затем масштабирует числовые и кодирует категориальные признаки.
    Подгонка (fit) ТОЛЬКО на train, чтобы избежать data leakage.
    Если models_dir передан, сохраняет трансформеры (scaler.pkl, encoder.pkl).
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Определим типы признаков до разделения
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Числовые признаки: {numeric_cols}")
    print(f"Категориальные признаки: {categorical_cols}")

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Обучающая: {X_train.shape[0]}, Тестовая: {X_test.shape[0]}")

    # 1. Масштабирование числовых (fit только на train)
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols, index=X_train.index
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test[numeric_cols]),
        columns=numeric_cols, index=X_test.index
    )

    # 2. One-Hot Encoding категориальных (fit только на train)
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_train_cat = pd.DataFrame(
        encoder.fit_transform(X_train[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index
    )
    X_test_cat = pd.DataFrame(
        encoder.transform(X_test[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index
    )

    # Соединяем обработанные части
    X_train_proc = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_proc = pd.concat([X_test_num, X_test_cat], axis=1)

    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    X_train_proc.to_csv(f'{output_dir}X_train.csv', index=False)
    X_test_proc.to_csv(f'{output_dir}X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}y_test.csv', index=False)
    print(f"Данные сохранены в {output_dir}")

    # Сохраняем трансформеры, если указан каталог
    if models_dir is not None:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(encoder, os.path.join(models_dir, 'encoder.pkl'))
        print(f"Трансформеры сохранены в {models_dir}")
        joblib.dump({'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols},
                    os.path.join(models_dir, 'feature_info.pkl'))

    return X_train_proc, X_test_proc, y_train, y_test

def preprocess_pipeline(raw_data_path='data/raw/employee_attrition_dataset_10000.csv'):
    """Полный пайплайн: загрузка -> очистка -> разделение + кодирование -> сохранение."""
    df = load_raw_data(raw_data_path)
    df_clean = clean_data(df)
    # Передаём models_dir='models/', чтобы сохранить scaler и encoder
    return split_and_preprocess(df_clean, output_dir='data/processed/', models_dir='models/')

if __name__ == "__main__":
    preprocess_pipeline()