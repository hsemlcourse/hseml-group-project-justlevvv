# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(filepath):
    """
    Загружает сырые данные из CSV-файла.
    """
    df = pd.read_csv(filepath)
    print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df

def clean_data(df):
    """
    Очистка данных:
    - Удаление ненужных столбцов (ID)
    - Преобразование целевой переменной Attrition в бинарную (1/0)
    - Проверка на пропуски и дубликаты
    """
    # Удаляем столбец Employee_ID, если он есть, т.к. он не несёт полезной информации
    if 'Employee_ID' in df.columns:
        df = df.drop('Employee_ID', axis=1)
        print("Столбец Employee_ID удалён")
    
    # Преобразуем целевую переменную: Yes -> 1, No -> 0
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Проверим, что других значений нет
    assert df['Attrition'].isin([0,1]).all(), "Обнаружены неожиданные значения в Attrition!"
    
    # Проверка на пропуски
    if df.isnull().sum().sum() > 0:
        print("Внимание: есть пропуски! Требуется ручная обработка.")
    else:
        print("Пропусков нет.")
    
    # Проверка на дубликаты
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates()
        print(f"Удалено дубликатов: {n_duplicates}")
    else:
        print("Дубликатов нет.")
    
    return df

def encode_categorical(df, target_col='Attrition'):
    """
    Применяет One-Hot Encoding к категориальным признакам.
    Для простоты (на этапе baseline) кодируем весь датасет до разделения.
    В дальнейшем это можно улучшить, чтобы избежать утечки данных.
    """
    # Определяем категориальные столбцы (все, кроме числовых и целевой переменной)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)  # целевую не кодируем
    
    print(f"Найдены категориальные столбцы: {categorical_cols}")
    
    # Применяем one-hot encoding с drop_first=True, 
    # чтобы избежать ловушки дамми-переменных (совершенной мультиколлинеарности)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"После кодирования: {df_encoded.shape[1]} столбцов")
    return df_encoded

def split_and_save(df, target_col='Attrition', test_size=0.2, random_state=42, output_dir='data/processed/'):
    """
    Разделяет данные на обучающую и тестовую выборки и сохраняет их в CSV.
    Возвращает X_train, X_test, y_train, y_test.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Обучающая выборка: {X_train.shape[0]} записей, Тестовая: {X_test.shape[0]} записей")
    
    # Сохраняем в CSV
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}y_test.csv', index=False)
    print(f"Данные сохранены в папку {output_dir}")
    
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(raw_data_path='data/raw/employee_attrition_dataset_10000.csv'):
    """
    Полный пайплайн предобработки: загрузка -> очистка -> кодирование -> разделение.
    Возвращает подготовленные выборки.
    """
    # Шаг 1: Загрузка
    df = load_raw_data(raw_data_path)
    
    # Шаг 2: Очистка
    df_clean = clean_data(df)
    
    # Шаг 3: Кодирование категориальных признаков
    df_encoded = encode_categorical(df_clean, target_col='Attrition')
    
    # Шаг 4: Разделение и сохранение
    X_train, X_test, y_train, y_test = split_and_save(df_encoded, target_col='Attrition')
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Если запустить этот файл напрямую, выполнится пайплайн
    preprocess_pipeline()