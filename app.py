# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Корень проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    """Загружает модель, scaler, encoder и информацию о признаках."""

    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    encoder_path = os.path.join(BASE_DIR, 'models', 'encoder.pkl')
    model_path = os.path.join(BASE_DIR, 'models', 'best_model_cp2.pkl')
    feature_info_path = os.path.join(BASE_DIR, 'models', 'feature_info.pkl')

    for path, name in [(scaler_path, 'Scaler'), (encoder_path, 'Encoder'),
                        (model_path, 'Model'), (feature_info_path, 'Feature info')]:
        if not os.path.exists(path):
            st.error(f"Не найден файл {name}: {path}")
            st.stop()

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    model = joblib.load(model_path)
    feature_info = joblib.load(feature_info_path)
    numeric_cols = feature_info['numeric_cols']
    categorical_cols = feature_info['categorical_cols']

    # Для выпадающих списков загружаем сырые данные
    raw_path = os.path.join(BASE_DIR, 'data', 'raw', 'employee_attrition_dataset_10000.csv')
    raw = pd.read_csv(raw_path)

    # Порядок столбцов после обработки (из train)
    X_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_train.csv'))
    feature_names = X_train.columns.tolist()

    return model, scaler, encoder, numeric_cols, categorical_cols, feature_names, raw

# Загрузка
model, scaler, encoder, numeric_cols, categorical_cols, feature_names, raw_data = load_artifacts()

st.title("Предсказание увольнения сотрудника")
st.markdown("Заполните характеристики сотрудника. Прогноз основан на модели Linear SVC.")

# ---------- Форма ----------
with st.form("prediction_form"):
    st.subheader("Числовые характеристики")
    num_inputs = {}
    # Средние значения из train для подсказок
    X_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_train.csv'))
    for col in numeric_cols:
        default_val = int(X_train[col].mean()) if col in X_train.columns else 0.0
        num_inputs[col] = st.number_input(f"{col}", value=default_val)

    st.subheader("Категориальные характеристики")
    cat_inputs = {}
    for col in categorical_cols:
        unique_vals = sorted(raw_data[col].dropna().unique())
        cat_inputs[col] = st.selectbox(f"{col}", unique_vals)

    submitted = st.form_submit_button("Предсказать")

# ---------- Предсказание ----------
if submitted:
    input_data = {}
    input_data.update(num_inputs)
    input_data.update(cat_inputs)
    input_df = pd.DataFrame([input_data])

    # Применяем те же преобразования, что и при обучении
    X_num = scaler.transform(input_df[numeric_cols])
    X_cat = encoder.transform(input_df[categorical_cols])

    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))
    X_num_df = pd.DataFrame(X_num, columns=numeric_cols)
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)

    # Выравниваем по столбцам train
    X_processed = X_processed.reindex(columns=feature_names, fill_value=0)

    # Предсказание
    proba = model.predict_proba(X_processed)[0][1]
    # Используем порог 0.5 для вероятности
    pred = 1 if proba >= 0.5 else 0
    result = "Уволится (Yes)" if pred == 1 else "Останется (No)"

    st.subheader("Результат")
    col1, col2 = st.columns(2)
    col1.metric("Вероятность увольнения", f"{proba:.2%}")
    col2.metric("Прогноз", result)
    if pred == 1:
        st.warning("Сотрудник находится в группе риска.")
    else:
        st.success("Сотрудник, вероятно, останется в компании.")