[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание увольнения сотрудника в следующие несколько месяцев

**Студент:** Грезев Лев Максимович

**Группа:** БИВ235


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Классификация

**Датасет:** Employee Attrition Prediction Dataset ([Перейти](https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset?resource=download))

**Целевая метрика:** Accuracy, F1-score


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-justlevvv.git
cd hseml-group-project-justlevvv

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить предобработку
python src/preprocessing.py

# 5. Открыть ноутбуки (notebooks/...) в Jupyter (или аналоге) и выполнить по порядку
jupyter notebook
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты CP1
Здесь коротко выпишите результаты.
| Модель | Accuracy | F1-score (Yes) | Примечание |
|--------|-------------|-------------|------------|
| Dummy (most_frequent) | 0.80 | 0.00 | Предсказывает только No |
| Лучшая модель | 0.51 | 0.28 | ... |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
