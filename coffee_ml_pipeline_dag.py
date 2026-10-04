from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def load_data():
    import pandas as pd
    file_path = "Coffe_sales.csv"
    df = pd.read_csv(file_path)
    print(f"Датасет успешно загружен. Строк: {df.shape[0]}")
    return file_path

def feature_engineering(ti):
    import pandas as pd
    file_path = ti.xcom_pull(task_ids='load_data_task')
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

    processed_path = "Coffe_sales_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"Генерация признаков завершена. Файл сохранен в {processed_path}")
    return processed_path

def train_and_log_models(ti):
    import pandas as pd
    import os
    import tempfile
    from pathlib import Path
    import mlflow
    import mlflow.sklearn

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score

    processed_file_path = ti.xcom_pull(task_ids='feature_engineering_task')
    df = pd.read_csv(processed_file_path)

    categorical_features = ['cash_type', 'Time_of_Day', 'Weekday', 'Month_name']
    numeric_features = ['hour_of_day', 'money', 'day_of_month', 'week_of_year', 'is_weekend']

    X = df[categorical_features + numeric_features]
    y = df['coffee_name']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    temp_dir = Path(tempfile.gettempdir())
    db_path = temp_dir / "airflow_mlflow_coffee.db"
    artifacts_path = temp_dir / "airflow_mlruns_coffee"

    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    EXPERIMENT_NAME = "airflow_coffee_classification"

    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=f"file:///{artifacts_path.as_posix()}"
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_params = {"learning_rate": 0.03946199615608672, "n_estimators": 283, "max_depth": 7}

    model = GradientBoostingClassifier(**best_params, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    with mlflow.start_run(run_name="airflow_optimized_gb"):
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_f1 = cv_results['test_score'].mean()

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_params(best_params)
        mlflow.log_metrics({"cv_f1_macro": cv_f1, "test_accuracy": test_acc, "test_f1_macro": test_f1})
        mlflow.sklearn.log_model(sk_model=pipeline, name="model", serialization_format="pickle")

    print(f"Airflow DAG успешно выполнил обучение. Test F1-macro: {test_f1:.4f}")

default_args = {
    'owner': 'data_engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='coffee_sales_ml_pipeline',
    default_args=default_args,
    description='Автоматизированный ML-pipeline для классификации кофейных напитков',
    schedule_interval='@weekly',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml_pipeline', 'coffee'],
) as dag:

    task_load = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    task_fe = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=feature_engineering,
    )

    task_train = PythonOperator(
        task_id='train_and_log_task',
        python_callable=train_and_log_models,
    )

    task_load >> task_fe >> task_train