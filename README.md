# Steam Pre-launch Success Predictor

Учебный MLOps-проект для **PMLDL Assignment 1: Deployment**. Модель оценивает вероятность успеха ещё не выпущенной Steam-игры по её концепции и параметрам запуска.

## Что считается успехом

Бинарная целевая метка `is_successful = 1`, если у вышедшей игры одновременно:

- не менее 100 отзывов;
- не менее 80% отзывов положительные.

В модель поступают только признаки, которые можно определить до релиза: жанры, планируемые теги и категории, разработчик, издатель, стартовая цена, платформы, языки и возрастной рейтинг. `Owners`, `CCU`, отзывы и другие пост-релизные показатели не используются как признаки. Результат — статистическая оценка относительно указанного критерия, а не гарантия коммерческого результата. Краткое описание исключено после ablation-теста: без него модель показала лучшие метрики.

Датасет: [All 55,000 Games on Steam (November 2022)](https://www.kaggle.com/datasets/tristan581/all-55000-games-on-steam-november-2022), лицензия CC BY-SA 4.0.

## Архитектура

1. **Data engineering:** загрузка CSV, очистка, удаление дубликатов и ценовых выбросов, создание таргета, стратифицированное разбиение на train/test.
2. **Model engineering:** TF-IDF текста + числовые признаки, LightGBM с весом редкого класса, калибровка вероятностей и подбор порога на validation; параметры и результаты логируются в MLflow.
3. **Deployment:** FastAPI и Streamlit запускаются в отдельных Docker-контейнерах.

DVC связывает первые две стадии в воспроизводимый граф. Скрипт полного пайплайна запускает DVC и затем пересобирает/поднимает deployment.

## Быстрый запуск

Требования: Python 3.12, Docker с Compose, `make`. Другой интерпретатор можно передать как `make setup PYTHON=/path/to/python`.

```bash
make setup
make download
make pipeline
make test
make deploy
```

После запуска:

- Web UI: <http://localhost:8501>
- Swagger API: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Пример API-запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "genre": "Action;Adventure;Indie",
    "tags": "Co-op;Multiplayer;Atmospheric",
    "categories": "Online Co-op;Steam Achievements",
    "price": 19.99,
    "platforms": ["windows", "linux"],
    "languages": ["English", "Russian"],
    "required_age": 12
  }'
```

## Автоматизация каждые 5 минут

Один ручной полный запуск:

```bash
./scripts/run_pipeline.sh
```

Установка задания в пользовательский `cron`:

```bash
./scripts/install_cron.sh
crontab -l
```

Каждые пять минут скрипт проверит/скачает сырой файл, выполнит `dvc repro` и запустит актуальные API и приложение через Docker Compose. DVC не переобучает модель, если входы и параметры не изменились. Лог расписания: `logs/pipeline.log`.

> `install_cron.sh` изменяет пользовательский crontab, поэтому запускайте его только на машине, где действительно нужен постоянный автоматический пайплайн.

## MLflow и метрики

```bash
.venv/bin/mlflow ui --backend-store-uri ./mlruns --port 5000
```

UI будет доступен на <http://localhost:5000>. Последние метрики также находятся в `models/metrics.json`, а сравнение DVC доступно командой `dvc metrics show`.

Основные метрики: ROC-AUC, Average Precision, F1, precision, recall и accuracy. При дисбалансе классов ключевыми являются ROC-AUC и Average Precision.

Результат проверенного запуска на фиксированном test split: ROC-AUC `0.8789`, Average Precision `0.6189`, Brier score `0.0904`, F1 `0.5986`, recall `0.6663`. Дисбаланс учтён весом положительного класса `5.204`, вероятности калибруются sigmoid-калибровкой на трёх фолдах, а порог `0.295` выбран только на validation-части train-набора.

## Структура

```text
code/
  datasets/          # download + preparation
  models/            # training/evaluation/packaging
  deployment/
    api/             # FastAPI + Dockerfile
    app/             # Streamlit + Dockerfile
    docker-compose.yml
data/raw/            # исходный CSV (не хранится в Git)
data/processed/      # train/test (DVC outputs)
models/              # модель и metrics.json
scripts/             # полный запуск и cron installer
tests/
dvc.yaml
params.yaml
```

## Настройка эксперимента

Все пороги, пути и гиперпараметры находятся в `params.yaml`. Например, изменение определения успеха:

```yaml
data:
  min_reviews: 100
  min_positive_ratio: 0.8
```

После изменения выполните `make pipeline`. DVC пересоздаст зависимые артефакты.
