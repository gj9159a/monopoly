# Монополия (боты против ботов)

Streamlit-спектатор и детерминируемый движок Монополии. UI подключен к движку и управляет симуляцией. Все ключевые правила настраиваются через YAML-файлы.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Запуск

```bash
streamlit run app.py
```

## Тесты

```bash
pytest
python -m pytest
```

Smoke-прогон на 500 шагов:

```bash
pytest -k smoke
python -m pytest -k smoke
```

## Seed и боты

- Seed и число ботов задаются в левой панели Streamlit.
- Используется один параметризованный бот. Можно указать путь к файлу параметров (json/yaml).

## Обучение параметров (self-play, CEM)

Оценка кандидата идёт против пула оппонентов:
- baseline: `monopoly/data/params_baseline.json`
- league: `monopoly/data/league/*.json` (например `last_best.json`, `top_k.json`)

Пример запуска тренировки:

```bash
python -m monopoly.train --iters 50 --population 48 --elite 12 --games-per-cand 20 --players 6 --seed 123 --opponents mixed --baseline monopoly/data/params_baseline.json --league-dir monopoly/data/league --cand-seats rotate --checkpoint-every 5 --out trained_params.json
```

Будут созданы:
- `trained_params.json` — лучший набор параметров
- `runs/<timestamp>/best_params.json` — лучший набор (checkpoint)
- `runs/<timestamp>/mean_std.json` — параметры распределения
- `runs/<timestamp>/train_log.csv` — лог обучения

Опционально можно включить параллельную оценку кандидатов:

```bash
python -m monopoly.train --workers 4 ...
```

Запуск симуляции с сохранёнными параметрами:

```bash
python -m monopoly.sim --params trained_params.json --players 6 --seed 42 --games 1
```

## Бенчмарк параметров

```bash
python -m monopoly.bench --games 200 --players 6 --seed 123 --candidate trained_params.json --baseline monopoly/data/params_baseline.json --league-dir monopoly/data/league --opponents mixed
```

## Данные локализации и правил

Файлы для редактирования:

- `monopoly/data/board.yaml` — поле с московскими названиями и каноническими ценами/рентой/домами/ипотекой (USD).
- `monopoly/data/cards_chance.yaml` — карточки Шанс (placeholder).
- `monopoly/data/cards_community.yaml` — карточки Казна (placeholder).
- `monopoly/data/rules.yaml` — флаги правил и базовые параметры (HR1/HR2/штраф тюрьмы и т.д.).
- `monopoly/data/params_baseline.json` — baseline параметры бота.
- `monopoly/data/league/*.json` — пул лучших параметров для self-play.

## Реализованные правила

- Бросок 2d6, перемещение, проход «Старт».
- Дубли и 3 дубля подряд → тюрьма.
- Тюрьма: попытки выбросить дубль или выход по штрафу (упрощённая логика бота).
- Карточки Шанс/Казна: эффекты реализованы (тексты placeholders).
- Ипотека и выкуп с процентом; рента по ипотеке = 0.
- Застройка: дома/отели, равномерное правило и лимиты банка.
- Собственность: HR1 — при попадании на незанятую собственность всегда аукцион.
- Рента по собственности/вокзалам/коммунальным.
- HR2 — если владелец в тюрьме, рента = 0.
- Монополия: удвоение ренты на незастроенных участках при полном наборе цвета.
- Банкротство: игрок выбывает, активы переходят кредитору/банку.
- Налоги (tax клетки).
- Free Parking пустой (без накоплений).

## Допущения и TODO

- Тексты карточек — placeholders (замените в YAML).
- Ручного участия нет; сделки отключены; упрощения v1 описаны ниже.
- Численные значения в `board.yaml` канонические (USD).

## Упрощения v1

- При продаже отеля наличие свободных домов в банке не проверяется.
- При банкротстве перед банком собственность сразу уходит на аукцион.
