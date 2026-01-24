# Монополия (боты против ботов)

Streamlit-спектатор и детерминируемый движок Монополии. UI подключен к движку и управляет симуляцией. Все ключевые правила настраиваются через YAML-файлы.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Важно: размещайте репозиторий в обычной папке с правами записи (не Temp/Program Files), иначе pip/pytest могут не иметь прав на временные каталоги.

## Запуск

```bash
streamlit run app.py
```

## Как пользоваться (коротко)

1) Откройте Streamlit и выберите режим **Тренировка**. Нажмите **Старт** — обучение идёт до плато.
2) По завершении нажмите **Запустить live матч сейчас** или перейдите в **Live матч** и стартуйте 6 deep‑ботов.
3) Лучшие параметры лежат в `runs/<timestamp>/best.json` — их можно переиспользовать для новых матчей.

## Тесты

```bash
pytest
python -m pytest
```

Проверка установки из коробки:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .[dev]
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

Быстрый прогон (small):

```bash
python -m monopoly.train --iters 5 --population 16 --elite 4 --games-per-cand 6 --players 6 --seed 123 --opponents mixed --cand-seats rotate --checkpoint-every 1 --out trained_params.json
```

Полный прогон (full):

```bash
python -m monopoly.train --iters 50 --population 48 --elite 12 --games-per-cand 20 --players 6 --seed 123 --opponents mixed --cand-seats rotate --checkpoint-every 5 --out trained_params.json
```

Метрики:
- `best_fitness` — лучший найденный fitness за всё обучение (больше = лучше).
- `mean_elite`/`std_elite` — среднее/разброс лучших кандидатов в итерации.
- `eval_cache.jsonl` — кэш оценок, ускоряет повторные оценки одинаковых θ.

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

## Autotrain (до плато) + UI

В Streamlit доступны режимы **Тренировка** и **Live матч**. Тренировка запускается в фоне, прогресс берётся из `runs/<timestamp>/status.json`, лог — из `train_log.csv` и `progress.txt`.

CLI-запуск автотренинга:

```bash
python -m monopoly.autotrain run --profile deep --workers auto
```

Параметры остановки по плато:
- `--epoch-iters` (по умолчанию 10)
- `--plateau-epochs` (по умолчанию 5)
- `--eps-winrate` (по умолчанию 0.01)
- `--eps-fitness` (по умолчанию 0.02)
- `--min-progress-games` (по умолчанию 200)
- `--delta` (по умолчанию 0.05)
- `--max-hours` — опциональный предохранитель (по умолчанию нет лимита)

Содержимое `runs/<timestamp>/`:
- `status.json` — текущий статус (epoch, best win-rate, plateau и т.п.)
- `train_log.csv` — история по эпохам
- `progress.txt` — человекочитаемый лог
- `best.json` — лучшие параметры
- `last_bench.json` — последний бенчмарк
- `summary.txt` — короткий итог по запуску
- `error.log` — stderr при падении (если было)

Live матч (6 “deep planner” ботов, запись состояния для UI):

```bash
python -m monopoly.live --players 6 --params runs/<timestamp>/best.json --mode deep --workers auto --time-per-decision-sec 3.0 --horizon-turns 60 --seed 42 --out runs/<timestamp>/live_state.json
```

## Бенчмарк параметров

```bash
python -m monopoly.bench --games 200 --players 6 --seed 123 --candidate trained_params.json --baseline monopoly/data/params_baseline.json --league-dir monopoly/data/league --opponents mixed
```

Можно использовать фиксированный набор сидов:

```bash
python -m monopoly.bench --games 200 --seeds-file monopoly/data/seeds.txt --candidate trained_params.json --baseline monopoly/data/params_baseline.json --league-dir monopoly/data/league --opponents mixed
```

## Лига и прогресс

Добавление лучшей конфигурации в лигу:

```bash
python -m monopoly.league add --params trained_params.json --name best_YYYYMMDD_HHMM --meta "iter=50; fitness=..." --fitness 0.123
```

Список лиги и очистка старых:

```bash
python -m monopoly.league list
python -m monopoly.league prune --keep 20
```

Оценка прогресса (последние 5 из лиги vs baseline и mix):

```bash
python -m monopoly.progress --league-dir monopoly/data/league --baseline monopoly/data/params_baseline.json --games 200 --seed 123
```

С фиксированным seed pack:

```bash
python -m monopoly.progress --league-dir monopoly/data/league --baseline monopoly/data/params_baseline.json --games 200 --seeds-file monopoly/data/seeds.txt
```

Рекомендуемый workflow:
1) train (50 итераций)
2) league auto-add
3) progress (200 игр) — сравнить с baseline и последними из лиги

## Данные локализации и правил

Файлы для редактирования:

- `monopoly/data/board.yaml` — поле с московскими названиями и каноническими ценами/рентой/домами/ипотекой (USD).
- `monopoly/data/cards_chance.yaml` — карточки Шанс (placeholder).
- `monopoly/data/cards_community.yaml` — карточки Казна (placeholder).
- `monopoly/data/rules.yaml` — флаги правил и базовые параметры (HR1/HR2/штраф тюрьмы и т.д.).
- `monopoly/data/params_baseline.json` — baseline параметры бота.
- `monopoly/data/league/*.json` — пул лучших параметров для self-play.
- `monopoly/data/seeds.txt` — фиксированный набор сидов для бенчмарка/прогресса.

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
