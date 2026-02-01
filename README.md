# Монополия (боты против ботов)

Streamlit-спектатор и детерминируемый движок Монополии. UI подключен к движку и управляет симуляцией. Все ключевые правила настраиваются через YAML-файлы.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Важно: размещайте репозиторий в обычной папке с правами записи (не Temp/Program Files), иначе pip/pytest могут не иметь прав на временные каталоги.

## Windows Quickstart

Зачем venv: изолирует зависимости проекта, чтобы не ставить их глобально.

PowerShell:

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

В UI включайте режимы **Тренировка** и **Live матч** через левую панель. Лучшие параметры лежат в `runs/<timestamp>/cycle_XXX/best.json`.

## Запуск

```bash
streamlit run app.py
```

## Как пользоваться (коротко)

1) Откройте Streamlit и выберите режим **Тренировка**. Нажмите **Старт** — обучение идёт до плато.
2) По завершении нажмите **Запустить live матч сейчас** или перейдите в **Live матч** и стартуйте 6 deep‑ботов.
3) Лучшие параметры лежат в `runs/<timestamp>/cycle_XXX/best.json` — их можно переиспользовать для новых матчей.

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
- Бот использует feature-based scoring с 3 стадиями (early/mid/late); параметры содержат веса для каждой стадии.

## Auto-evolve (bootstrap -> league -> meta-cycle)

Важно:
- Обучение идёт без thinking-mode, чтобы не замедлять и не зашумлять оценку кандидатов.
- Оппоненты формируются только из `monopoly/data/league/index.json` (TOP-16).
- В Streamlit режим **Тренировка** запускает auto-evolve в фоне.

CLI-запуск:

```bash
python -m monopoly.autoevolve run --workers auto
```

Содержимое `runs/<timestamp>/`:
- `status.json` — мета-статус auto-evolve (цикл, new_bests, meta-plateau).
- `cycle_XXX/status.json` — статус текущего цикла.
- `cycle_XXX/train_log.csv` — история по эпохам.
- `cycle_XXX/progress.txt` — человекочитаемый лог.
- `cycle_XXX/best.json` — лучшие параметры цикла.
- `error.log` — stderr при падении (если было).

Запуск симуляции с параметрами цикла:

```bash
python -m monopoly.sim --params runs/<timestamp>/cycle_XXX/best.json --players 6 --seed 42 --games 1
```

Live матч (6 “deep planner” ботов, запись состояния для UI):

```bash
python -m monopoly.live --players 6 --params runs/<timestamp>/cycle_XXX/best.json --mode deep --workers auto --time-per-decision-sec 3.0 --horizon-turns 60 --seed 42 --out runs/<timestamp>/live_state.json
```

## Бенчмарк параметров

```bash
python -m monopoly.bench --games 200 --players 6 --seed 123 --candidate runs/<timestamp>/cycle_XXX/best.json --baseline monopoly/data/params_baseline.json --league-dir monopoly/data/league --opponents mixed
```

Можно использовать фиксированный набор сидов:

```bash
python -m monopoly.bench --games 200 --seeds-file monopoly/data/seeds.txt --candidate runs/<timestamp>/cycle_XXX/best.json --baseline monopoly/data/params_baseline.json --league-dir monopoly/data/league --opponents mixed
```

## Лига и прогресс

Добавление лучшей конфигурации в лигу (TOP-16 по fitness):

```bash
python -m monopoly.league add --params runs/<timestamp>/cycle_XXX/best.json --name best_YYYYMMDD_HHMM --meta "iter=50; fitness=..." --fitness 0.123 --top-k 16
```

Список лиги и очистка старых:

```bash
python -m monopoly.league list
python -m monopoly.league prune --top-k 16
```

Оценка прогресса (топ-5 из лиги vs baseline и mix):

```bash
python -m monopoly.progress --league-dir monopoly/data/league --baseline monopoly/data/params_baseline.json --games 200 --seed 123
```

С фиксированным seed pack:

```bash
python -m monopoly.progress --league-dir monopoly/data/league --baseline monopoly/data/params_baseline.json --games 200 --seeds-file monopoly/data/seeds.txt
```

Рекомендуемый workflow:
1) auto-evolve (meta-циклы)
2) progress (200 игр) — сравнить с baseline и лигой

### Как считается fitness

- Для каждой игры считается `win_like_outcome`:
  - если игра завершилась естественно (банкротства): 1.0 за победу, иначе 0.0;
  - если игра остановлена по `max_steps`: используется место по net worth (1→1.00, 2→0.60, 3→0.30, 4→0.10, 5–6→0.00).
- `win_lcb` — нижняя граница Wilson CI для `win_like_outcome` (confidence=0.80).
- Дополнительно считаются `place_score`, `advantage = tanh((net_worth - mean_others)/scale)` и `cutoff_rate`.
- Итоговая формула:
  `fitness = 1000 * win_lcb + 10 * place_score + 1 * advantage - 5 * cutoff_rate`.

## Данные локализации и правил

Файлы для редактирования:

- `monopoly/data/board.yaml` — поле с московскими названиями и каноническими ценами/рентой/домами/ипотекой (USD).
- `monopoly/data/cards_chance.yaml` — карточки Шанс (placeholder).
- `monopoly/data/cards_community.yaml` — карточки Казна (placeholder).
- `monopoly/data/cards_texts_ru_official.yaml.template` — шаблон для официальных текстов (id -> text_ru).
- `monopoly/data/cards_texts_ru_official.yaml` — опциональный файл с официальными текстами (id -> text_ru), добавьте его вручную на основе шаблона (файл в .gitignore).
- `monopoly/data/rules.yaml` — флаги правил и базовые параметры (HR1/HR2/штраф тюрьмы и т.д.).
- `monopoly/data/params_baseline.json` — baseline параметры бота.
- `monopoly/data/league/index.json` — индекс лиги (TOP-16 по fitness, ранги 1..N).
- `monopoly/data/league/*.json` — параметры ботов из index.json.
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
- Сделки между ботами (offer/accept), включая передачу заложенной собственности по правилам.
- Бот v2: позиционная угроза, scarcity/denial домов и отелей, синергии ЖД/коммуналок, давление по кэшу оппонентов, HR2-aware сигналы для решения тюрьмы.

## Допущения и TODO

- Тексты карточек — placeholders (замените в YAML).
- Ручного участия нет; сделки включены.
- Численные значения в `board.yaml` канонические (USD).

