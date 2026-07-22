# UnTIE

UnTIE извлекает короткие структурированные ответы из научных документов. Поддерживаемое ядро реализует два сценария:

1. **Baseline:** разбиение на предложения → перекрывающиеся чанки → extractive QA → семантическая агрегация ответов.
2. **Attention reranking:** валидация baseline → ключевые слова по attention → семантический контраст ключей → скоринг чанков → QA → выбор по консенсусу.

Refactored-пакет безопасен для импорта: при `import untie` не инициализируется CUDA, не скачиваются веса и не загружаются NLP-модели.

## Установка

Создайте виртуальное окружение вне Git и установите проект:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/core.txt
```

Для тестов и ноутбуков используйте `requirements/dev.txt`. Исторические дампы зависимостей в `requirements/requirements*.txt` и `updatereqs.txt` сохранены только для воспроизводимости; они содержат посторонние системные пакеты и не являются поддерживаемыми манифестами установки.

## Модели

Скачивайте локальные веса явно:

```bash
python tools/model_download/download.py en-qa
python tools/model_download/download.py en-sentence
python tools/model_download/download.py ru-qa
```

Профили по умолчанию:

- `en`: локальные веса QA `deepset/roberta-base-squad2` и локальный sentence encoder; attention поддерживается.
- `ru`: локальные веса QA на базе XLM-R и `DeepPavlov/rubert-base-cased-sentence`; в базовом CLI-профиле attention выключен, но в refactored batch-скрипте для RusErrC включается явно.

Пути можно переопределить, создав `ModelProfile` и `PipelineConfig` в Python. Выбор устройства по умолчанию — первый CUDA-устройство при наличии, иначе CPU; для принудительного CPU передайте `--device cpu`.

## CLI

Baseline:

```bash
python -m untie.cli article.txt \
  --language en \
  --question "Which task was solved?"
```

Attention reranking:

```bash
python -m untie.cli article.txt \
  --language en \
  --mode attention \
  --question "Which task was solved?" \
  --reference-answer "semantic segmentation"
```

Команда выводит один JSON-объект с полями `answer`, `confidence`, `chunks_used` и метаданными пайплайна. Точки совместимости: `scripts/run_baseline.py` и `scripts/run_attention.py`.

## Python API

Реализации моделей подключаются через протоколы в `untie.protocols`. Это позволяет тестировать пайплайн без Hugging Face, GPU и локальных весов. Для production-загрузки используйте `untie.models.ModelFactory`.

Основные модули:

- `untie.config` — валидируемые профили моделей и пороги
- `untie.domain` — типизированные значения пайплайна
- `untie.text`, `untie.chunking` — подготовка документа
- `untie.qa` — извлечение, валидация, агрегация и консенсус ответов
- `untie.keywords`, `untie.attention`, `untie.ranking` — переранжирование
- `untie.pipelines` — оркестрация baseline и attention

## Тесты

```bash
python -m pytest
```

Тесты используют fake tokenizer, encoder и QA; они не скачивают модели и не требуют CUDA. Контрактные примеры EN/RU — в `tests/fixtures/`.

## Структура репозитория

- `untie/` — поддерживаемый пакет
- `tests/` — тесты без моделей и небольшие фикстуры
- `scripts/` — legacy research-код и совместимые launcher-скрипты
- `experiments/` — рабочая область и рекомендации по миграции ноутбуков
- `tools/model_download/` — явное получение моделей
- `artifacts/` — игнорируемые сгенерированные результаты
- `docs/component-inventory.md` — карта активных и неактивных компонентов

Существующие ноутбуки пока остаются в `scripts/`, пока не будут мигрированы встроенные пути. Они сохранены как эксперименты и не должны использоваться как источник переиспользуемой логики пайплайна.
