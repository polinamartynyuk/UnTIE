# Инвентаризация компонентов

## Поддерживаемое ядро

- `untie.config` — валидируемые EN/RU-профили и настройки пайплайна.
- `untie.models` — ленивая загрузка tokenizer, QA-модели, pipeline и sentence encoder.
- `untie.text`, `untie.chunking` — разбиение на предложения и перекрывающиеся чанки.
- `untie.qa` — извлечение, валидация, агрегация и консенсус ответов.
- `untie.keywords`, `untie.ranking`, `untie.attention` — этапы переранжирования.
- `untie.pipelines` — сценарии baseline и attention.
- `untie.cli` — поддерживаемая точка входа из командной строки.
- `untie.results_analysis` — анализ JSON-результатов batch-экспериментов.

## Слой совместимости

- `scripts/run_baseline.py` и `scripts/run_attention.py` проксируют вызовы в CLI пакета.
- Нумерованные скрипты и модули ниже `scripts/` сохранены для совместимости с ноутбуками,
  но больше не являются каноническим API.
- `scripts/03_Keywords_with_attention_refactored.py` — refactored EN batch через `untie`.
- `scripts/03_Keywords_with_attention_refactored_ru.py` — refactored RU batch для RusErrC.

## Сохранённые эксперименты

- Нумерованные ноутбуки: историческое исследование chunking, QA, attention, RAG,
  русских датасетов, Qwen и анализа результатов.
- `scripts/models_processing/save_model_scripts` — заменено на `tools/model_download/download.py`.
- Сгенерированные CSV, JSON, PNG и log-файлы — локальные артефакты.

## Неактивные legacy-компоненты

- `FilterMode.BY_QUESTION` не имеет реализации в legacy-оркестраторе.
- Legacy NER getters — заглушки.
- `EnhancedAnswer`, `filter_characteristic_words`, несинглтоновый `SentenceTokenizer`
  и несколько вариантов consensus не имеют production-вызовов.
- LangChain/FAISS и KeyBERT встречаются только в экспериментальном коде или дампах зависимостей.

Ни один сохранённый legacy-компонент не импортируется новым пакетом `untie`.
