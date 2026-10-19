# V4.5-RU compact refinement

Изолированная реализация V4.5-A для RuSERRC и SciMDIX. Старые datasets, caches и artifacts используются только для чтения; каждый запуск создаёт новый `artifacts/qa_chunk_selection_v4_5_ru/run_<timestamp>`.

Primary comparison: DiTy + ASEM-style adaptive correction без Q5 selection/veto против DiTy B0. BGE используется как strong tested reference. RuSERRC и SciMDIX уже участвовали в разработке, поэтому результаты являются development characterization.

Проверка входов без загрузки моделей и без создания артефакта:

    untie_venv/bin/python -m experiments.qa_chunk_selection.v4_5_ru_compact_refinement --check-only

Фоновый запуск полного эксперимента:

    bash experiments/qa_chunk_selection/v4_5_ru_compact_refinement/launch_background.sh

Скрипт напечатает PID, путь к журналу и новый каталог результата. Прерванный запуск можно продолжить с тем же output:

    untie_venv/bin/python -m experiments.qa_chunk_selection.v4_5_ru_compact_refinement --phase all --datasets russerrc,scimdix --output-dir /ABS/PATH/TO/run_TIMESTAMP --resume

Smoke-проверка запускает один датасет, fold 0 и seed 42 и не является научным результатом:

    bash experiments/qa_chunk_selection/v4_5_ru_compact_refinement/launch_background.sh --smoke --benchmark-questions 2 --benchmark-warmup 1 --benchmark-repetitions 1

