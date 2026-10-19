# V4.4-RU: ranker × second-component factorial

Независимый эксперимент для `RuSERRC` и `SciMDIX`. Он не меняет артефакты
подготовки данных, английский V4.4, прежний русский V4.4 или их результаты.

## Что проверяется

- ранкеры `ARGA100/ru-reranker-modernbert-small` и
  `DiTy/cross-encoder-russian-msmarco`;
- 5 document-grouped CV folds и seeds 42/43/44;
- Q0 и fold-local выбор наиболее успешного вопроса по аспекту;
- K=3, 4, 5;
- B0, V6, SEM, LEX, LEX-LEM, ASEM, ALEX, ASEM-Q5 и ALEX-Q5;
- отдельные эффекты ранкера, вопроса, второго компонента и полного метода.

Модели ранкеров обучаются на том же objective и том же соотношении
positive:hard-negative:random-negative `1:3:1`, что существующая реализация.
Веса и пороги второго компонента подбираются отдельно для каждого датасета,
ранкера, K и внешнего фолда только по train+calibration. Веса моделей после
fold не сохраняются; сохраняются OOF-предсказания и отчёты обучения.

## Предварительная проверка

```bash
cd /home/aiserver/POLINA/UnTIE_project
untie_venv/bin/python -m experiments.qa_chunk_selection.v4_4_ru_ranker_factorial \
  --check-only \
  --output-dir artifacts/qa_chunk_selection_v4_4_ru_rankers_check
```

Проверка не создаёт каталог, не загружает модели и не запускает эксперимент.

## Фоновый запуск

```bash
cd /home/aiserver/POLINA/UnTIE_project
./experiments/qa_chunk_selection/v4_4_ru_ranker_factorial/launch_background.sh
```

Скрипт печатает PID, путь к логу и новый timestamped-каталог результатов.
Ход работы виден через напечатанную команду `tail -f`.

Чтобы ограничить прогон одним набором или ранкером, передайте параметры:

```bash
./experiments/qa_chunk_selection/v4_4_ru_ranker_factorial/launch_background.sh \
  --datasets scimdix --rankers modernbert_small
```

После прерывания продолжайте в том же каталоге:

```bash
nohup untie_venv/bin/python \
  -m experiments.qa_chunk_selection.v4_4_ru_ranker_factorial \
  --output-dir /ABSOLUTE/PATH/TO/EXISTING_OUTPUT \
  --resume --phase all > /ABSOLUTE/PATH/TO/resume.log 2>&1 &
```

Если модели заранее помещены в Hugging Face cache, можно добавить
`--local-files-only`.

## Результаты

Каждый датасет получает отдельные каталоги `cross_encoders/`, `protocol/`,
`analysis/` и `reports/`. Главный файл декомпозиции:
`<output>/<dataset>/analysis/factor_decomposition.json`. Общая краткая сводка:
`<output>/reports/results.md`.

