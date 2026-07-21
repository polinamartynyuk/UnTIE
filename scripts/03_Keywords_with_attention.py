import sys
sys.path.append("/global_functions")
from global_functions.global_functions import *
import logging

#----------------------------
#       Подключение логгера

from logs_processing.logger_config import setup_logger

logger = setup_logger("main", "03_Keywords_with_attention2.log")
extra_logger = setup_logger("details", "03_extra_Keywords_with_attention2.log")

#----------------------------
# Датасет
datasets_path = "../datasets"
name = "scirex_structured.json"

# Результаты
datasets_results_path = "../datasets_results"
res_name = "results_keys_02.json"

# Модель документа
model_params_path = "../model_params"
model_name = "scart_init_model.json"
#----------------------------

# Загрузка датасета
df_loaded = load_dataframe_from_json(f"{datasets_path}/{name}")
df_loaded = replace_underscores_in_array_column(df=df_loaded,
                                        column_name="tasks")


# Создаем новый датафрейм для хранения результатов стратегий
df_strategies = pd.DataFrame(columns=[
    'doc_id',
    'original_text', 
    'tasks_cleaned',
    'first_answer',
    'score_chunk_strategy',
    'choose_cluster_strategy',
    'choose_answer_strategy',
    'final_answer',
    'base_metrics',
    'corrected_metrics'
])

# Чтение параметров модели
df_model_params = pd.read_json(f"{model_params_path}/{model_name}")

field_name = df_model_params["fields"][0]["field_name"]
questions = df_model_params["fields"][0]["questions"]
keywords = df_model_params["fields"][0]["keywords"]
text = df_model_params["fields"][0]["text"]

'''
score_chunk_strategy - стратегия вычисления score фрагментов:
- only_score_diff
- only_weight
- equal_combination

choose_cluster_strategy - стратегия выбора кластера:
- highest_avg_score
- weighted_score
- highest_cohesion

choose_answer_strategy - стратегия выбора ответа в кластере:
- highest_chunk_score
- highest_similarity
- combined_score
'''
for num_of_sample in range(len(df_loaded)):

    try:

        logger.info(f"Обработка {num_of_sample} начата")
        extra_logger.info(f"Обработка {num_of_sample} начата")

        # Тематический аспект и данные для фильтрации
        them_aspect = ThematicAspect(field_name, convert_into_question_class(questions))
        filtering_set = FilteringSet(text, keywords)

        # Метаданные
        doc_id = df_loaded["doc_id"][num_of_sample]
        text_of_doc = df_loaded["original_text"][num_of_sample]
        reference_answers = df_loaded["tasks_cleaned"][num_of_sample]

        # Записываемые значения
        first_chosen_answer = "--None--"


        # Делим на чанки
        text_proc = TextProcesser()
        sents = text_proc.split_into_sentences(text_of_doc)

        sentences = [Sentence(sents[i], i) for i in range(len(sents))]

        extra_logger.info(f"---> Число предложений: {len(sentences)}")

        chunk_builder = ChunkBuilder()
        text_chunks = chunk_builder.build_chunks(sentences)

        extra_logger.info(f"---> Число чанков: {len(text_chunks)}")

        
        # Первый быстрый ответ--------------------------------------
        simp_ans_finder = SimpleAnswerFinder()

        first_question_with_answers = simp_ans_finder.find_answers(
            question = convert_into_question_class(questions)[0],
            chunks = text_chunks
        )

        extra_logger.info("\n---\n---> Все ответы:")
        for ans in first_question_with_answers.answers:
            extra_logger.info(f"Текст: {ans.text}\nУверенность модели: {ans.confidence}\nЧанк: {ans.chunk.text[:100]}")
            extra_logger.info("----------------------\n")

        res1 = extract_short_answer(text=text_of_doc,
                        them_asp=them_aspect, 
                        filter_mode=FilterMode.NO_FILTER,
                        filter_set=filtering_set,
                        agg_mode=AggMode.NO_AGG)

        extra_logger.info(f"\n---> Первый ответ:{res1['final_answer'].text}")

        first_chosen_answer = res1['final_answer'].text

        first_answer_metrics = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                        text1=reference_answers[0], 
                                        text2=first_chosen_answer)
        #--------------------------------------------------------------

        #   Формируем данные для записи

        cur_doc_id = doc_id
        cur_text_of_doc = text_of_doc
        cur_reference_answers = reference_answers
        cur_first_chosen_answer = first_chosen_answer
        cur_first_answer_metrics = str(first_answer_metrics)

        #--------------------------------------------------------------

        #   Поиск валидных ответов

        validator = AnswerValidator()
        validator_result = validator.validate_answers(first_question_with_answers.answers, 
                                                        reference_answers[0])

        if not validator_result: # в массиве нет верных ответов

            logger.error(f"Обработка {num_of_sample}: нет верных ответов в массиве, корректировка невозможна")
            extra_logger.error(f"Обработка {num_of_sample}: нет верных ответов в массиве, корректировка невозможна")

            filler = "--No valid answers--"

            # Формируем данные
            new_row = {
                    'doc_id': cur_doc_id,
                    'original_text': cur_text_of_doc,
                    'tasks_cleaned': cur_reference_answers,
                    'first_answer': cur_first_chosen_answer,
                    'base_metrics': cur_first_answer_metrics,

                    'score_chunk_strategy': filler,
                    'choose_cluster_strategy': filler,
                    'choose_answer_strategy': filler,
                    'final_answer': filler,
                    'corrected_metrics': filler
            }

            # Добавляем в датафрейм
            df_strategies = pd.concat([df_strategies, pd.DataFrame([new_row])], ignore_index=True)
            extra_logger.info(f"Добавлено: {new_row}")

            # Сохраняем
            save_dataframe_to_json(df=df_strategies, 
                            file_path=f"{datasets_results_path}/{res_name}")
            extra_logger.info(f"Строка внесена в итоговый файл")

        #--------------------------------------------------------------

        else: # верные ответы нашлись

            extra_logger.info("Корректные ответы:")
            for ans in validator_result:
                extra_logger.info(f"\t {ans.text}")

            extra_logger.info(f"Обнаружена возможность скорректировать ответ!")
            valid_chunks = [ans.chunk.text for ans in validator_result]

            extra_logger.info("Искомые фрагменты:")
            for chunk in valid_chunks:
                extra_logger.info(f"\t {chunk}")

            dicts_with_attention_keywords = []

            # Собираем ключевые слова по вниманию
            for chunk in valid_chunks:
                key_dict = get_qa_attention_weights(model = BERTEngQASingleton(),
                                                question = questions[0], 
                                                context = chunk,
                                                device = get_device())
                dicts_with_attention_keywords.append(key_dict)

            
            # Постобработка ключевых слов по вниманию
            cleaned_keywords_by_chunks = []
            for key_dict in dicts_with_attention_keywords:
                    cleaned_keys_batch = get_keywords_top_attention_clean(key_dict['tokens'], 
                            key_dict['attention_weights'], top_k=100)
                    cleaned_keywords_by_chunks.append(cleaned_keys_batch)

            # Чистим чанки текста
            filtered_valid_chunks = dymamic_filter_uniform_words(valid_chunks, IDF_THRESHOLD)

            # Оставляем те слова, которые встретились в очищенных чанках
            filtered_keywords_by_chunks = []
            for key_dict, filtered_chunk in zip(cleaned_keywords_by_chunks, filtered_valid_chunks[0]):
                filtered_keywords_by_chunks.append(filter_words_by_text(key_dict, filtered_chunk))

            # Объединяем слова по чанкам
            all_filtered_keywords = merge_and_deduplicate_word_arrays(filtered_keywords_by_chunks)
            all_filtered_keywords = postprocess_words_array(all_filtered_keywords)

            # Добавляем леммы и стемы
            ls_filtered_keywords = get_lemm_stemm_words(all_filtered_keywords)

            # Считаем score_diff + фильтруем по нему
            ls_filtered_keywords = filter_keywords_from_dict(
                                keywords_with_weights = ls_filtered_keywords,
                                them_asp = them_aspect,
                                antireference = reference_answers[0]
            )

            extra_logger.info(f"Выделенные ключевые слова: ")
            for keyword in ls_filtered_keywords:
                extra_logger.info(f"{keyword}")

            #--------------------------------------------------------------

            # Оцениваем чанки по трем разным оценкам и находим ответы

            scored_ans_finder = ScoredAnswerFinder()


            #   Использование только score_diff
            sd_scored = extra_score_chunks_advanced(text_chunks, 
                                                    ls_filtered_keywords, 
                                                    weight_ratio=0.0)

            sd_question_with_answers = scored_ans_finder.find_answers(
                question = convert_into_question_class(questions)[0],
                scored_chunks = sd_scored
            )

            # Использование только weight
            w_scored = extra_score_chunks_advanced(text_chunks, 
                                                    ls_filtered_keywords, 
                                                    weight_ratio=1.0)

            w_question_with_answers = scored_ans_finder.find_answers(
                question = convert_into_question_class(questions)[0],
                scored_chunks = w_scored
            )

            # Равное сочетание обоих весов (по умолчанию)
            sdw_scored = extra_score_chunks_advanced(text_chunks, 
                                                    ls_filtered_keywords, 
                                                    weight_ratio=0.5)

            sdw_question_with_answers = scored_ans_finder.find_answers(
                question = convert_into_question_class(questions)[0],
                scored_chunks = sdw_scored
            )

            #--------------------------------------------------------------

            #   Перебираем все стратегии

            # - sd_question_with_answers
            # - w_question_with_answers
            # - sdw_question_with_answers

            ans_con_finder = AnswerConsensusFinder()

            cluster_selection_strategy = [
                    "highest_avg_score",
                    "weighted_score",
                    "highest_cohesion"
            ]

            answer_selection_strategy = [
                    "highest_chunk_score",
                    "highest_similarity",
                    "combined_score"
            ]

            for cluster_strategy in cluster_selection_strategy:
                for answer_strategy in answer_selection_strategy:

                    #   1) SD
                    sd_current_answer = ans_con_finder.extra_find_consensus_with_clustering(
                                        answers=sd_question_with_answers.answers,
                                        cluster_selection_strategy=cluster_strategy,
                                        answer_selection_strategy=answer_strategy
                                    )

                    corrected_metrics = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                            text1=reference_answers[0], 
                                            text2=sd_current_answer.text)

                    # Формируем данные
                    new_row = {
                        'doc_id': cur_doc_id,
                        'original_text': cur_text_of_doc,
                        'tasks_cleaned': cur_reference_answers,
                        'first_answer': cur_first_chosen_answer,
                        'base_metrics': cur_first_answer_metrics,

                        'score_chunk_strategy': "only_score_diff",
                        'choose_cluster_strategy': cluster_strategy,
                        'choose_answer_strategy': answer_strategy,
                        'final_answer': sd_current_answer.text,
                        'corrected_metrics': str(corrected_metrics)
                    }

                    # Добавляем в датафрейм
                    df_strategies = pd.concat([df_strategies, pd.DataFrame([new_row])], ignore_index=True)

                    # Сохраняем
                    save_dataframe_to_json(df=df_strategies, 
                                file_path=f"{datasets_results_path}/{res_name}")

                    #   2) W
                    w_current_answer = ans_con_finder.extra_find_consensus_with_clustering(
                                        answers=w_question_with_answers.answers,
                                        cluster_selection_strategy=cluster_strategy,
                                        answer_selection_strategy=answer_strategy
                                    )

                    corrected_metrics = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                            text1=reference_answers[0], 
                                            text2=w_current_answer.text)

                    # Формируем данные
                    new_row = {
                        'doc_id': cur_doc_id,
                        'original_text': cur_text_of_doc,
                        'tasks_cleaned': cur_reference_answers,
                        'first_answer': cur_first_chosen_answer,
                        'base_metrics': cur_first_answer_metrics,

                        'score_chunk_strategy': "only_weight",
                        'choose_cluster_strategy': cluster_strategy,
                        'choose_answer_strategy': answer_strategy,
                        'final_answer': w_current_answer.text,
                        'corrected_metrics': str(corrected_metrics)
                    }

                    # Добавляем в датафрейм
                    df_strategies = pd.concat([df_strategies, pd.DataFrame([new_row])], ignore_index=True)

                    # Сохраняем
                    save_dataframe_to_json(df=df_strategies, 
                                file_path=f"{datasets_results_path}/{res_name}")


                    #   3) SDW
                    sdw_current_answer = ans_con_finder.extra_find_consensus_with_clustering(
                                        answers=sdw_question_with_answers.answers,
                                        cluster_selection_strategy=cluster_strategy,
                                        answer_selection_strategy=answer_strategy
                                    )

                    corrected_metrics = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                            text1=reference_answers[0], 
                                            text2=sdw_current_answer.text)

                    # Формируем данные
                    new_row = {
                        'doc_id': cur_doc_id,
                        'original_text': cur_text_of_doc,
                        'tasks_cleaned': cur_reference_answers,
                        'first_answer': cur_first_chosen_answer,
                        'base_metrics': cur_first_answer_metrics,

                        'score_chunk_strategy': "equal_weight_score_diff",
                        'choose_cluster_strategy': cluster_strategy,
                        'choose_answer_strategy': answer_strategy,
                        'final_answer': sdw_current_answer.text,
                        'corrected_metrics': str(corrected_metrics)
                    }

                    # Добавляем в датафрейм
                    df_strategies = pd.concat([df_strategies, pd.DataFrame([new_row])], ignore_index=True)

                    # Сохраняем
                    save_dataframe_to_json(df=df_strategies, 
                                file_path=f"{datasets_results_path}/{res_name}")

    except Exception as e:
        print(f"! Problem with {num_of_sample}")

        logger.exception("Произошла ошибка:")
        raise ValueError(f"Exception text: {str(e)}")