import sys
sys.path.append("/global_functions")
from global_functions.global_functions import *
import logging

#----------------------------
#       Подключение логгера

from logs_processing.logger_config import setup_logger

logger = setup_logger(__name__, "02_All_dataset_process.log")

#----------------------------
# Датасет
datasets_path = "../datasets"
name = "scirex_structured.json"

# Результаты
datasets_results_path = "../datasets_results"
res_name = "results_2.json"

# Модель документа
model_params_path = "../model_params"
model_name = "scart_init_model.json"
#----------------------------

# Загрузка датасета
df_loaded = load_dataframe_from_json(f"{datasets_path}/{name}")
df_loaded = replace_underscores_in_array_column(df=df_loaded,
                                        column_name="tasks")

# Новые колонки
df_loaded['first_answer'] = None
df_loaded['corrected_answer'] = None
df_loaded['keywords'] = None
df_loaded['metrics_orig'] = None
df_loaded['metrics_new'] = None

# Оставляем только нужные колонки и новые колонки
columns_to_keep = ['doc_id', 'original_text', 'tasks', 'tasks_cleaned', 
                    'first_answer', 'corrected_answer', 'keywords', 
                    'metrics_orig', 'metrics_new']
df_loaded = df_loaded[columns_to_keep]

# Чтение параметров модели
df_model_params = pd.read_json(f"{model_params_path}/{model_name}")

field_name = df_model_params["fields"][0]["field_name"]
questions = df_model_params["fields"][0]["questions"]
keywords = df_model_params["fields"][0]["keywords"]
text = df_model_params["fields"][0]["text"]


for num_of_sample in range(len(df_loaded)):

    try:

        logger.info(f"Обработка {num_of_sample} начата")

        # Тематический аспект и данные для фильтрации
        them_aspect = ThematicAspect(field_name, convert_into_question_class(questions))
        filtering_set = FilteringSet(text, keywords)

        text_of_doc = df_loaded["original_text"][num_of_sample]
        reference_answers = df_loaded["tasks_cleaned"][num_of_sample]

        res1 = extract_short_answer(text=text_of_doc,
                    them_asp=them_aspect, 
                    filter_mode=FilterMode.NO_FILTER,
                    filter_set=filtering_set,
                    agg_mode=AggMode.NO_AGG)

        #df_loaded.loc[num_of_sample, "first_answer"] = res1['final_answer'].text
        df_loaded.loc[num_of_sample, "first_answer"] = getattr(res1['final_answer'], "text", "--None--")
        

        # МЕТРИКИ
        metrics_orig = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                        text1=df_loaded.loc[num_of_sample, "tasks_cleaned"][0], 
                                        text2=res1['final_answer'].text)

        df_loaded.loc[num_of_sample, "metrics_orig"] = str(metrics_orig)    

        res2 = extract_keywords(them_asp=them_aspect,
                            reference_answer=reference_answers[0])

        if keys_extracted(res2):

            filtered_keywords = filter_keywords(result=res2, 
                            them_asp=them_aspect,
                            similarity_threshold=0.5,
                            dissimilarity_threshold=0.55)

            # TODO > Здесь можно тестировать, насколько конкретно эти слова позволяют искать нужный фрагмент текста
            
            new_filtering_set = FilteringSet(text, filtered_keywords)
            new_them_aspect = ThematicAspect(field_name, convert_into_question_class(questions))

            res3 = extract_short_answer(text=text_of_doc,
                        them_asp=new_them_aspect, 
                        filter_mode=FilterMode.BY_KW_LEMSTEM,
                        filter_set=new_filtering_set,
                        agg_mode=AggMode.NO_AGG)

            #df_loaded.loc[num_of_sample, "first_answer"] = res1['final_answer'].text

            # Если атрибута нет, вернёт пустую строку
            df_loaded.loc[num_of_sample, "corrected_answer"] = getattr(res3['final_answer'], "text", "--None--")
            

            df_loaded.loc[num_of_sample, "keywords"] = str(filtered_keywords)

            if (getattr(res3['final_answer'], "text", None) is not None):

                # МЕТРИКИ
                metrics_new = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                        text1=df_loaded.loc[num_of_sample, "tasks_cleaned"][0], 
                                        text2=res3['final_answer'].text)

                df_loaded.loc[num_of_sample, "metrics_new"] = str(metrics_new)
            else:
                df_loaded.loc[num_of_sample, "metrics_new"] = "--None--"

            save_dataframe_to_json(df=df_loaded, 
                            file_path=f"{datasets_results_path}/{res_name}")

        else: # Ключевые слова не извлечены, корректировка ответа невозможна

            #df_loaded.loc[num_of_sample, "first_answer"] = res1['final_answer'].text
            df_loaded.loc[num_of_sample, "corrected_answer"] = "--None--"
            df_loaded.loc[num_of_sample, "keywords"] = "--None--"

            # МЕТРИКИ
            metrics_new = calculate_string_metrics(model=SentenceTokenizerSingleton(),
                                    text1=df_loaded.loc[num_of_sample, "tasks_cleaned"][0], 
                                    text2=res1['final_answer'].text)

            df_loaded.loc[num_of_sample, "metrics_new"] = str(metrics_new)

            save_dataframe_to_json(df=df_loaded, 
                            file_path=f"{datasets_results_path}/{res_name}")

        logger.info(f"Обработка {num_of_sample} завершена")

    except Exception as e:
        print(f"! Problem with {num_of_sample}")

        logger.exception("Произошла ошибка:")
        raise ValueError(f"Exception text: {str(e)}")
        # continue

#df_empty = get_empty_answer_records(df_loaded)