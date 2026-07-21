from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from pymorphy3 import MorphAnalyzer
import numpy as np
from typing import List
from nltk.stem import SnowballStemmer
import spacy
import torch
from Levenshtein import distance as levenshtein_distance
from typing import Tuple, Any, Dict

#---------------------------------------------------- 
#       Подключение логгера

from logs_processing.logger_config import setup_logger

logger = setup_logger(__name__, "functions.log")

#---------------------------------


# For lemmas and stemms
nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer("english")

def find_keywords(model,
                sentences_candidat, 
                sentences_reference, 
                collocation_len, 
                num_of_keywords):

    # sentences - текстовые предложения / тексты
    # collocation_len - количество слов в словосочетании
    # num_of_keywords - число выделяемых слов/словосочетаний

    logger.info(f"sentences_candidat:\n '{sentences_candidat}'")
    logger.info(f"sentences_reference:\n '{sentences_reference}'")

    if sentences_candidat == [] or sentences_reference == []:
        return []

    doc = ''.join(sentences_candidat)
    reference_text = ''.join(sentences_reference)

    n_gram_range = (collocation_len, collocation_len)
    stop_words = "english"

    logger.info(f"Текст для извлечения ключевых слов:\n '{doc}'")

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names_out()

    candidates = clean_digits(candidates)

    doc_embedding = model.encode([reference_text])
    candidate_embeddings = model.encode(candidates)

    top_n = num_of_keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    #print("*****")
    #print(distances)
    #print("***")
    #print(top_n)
    #print(distances.argsort()[0][-top_n:])
    if top_n <= len(candidates):
        keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    else:
        keywords = [candidates[index] for index in distances.argsort()[0][-len(candidates):]]

    return keywords


def contains_digit(sequence):
    return any(character.isdigit() for character in sequence)


def clean_digits(word_arr):

    filtered_arr = []

    for word in word_arr:
        if contains_digit(word):
            words = word.split()
            w_len = len(words) 
            if w_len==3:
                if (not contains_digit(words[0])) and (not contains_digit(words[2])):
                    filtered_arr.append(word)
            elif w_len==4:
                if (not contains_digit(words[0])) and (not contains_digit(words[3])):
                    filtered_arr.append(word)
        else:
            filtered_arr.append(word)

    filtered_arr = list(set(filtered_arr))

    return filtered_arr



morph = MorphAnalyzer()
def lemmatize(word):
    return morph.parse(word)[0].normal_form

def lemmatize_en(word):
    token = nlp(word)
    return token[0].lemma_

def filter_uniform_words(texts: List[str], idf_threshold: float = 1.5) -> List[str]:
    """
    Фильтрует слова, которые встречаются слишком равномерно (низкий IDF).
    
    Параметры:
        texts: список документов
        idf_threshold: минимальное значение IDF для сохранения слова.
                      Чем ниже порог, тем агрессивнее фильтрация.
                      Значение 1.5 ≈ отсекает слова в >60% документов.
    
    Возвращает:
        Список документов только с характерными словами.
    """

    logger.info(f"Тексты до лемматизации:\n {texts}")

    # 1. Подготовка данных
    original_words = [text.split() for text in texts]
        
    # 2. Создаем лемматизированные версии текстов
    lemmatized_texts = [
            ' '.join(lemmatize_en(word) for word in words)
            for words in original_words
    ]

    texts = lemmatized_texts

    logger.info(f"Тексты после лемматизации:\n {texts}")

    # 1. Считаем IDF для всех слов
    vectorizer = TfidfVectorizer(use_idf=True, norm=None, smooth_idf=False)
    X = vectorizer.fit_transform(texts)
    idf = vectorizer.idf_
    feature_names = vectorizer.get_feature_names_out()
    
    # 2. Фильтруем слова с низким IDF (слишком распространенные)
    words_to_keep = {
        word for word, score in zip(feature_names, idf) 
        if score >= idf_threshold
    }
    
    # 3. Применяем фильтр к исходным текстам
    filtered_texts = []
    for text in texts:
        words = text.split()
        filtered_words = [w for w in words if w in words_to_keep]
        filtered_texts.append(' '.join(filtered_words))

    logger.info(f"Тексты после фильтрации:\n {filtered_texts}")

    return filtered_texts

def dymamic_filter_uniform_words(
    texts: List[str],
    initial_idf_threshold: float = 1.5,
    min_idf_threshold: float = 0.5,
    step_reduction: float = 0.2,
    min_step: float = 0.05,
    target_words_per_doc: int = 3
    ) -> Tuple[List[str], float]:
    """
    Фильтрует слова с низкой дискриминативной способностью (IDF) с динамической настройкой порога.
    
    Параметры:
        texts: список документов
        initial_idf_threshold: начальное значение порога IDF
        min_idf_threshold: минимально допустимое значение IDF
        step_reduction: шаг уменьшения порога при отсутствии результатов
        min_step: минимальный шаг уменьшения
        target_words_per_doc: целевое минимальное количество слов в документе
    
    Возвращает:
        Кортеж: (отфильтрованные тексты, использованное значение IDF)
    """
    # 1. Подготовка данных
    original_words = [text.split() for text in texts]
    lemmatized_texts = [
        ' '.join(lemmatize_en(word) for word in words)
        for words in original_words
    ]
    
    # 2. Вычисление IDF
    vectorizer = TfidfVectorizer(use_idf=True, norm=None, smooth_idf=False)
    X = vectorizer.fit_transform(lemmatized_texts)
    idf = vectorizer.idf_
    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Динамический подбор порога
    current_threshold = initial_idf_threshold
    current_step = step_reduction
    best_result = None
    best_threshold = current_threshold
    
    while current_threshold >= min_idf_threshold:
        # Фильтрация слов
        words_to_keep = {
            word for word, score in zip(feature_names, idf) 
            if score >= current_threshold
        }
        
        # Применение фильтра
        filtered_texts = []
        valid_docs_count = 0
        
        for text in lemmatized_texts:
            words = text.split()
            filtered_words = [w for w in words if w in words_to_keep]
            filtered_text = ' '.join(filtered_words)
            filtered_texts.append(filtered_text)
            
            if len(filtered_words) >= target_words_per_doc:
                valid_docs_count += 1
        
        # Проверка качества фильтрации
        doc_coverage = valid_docs_count / len(texts)
        
        # Сохраняем лучший результат
        if best_result is None or doc_coverage > best_result[0]:
            best_result = (doc_coverage, filtered_texts, current_threshold)
        
        # Критерий остановки
        if doc_coverage >= 0.8:  # 80% документов с достаточным количеством слов
            break
            
        # Уменьшаем порог
        current_threshold -= current_step
        current_step = max(current_step * 0.8, min_step)
    
    # Возвращаем лучший найденный результат
    if best_result is None:
        logger.warning("Не удалось найти подходящий порог IDF. Возвращаю исходные тексты.")
        return texts, 0.0
    
    logger.info(f"Использованный порог IDF: {best_result[2]:.2f}, покрытие документов: {best_result[0]*100:.1f}%")
    return best_result[1], best_result[2]

def filter_characteristic_words(texts: List[str], 
                              min_tfidf: float = 0.1,
                              max_df: float = 0.8) -> List[str]:
    """
    Фильтрует тексты, оставляя только слова, характерные для каждого документа.
    
    Параметры:
        texts: Список текстов (документов)
        min_tfidf: Минимальное значение TF-IDF для сохранения слова
        max_df: Максимальная доля документов, в которых может встречаться слово
    
    Возвращает:
        Список отфильтрованных текстов (только с характерными словами)
    """
    # 1. Инициализация TF-IDF векторизатора
    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: [lemmatize(w) for w in text.split()],
        max_df=max_df,  # Игнорировать слова, встречающиеся в >80% документов
        min_df=2,       # Игнорировать слова, встречающиеся в <2 документах
        stop_words='russian',  # Удаление стандартных стоп-слов
        ngram_range=(1, 2)    # Учитывать словосочетания
    )
    
    # 2. Преобразование текстов в TF-IDF матрицу
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 3. Фильтрация слов с низкой значимостью
    filtered_texts = []
    for i, doc in enumerate(texts):
        # Получаем индексы значимых слов для документа
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        # Оставляем только слова с TF-IDF выше порога
        significant_words = [
            feature_names[j] for j in feature_index 
            if tfidf_matrix[i,j] >= min_tfidf
        ]
        # Фильтруем исходный текст
        words = doc.split()
        filtered_words = [w for w in words if w in significant_words]
        filtered_texts.append(' '.join(filtered_words))
    
    return filtered_texts

def get_lemm_stemm_words(words: List[str]) -> List[str]:

    lemmas = [lemmatize_en(word) for word in words]
    stems = [stemmer.stem(word) for word in words]

    if not (len(words) == len(lemmas) == len(stems)):
        raise ValueError("Words, lemmas and stems should have same length")

    return [
        {"word": word, "lemma": lemma, "stem": stem}
        for word, lemma, stem in zip(words, lemmas, stems)
    ]

def calculate_string_metrics(model: 'Transformers Encoder Model',
                            text1: str, text2: str) -> dict:
    """
    Вычисляет метрики сравнения двух строк:
    - Косинусное сходство эмбеддингов
    - Расстояние Левенштейна
    
    Параметры:
        text1: первая строка
        text2: вторая строка
        
    Возвращает:
        dict
    """
    # Вычисляем эмбеддинги
    embeddings = model.tokenizer.encode([text1, text2])
    
     # Вычисляем косинусное сходство через sklearn
    cosine_sim = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]
    
    # Расстояние Левенштейна
    lev_dist = levenshtein_distance(text1, text2)
    
    return {'cosine_sim': float(cosine_sim), 
            'lev_dist': lev_dist
    }

def get_qa_attention_weights(model, question, context, device):
    """
    Визуализирует веса внимания модели RoBERTa для QA между вопросом и контекстом.
    
    Args:
        question (str): Текст вопроса
        context (str): Текст контекста для поиска ответа
        model_name (str): Название предобученной модели (по умолчанию 'deepset/roberta-base-squad2')
    
    Returns:
        dict: Словарь с токенами и их весами внимания
    """
    
    # Загрузка модели и токенизатора
    model = model

    tokenizer = model.tokenizer
    model = model.model
    
    # Токенизация
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_ids = inputs['input_ids']
    
    # Получение предсказаний модели с вниманием
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions  # Все слои внимания
    
    # Среднее внимание по всем головам и слоям
    all_attentions = torch.stack(attentions)  # [layers, batch, heads, seq_len, seq_len]
    averaged_attention = all_attentions.mean(dim=(0, 1, 2))  # [seq_len, seq_len]
    
    # Индексы токенов контекста (после [SEP])
    sep_index = torch.where(input_ids[0] == tokenizer.sep_token_id)[0][0].item()
    context_start = sep_index + 1
    context_end = len(input_ids[0]) - 1  # Исключаем конечный [SEP]
    
    # Извлекаем внимание к токенам контекста
    context_attention = averaged_attention[context_start:context_end+1, context_start:context_end+1]
    
    # Агрегируем внимание по строкам (входящие связи)
    token_importance = context_attention.mean(dim=0)
    
    # Нормализация
    token_importance = token_importance / token_importance.sum()
    
    # Сопоставление с токенами
    context_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][context_start:context_end+1])
    
    return {
        'tokens': context_tokens,
        'attention_weights': token_importance.cpu().numpy(),
        'question': question,
        'context': context
    }

def clean_token(token):
    """Очищает токен от специальных символов RoBERTa"""
    # Убираем символ пробела Ġ
    cleaned = token.replace('Ġ', ' ')
    # Убираем лишние пробелы
    cleaned = cleaned.strip()
    # Убираем точки, которые являются частью токенизации (но не в середине слова)
    if cleaned.endswith('.') and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    # Приводим к нижнему регистру
    cleaned = cleaned.lower()
    return cleaned

def reconstruct_words(tokens, weights):
    """Восстанавливает полные слова из субтокенов"""
    words = []
    word_weights = []
    current_word = ""
    current_weight = 0
    count = 0
    
    for token, weight in zip(tokens, weights):
        cleaned_token = clean_token(token)
        
        # Пропускаем пустые токены после очистки
        if not cleaned_token:
            continue
            
        # Если токен начинается с пробела - это новое слово
        if token.startswith('Ġ') and current_word:
            # Убираем возможную точку в конце слова
            if current_word.endswith('.'):
                current_word = current_word[:-1]
            words.append(current_word)
            word_weights.append(current_weight / count if count > 0 else 0)
            current_word = cleaned_token
            current_weight = weight
            count = 1
        else:
            # Продолжение текущего слова (субтокен)
            current_word += cleaned_token
            current_weight += weight
            count += 1
    
    # Добавляем последнее слово
    if current_word:
        # Убираем возможную точку в конце последнего слова
        if current_word.endswith('.'):
            current_word = current_word[:-1]
        words.append(current_word)
        word_weights.append(current_weight / count if count > 0 else 0)
    
    return words, np.array(word_weights)

def filter_special_tokens(tokens, weights):
    """Фильтрует специальные токены"""
    valid_tokens = []
    valid_weights = []
    
    special_tokens = ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '.', '?', '!', ',', '/', '//', '][']
    
    for token, weight in zip(tokens, weights):
        # Пропускаем специальные токены и знаки препинания
        if (token not in special_tokens and 
            not token.startswith('<') and 
            not token.endswith('>') and
            len(token.strip('.')) > 0):  # Исключаем токены, состоящие только из точек
            valid_tokens.append(token)
            valid_weights.append(weight)
    
    return valid_tokens, np.array(valid_weights)

def visualize_top_attention_clean(tokens, weights, top_k=10):
    """Визуализация с очищенными токенами"""
    # Фильтруем специальные токены
    filtered_tokens, filtered_weights = filter_special_tokens(tokens, weights)
    
    # Восстанавливаем полные слова
    words, word_weights = reconstruct_words(filtered_tokens, filtered_weights)
    
    # Нормализуем веса после восстановления
    # if len(word_weights) > 0:
    #    word_weights = word_weights / word_weights.sum()
    
    indices = np.argsort(word_weights)[::-1][:top_k]
    
    print("Топ-%d наиболее важных слов:" % top_k)
    for i, idx in enumerate(indices):
        if i < len(words):
            print(f"{i+1}. '{words[idx]}': {word_weights[idx]:.4f}")

def get_keywords_top_attention_clean(tokens, weights, top_k=10):
    """Визуализация с очищенными токенами"""
    # Фильтруем специальные токены
    filtered_tokens, filtered_weights = filter_special_tokens(tokens, weights)
    
    # Восстанавливаем полные слова
    words, word_weights = reconstruct_words(filtered_tokens, filtered_weights)
    
    # Нормализуем веса после восстановления
    # if len(word_weights) > 0:
    #    word_weights = word_weights / word_weights.sum()
    
    indices = np.argsort(word_weights)[::-1][:top_k]

    keywords = []

    for i, idx in enumerate(indices):
        if i < len(words):
            keywords.append({
                'word': words[idx],
                'weight': word_weights[idx]
            })

    return keywords

def filter_words_by_text(words_dicts: list, text: str) -> list:
    """
    Оставляет в массиве словарей только те слова, которые присутствуют в тексте.
    """
    text_words = set(text.lower().split())
    return [word_dict for word_dict in words_dicts if word_dict['word'].lower() in text_words]

def merge_and_deduplicate_word_arrays(arrays: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Объединяет массивы словарей, оставляя для каждого слова запись с наивысшим весом.
    Сортирует результат по весу в убывающем порядке.
    """
    word_max_weight = {}
    
    # Проходим по всем массивам и находим максимальный вес для каждого слова
    for array in arrays:
        for word_dict in array:
            word = word_dict['word']
            weight = float(word_dict['weight'])  # Конвертируем np.float32 в float
            
            # Если слово уже встречалось, берем максимальный вес
            if word not in word_max_weight or weight > word_max_weight[word]:
                word_max_weight[word] = weight
    
    # Создаем результирующий массив
    result = [{'word': word, 'weight': weight} for word, weight in word_max_weight.items()]
    
    # Сортируем по весу в убывающем порядке
    result.sort(key=lambda x: x['weight'], reverse=True)
    
    return result

def postprocess_words_array(words_dicts: list) -> list:
    """
    Постобработка массива слов: удаление стоп-слов английского языка.
    """
    # Загружаем стоп-слова
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this', 'that',
        'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'as', 'from'
    }
    """
    stop_words = {
    # Артикли и основные союзы
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    
    # Глаголы-связки и вспомогательные глаголы
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    
    # Местоимения
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    
    # Предлоги
    'as', 'from', 'into', 'upon', 'about', 'against', 'between', 'through', 'during',
    'before', 'after', 'above', 'below', 'up', 'down', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    
    # Часто употребимые наречия
    'often', 'early', 'unlikely', 'usually', 'always', 'never', 'sometimes', 'frequently',
    'rarely', 'seldom', 'generally', 'normally', 'typically', 'occasionally', 'constantly',
    'continuously', 'regularly', 'daily', 'weekly', 'monthly', 'yearly', 'annually',
    'soon', 'late', 'later', 'earlier', 'recently', 'currently', 'presently', 'immediately',
    'quickly', 'slowly', 'suddenly', 'gradually', 'rapidly', 'easily', 'hardly', 'barely',
    'scarcely', 'clearly', 'obviously', 'evidently', 'apparently', 'possibly', 'probably',
    'likely', 'unlikely', 'certainly', 'definitely', 'absolutely', 'completely', 'totally',
    'entirely', 'fully', 'partially', 'mostly', 'mainly', 'chiefly', 'primarily',
    'especially', 'particularly', 'specifically', 'exactly', 'precisely', 'accurately',
    
    # Относительные местоимения и вопросительные слова
    'who', 'whom', 'whose', 'which', 'what', 'when', 'where', 'why', 'how',
    
    # Слова-связки и переходные слова
    'therefore', 'however', 'thus', 'hence', 'consequently', 'accordingly', 'furthermore',
    'moreover', 'additionally', 'besides', 'also', 'too', 'likewise', 'similarly',
    'conversely', 'instead', 'rather', 'otherwise', 'nevertheless', 'nonetheless',
    'although', 'though', 'while', 'whereas', 'despite', 'regardless', 'notwithstanding',
    
    # Другие частые слова без значительной смысловой нагрузки
    'very', 'quite', 'rather', 'fairly', 'pretty', 'somewhat', 'almost', 'nearly',
    'just', 'even', 'still', 'yet', 'already', 'else', 'otherwise', 'indeed',
    'certainly', 'surely', 'maybe', 'perhaps', 'possibly', 'probably', 'likely',
    
    # Числительные и количественные местоимения
    'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next', 'previous',
    'many', 'much', 'more', 'most', 'few', 'less', 'least', 'several', 'various',
    
    # Прочие частые слова
    'well', 'back', 'down', 'up', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'now', 'again', 'ever', 'never', 'always'
}
    
        
    # Создаем новый массив без стоп-слов
    result = []
    for word in words_dicts:
        if word['word'] not in stop_words and len(word['word']) > 1:  # исключаем стоп-слова и одиночные символы
            result.append(word)
    
    return result

def get_lemm_stemm_words(words_with_weights: List[dict]) -> List[dict]:
    """
    Добавляет леммы и стеммы к словам с сохранением весов.
    """
    words = [item['word'] for item in words_with_weights]
    weights = [item['weight'] for item in words_with_weights]
    
    lemmas = [lemmatize_en(word) for word in words]
    stems = [stemmer.stem(word) for word in words]

    if not (len(words) == len(lemmas) == len(stems) == len(weights)):
        raise ValueError("Words, lemmas, stems and weights should have same length")

    return [
        {
            "word": word,
            "lemma": lemma,
            "stem": stem,
            "weight": weight
        }
        for word, lemma, stem, weight in zip(words, lemmas, stems, weights)
    ]