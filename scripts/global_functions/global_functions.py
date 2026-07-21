#---------------------------------------------------- 
#       Подключение модулей

from models_processing.models_setting import *
from text_processing.text_extraction import *
from classes.model_classes import *
from functions.data_processing import *

#---------------------------------------------------- 
#       Подключение логгера

from logs_processing.logger_config import setup_logger

logger = setup_logger(__name__, "global_functions.log")

#----------------------------------------------------
#       Настройки

from config.config import KEYWORDS_FILTERING, CHUNKS_FILTERING

similarity_threshold = KEYWORDS_FILTERING["similarity_threshold"]
dissimilarity_threshold = KEYWORDS_FILTERING["dissimilarity_threshold"]

lemm_stemm_min_keywords = CHUNKS_FILTERING['by_kw_lemm_stemm_min_keywords']

#----------------------------------------------------   

from enum import Enum

#----------------------------------------------------
    
class FilterMode(Enum):
        NO_FILTER = 1
        BY_QUESTION = 2
        BY_KEYWORDS = 3
        BY_KW_LEMSTEM = 4

class AggMode(Enum):
        NO_AGG = 1
        CLUST = 2

#----------------------------------------------------

def convert_into_question_class(questions_text: List[str]) -> List[Question]:
    return [Question(question_text) for question_text in questions_text]

#----------------------------------------------------

#def extract_short_answer_from_chunks(chunks: chunks,
                                 #   question: question):


#----------------------------------------------------

def extract_short_answer(text: str, 
                them_asp: ThematicAspect, 
                filter_mode: FilterMode,
                filter_set: FilteringSet,
                agg_mode: AggMode) -> dict:

    text_proc = TextProcesser()
    sents = text_proc.split_into_sentences(text)

    sentences = [Sentence(sents[i], i) for i in range(len(sents))]

    logger.info("---> Number of sentences:")
    logger.info(f"--> {len(sentences)}")

    chunk_builder = ChunkBuilder()
    text_chunks = chunk_builder.build_chunks(sentences)

    logger.info("---\n---> Number of chunks:")
    logger.info(f"--> {len(text_chunks)}")
    

    filtered_text_chunks = []
    if filter_mode == FilterMode.NO_FILTER:

        filtered_text_chunks = text_chunks

        pass

    elif filter_mode == FilterMode.BY_KEYWORDS:

        ch_filter = ChunkFilter()

        filtered_text_chunks = ch_filter.filter_chunks_by_keywords(text_chunks, filter_set.keywords)

        logger.info("---\n---> Number of filtered by keywords chunks:")
        logger.info(f"--> {len(filtered_text_chunks)}")

    elif filter_mode == FilterMode.BY_KW_LEMSTEM:

        ch_filter = ChunkFilter()

        filtered_text_chunks = ch_filter.filter_chunks_by_keywords_lemm_stemm(chunks=text_chunks, 
                                                    keywords=get_lemm_stemm_words(filter_set.keywords),
                                                    min_keywords=lemm_stemm_min_keywords)

        logger.info("\n---\n---> Number of filtered by keyword's lemms & stemms chunks:")
        logger.info(f"--> {len(filtered_text_chunks)}")

    if (len(filtered_text_chunks) == 0):
        return {
        "final_answer": None,
        "thematic_aspect": them_asp
        }

    text_chunks = filtered_text_chunks


    simp_ans_finder = SimpleAnswerFinder()

    questions_with_answers = []

    for question in them_asp.questions:
        question_with_answers = simp_ans_finder.find_answers(
        question = question,
        chunks = text_chunks
        )
        questions_with_answers.append(question_with_answers)

    logger.info("\n---\n---> Founded answers:")
    for question_with_answers in questions_with_answers:
        logger.info(f"----->{questions_with_answers.index(question_with_answers) + 1}/{len(questions_with_answers)}")
        logger.info(f"----->{question_with_answers.text}")
        for ans in question_with_answers.answers:
            logger.info(ans.text)
            logger.info(ans.confidence)
        logger.info(f"------------------------")

    them_asp.reset_questions(questions_with_answers)

    ans_agg = AnswerAggregator()
    fin_ans = None

    if agg_mode == AggMode.NO_AGG:

        fin_ans = ans_agg.aggregate_without_clustering(them_asp)

    elif agg_mode == AggMode.CLUST:

        fin_ans = ans_agg.aggregate_with_clustering(them_asp)


    return {
        "used_chunks": text_chunks,
        "final_answer": fin_ans,
        "thematic_aspect": them_asp
    }


def extract_keywords(them_asp: ThematicAspect,
                    reference_answer: str) -> dict:

    validator = AnswerValidator()
    answers = them_asp.get_all_answers()

    result = validator.process(answers, reference_answer)

    logger.info("---\n---> Keyword extraction result:")
    logger.info(result)

    return result

def keys_extracted(result: dict) -> bool:

    if result["analysis"] is None:
        return False
    else:
        return True

def filter_keywords(result: dict, 
                    them_asp:ThematicAspect,
                    similarity_threshold: float = similarity_threshold,
                    dissimilarity_threshold: float = dissimilarity_threshold) -> List[str]:

    keywords = result["analysis"]["keywords"]
    reference = them_asp.name
    antireference = result["reference_answer"]
    
    
    sentence_tokenizer = SentenceTokenizerSingleton()
    
    # Получаем эмбеддинги
    reference_embedding = sentence_tokenizer.tokenizer.encode(reference, convert_to_tensor=True)
    antireference_embedding = sentence_tokenizer.tokenizer.encode(antireference, convert_to_tensor=True)
    keyword_embeddings = sentence_tokenizer.tokenizer.encode(keywords, convert_to_tensor=True)
    
    # Нормализуем векторы
    ref_norm = torch.nn.functional.normalize(reference_embedding, p=2, dim=0)
    antiref_norm = torch.nn.functional.normalize(antireference_embedding, p=2, dim=0)
    keywords_norm = torch.nn.functional.normalize(keyword_embeddings, p=2, dim=1)
    
    # Косинусное сходство
    ref_similarities = torch.mm(keywords_norm, ref_norm.unsqueeze(1)).flatten()
    antiref_similarities = torch.mm(keywords_norm, antiref_norm.unsqueeze(1)).flatten()
    
    ## Фильтрация
    #mask = (ref_similarities >= similarity_threshold) & (antiref_similarities <= dissimilarity_threshold)
    #return [kw for kw, m in zip(keywords, mask) if m]

    # Преобразуем результаты в numpy arrays для удобства
    sim_pos_np = ref_similarities.cpu().numpy().flatten()
    sim_neg_np = antiref_similarities.cpu().numpy().flatten()

    # Находим индексы слов, которые больше похожи на пример
    selected_indices = np.where(sim_pos_np > sim_neg_np)[0]

    # Формируем результат: слово и разность сходств (для ранжирования)
    results = []
    for idx in selected_indices:
        word = keywords[idx]
        score_diff = sim_pos_np[idx] - sim_neg_np[idx]
        results.append((word, score_diff))

    # Сортируем по убыванию разности сходств
    results.sort(key=lambda x: x[1], reverse=True)

    return results


#----------------------------------------------------

def filter_keywords_from_dict(keywords_with_weights: List[dict], 
                    them_asp: ThematicAspect,
                    antireference: str,
                    similarity_threshold: float = similarity_threshold,
                    dissimilarity_threshold: float = dissimilarity_threshold) -> List[dict]:
    
    # Извлекаем слова из словарей
    keywords = [item['word'] for item in keywords_with_weights]
    reference = them_asp.name
    
    sentence_tokenizer = SentenceTokenizerSingleton()
    
    # Получаем эмбеддинги
    reference_embedding = sentence_tokenizer.tokenizer.encode(reference, convert_to_tensor=True)
    antireference_embedding = sentence_tokenizer.tokenizer.encode(antireference, convert_to_tensor=True)
    keyword_embeddings = sentence_tokenizer.tokenizer.encode(keywords, convert_to_tensor=True)
    
    # Нормализуем векторы
    ref_norm = torch.nn.functional.normalize(reference_embedding, p=2, dim=0)
    antiref_norm = torch.nn.functional.normalize(antireference_embedding, p=2, dim=0)
    keywords_norm = torch.nn.functional.normalize(keyword_embeddings, p=2, dim=1)
    
    # Косинусное сходство
    ref_similarities = torch.mm(keywords_norm, ref_norm.unsqueeze(1)).flatten()
    antiref_similarities = torch.mm(keywords_norm, antiref_norm.unsqueeze(1)).flatten()
    
    # Преобразуем результаты в numpy arrays для удобства
    sim_pos_np = ref_similarities.cpu().numpy().flatten()
    sim_neg_np = antiref_similarities.cpu().numpy().flatten()

    # Находим индексы слов, которые больше похожи на пример
    selected_indices = np.where(sim_pos_np > sim_neg_np)[0]

    # Формируем результат: добавляем score_diff к существующим словарям
    results = []
    for idx in selected_indices:
        # Копируем исходный словарь и добавляем score_diff
        keyword_dict = keywords_with_weights[idx].copy()
        keyword_dict['score_diff'] = sim_pos_np[idx] - sim_neg_np[idx]
        results.append(keyword_dict)

    # Сортируем по убыванию разности сходств
    results.sort(key=lambda x: x['score_diff'], reverse=True)

    return results

def extra_score_chunks_by_keywords(
    chunks: List[TextChunk], 
    keywords: List[dict],
    case_sensitive: bool = False,
    use_combined_weights: bool = True,
    weight_ratio: float = 0.5,  # Соотношение между weight и score_diff (0.5 = равное значение)
    min_matches: int = 1
) -> List[ScoredChunk]:
    """
    Оценивает фрагменты на основе количества и важности ключевых слов.
    Учитывает оба веса: weight (из модели внимания) и score_diff (из сравнения с эталонами).
    
    Параметры:
        chunks: Список объектов TextChunk для оценки
        keywords: Список словарей формата [{"word": str, "lemma": str, "stem": str, 
                 "weight": float, "score_diff": float}, ...]
        case_sensitive: Учитывать регистр при поиске
        use_combined_weights: Использовать комбинированные веса (weight и score_diff)
        weight_ratio: Соотношение между weight и score_diff (0.0 - только score_diff, 1.0 - только weight)
        min_matches: Минимальное количество совпадений для включения фрагмента в результат
    
    Возвращает:
        Список объектов ScoredChunk с оценками и информацией о совпадениях
    """
    if not keywords or not chunks:
        return []
    
    # Подготавливаем ключевые слова и их комбинированные веса
    keyword_weights = {}
    search_terms = set()
    term_to_keyword = {}
    
    for keyword in keywords:
        word = keyword['word']
        
        # Вычисляем комбинированный вес
        weight = keyword.get('weight', 1.0)
        score_diff = keyword.get('score_diff', 1.0)
        
        if use_combined_weights:
            # Комбинируем веса с учетом соотношения
            combined_weight = (weight * weight_ratio) + (score_diff * (1 - weight_ratio))
        else:
            # Используем только score_diff (по умолчанию)
            combined_weight = score_diff
        
        keyword_weights[word] = combined_weight
        
        # Добавляем лемму и стем для поиска
        lemma = keyword.get('lemma', word)
        stem = keyword.get('stem', word)
        
        search_terms.add(lemma)
        search_terms.add(stem)
        
        term_to_keyword[lemma.lower()] = word
        term_to_keyword[stem.lower()] = word
    
    # Создаем regex-паттерн для поиска
    pattern = re.compile(
        r'\b(?:{})\b'.format('|'.join(map(re.escape, search_terms))),
        flags=0 if case_sensitive else re.IGNORECASE
    )
    
    scored_chunks = []
    
    for chunk in chunks:
        # Находим все совпадения терминов в тексте
        text_to_search = chunk.text if case_sensitive else chunk.text.lower()
        found_terms = pattern.findall(text_to_search)
        
        # Собираем уникальные найденные ключевые слова и их веса
        matched_keywords = set()
        total_score = 0.0
        
        for term in found_terms:
            normalized_term = term if case_sensitive else term.lower()
            if normalized_term in term_to_keyword:
                keyword_word = term_to_keyword[normalized_term]
                if keyword_word not in matched_keywords:
                    matched_keywords.add(keyword_word)
                    total_score += keyword_weights.get(keyword_word, 1.0)
        
        # Применяем дополнительные факторы для улучшения оценки
        if matched_keywords:
            # Нормализуем по количеству слов в фрагменте
            normalization_factor = 1.0 + (len(matched_keywords) / max(1, chunk.word_token_count / 10))
            
            # Учитываем плотность ключевых слов
            density_bonus = min(2.0, 1.0 + (len(matched_keywords) / max(1, len(chunk.text.split()) / 20)))
            
            final_score = total_score * normalization_factor * density_bonus
            
            if len(matched_keywords) >= min_matches:
                scored_chunks.append(ScoredChunk(
                    chunk=chunk,
                    score=final_score,
                    matched_keywords=list(matched_keywords),
                    keyword_scores={kw: keyword_weights.get(kw, 1.0) for kw in matched_keywords},
                    # Добавляем информацию об исходных весах
                    original_weights={kw: {
                        'weight': next((k['weight'] for k in keywords if k['word'] == kw), 1.0),
                        'score_diff': next((k['score_diff'] for k in keywords if k['word'] == kw), 1.0)
                    } for kw in matched_keywords}
                ))
    
    # Сортируем по убыванию оценки
    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    
    return scored_chunks

# Расширенная версия с учетом обоих весов
def extra_score_chunks_advanced(
    chunks: List[TextChunk], 
    keywords: List[dict],
    case_sensitive: bool = False,
    min_matches: int = 1,
    position_weight: float = 0.3,
    frequency_weight: float = 0.7,
    weight_ratio: float = 0.5  # Соотношение между weight и score_diff
) -> List[ScoredChunk]:
    """
    Расширенная версия оценки фрагментов с учетом позиции и частоты ключевых слов.
    Учитывает оба веса: weight и score_diff.
    """
    if not keywords or not chunks:
        return []
    
    # Подготавливаем ключевые слова с комбинированными весами
    keyword_weights = {}
    search_terms = set()
    term_to_keyword = {}
    
    for keyword in keywords:
        word = keyword['word']
        
        # Комбинируем веса
        weight = keyword.get('weight', 1.0)
        score_diff = keyword.get('score_diff', 1.0)
        combined_weight = (weight * weight_ratio) + (score_diff * (1 - weight_ratio))
        
        keyword_weights[word] = combined_weight
        
        lemma = keyword.get('lemma', word)
        stem = keyword.get('stem', word)
        
        search_terms.add(lemma)
        search_terms.add(stem)
        
        term_to_keyword[lemma.lower()] = word
        term_to_keyword[stem.lower()] = word
    
    pattern = re.compile(
        r'\b(?:{})\b'.format('|'.join(map(re.escape, search_terms))),
        flags=0 if case_sensitive else re.IGNORECASE
    )
    
    scored_chunks = []
    
    for chunk in chunks:
        text_to_search = chunk.text if case_sensitive else chunk.text.lower()
        found_matches = list(pattern.finditer(text_to_search))
        
        if not found_matches:
            continue
        
        # Анализируем совпадения
        matched_keywords = set()
        position_scores = []
        keyword_frequencies = {}
        
        for match in found_matches:
            term = match.group()
            normalized_term = term if case_sensitive else term.lower()
            
            if normalized_term in term_to_keyword:
                keyword_word = term_to_keyword[normalized_term]
                matched_keywords.add(keyword_word)
                
                # Оценка позиции (слова в начале текста более важны)
                position = match.start() / max(1, len(text_to_search))
                position_score = 1.0 - position  # чем раньше, тем выше оценка
                position_scores.append(position_score)
                
                # Подсчет частоты
                keyword_frequencies[keyword_word] = keyword_frequencies.get(keyword_word, 0) + 1
        
        if len(matched_keywords) >= min_matches:
            # Вычисляем общую оценку с учетом комбинированных весов
            base_score = sum(keyword_weights.get(kw, 1.0) for kw in matched_keywords)
            
            # Учитываем позицию (средняя оценка позиций)
            avg_position_score = np.mean(position_scores) if position_scores else 0.5
            position_contribution = 1.0 + position_weight * avg_position_score
            
            # Учитываем частоту (логарифмическая шкала для избежания перекоса)
            freq_contribution = 1.0 + frequency_weight * np.log1p(sum(keyword_frequencies.values()))
            
            # Учитываем уникальность (количество разных ключевых слов)
            uniqueness_bonus = 1.0 + (len(matched_keywords) / len(keywords)) * 0.5
            
            final_score = base_score * position_contribution * freq_contribution * uniqueness_bonus
            
            scored_chunks.append(ScoredChunk(
                chunk=chunk,
                score=final_score,
                matched_keywords=list(matched_keywords),
                keyword_scores=keyword_frequencies,
                # Добавляем информацию об исходных весах
                original_weights={kw: {
                    'weight': next((k['weight'] for k in keywords if k['word'] == kw), 1.0),
                    'score_diff': next((k['score_diff'] for k in keywords if k['word'] == kw), 1.0),
                    'combined_weight': keyword_weights.get(kw, 1.0)
                } for kw in matched_keywords}
            ))
    
    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    
    return scored_chunks