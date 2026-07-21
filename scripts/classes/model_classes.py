#---------------------------------------------------- 
#       Загрузка путей

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

#---------------------------------------------------- 

from functions.functions import *

from models_processing.models_setting import *
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import cosine_similarity
import re

#---------------------------------------------------- 
#       Подключение логгера

from logs_processing.logger_config import setup_logger

logger = setup_logger(__name__, "model_classes.log")

#---------------------------------

#------------------------------------
# --- Параметры

from config.config import *

CHUNK_MAX_TOK = CHUNKING['chunk_max_tok']
OVERLAP_TOK = CHUNKING['overlap_tok']

ANS_CLUSTER_THRESHOLD = ANSWERS_AGG['answers_cluster_threshold']

STRICT_THRESHOLD = ANSWERS_VAL['strict_threshold']
MIN_THRESHOLD = ANSWERS_VAL['min_threshold']

IDF_THRESHOLD = KEYWORDS_EXTRACTION['idf_threshold']

#------------------------------------

class SentenceTokenizerSingleton:
    _instance = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tokenizer = get_concrete_sentence_tokenizer()[0]
        return cls._instance
    
    @property
    def tokenizer(self):
        return self._tokenizer
#------------------------------------

class BERTWordTokenizerSingleton:
    _instance = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tokenizer = get_concrete_tokenizer()
        return cls._instance
    
    @property
    def tokenizer(self):
        return self._tokenizer
#------------------------------------

class BERTEngQASingleton:
    _instance = None
    _tokenizer = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tokenizer = get_concrete_tokenizer()
            cls._model = get_concrete_QA_model()
        return cls._instance
    
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model
#------------------------------------

@dataclass
class Sentence:
    text: str
    num: int
    # word_tokens: list[str]
    # word_embeddings: list[int]
    # word_token_count: int
    # sent_embedding: list[float]
    
    def __post_init__(self, 
                      word_tokenizer = BERTWordTokenizerSingleton(),
                      sentence_tokenizer = SentenceTokenizerSingleton()):

        self.word_tokens = word_tokenizer.tokenizer.tokenize(self.text)
        self.word_embeddings = word_tokenizer.tokenizer.encode(self.text)
        self.word_token_count = len(self.word_tokens)
        self.sent_embedding = sentence_tokenizer.tokenizer.encode(self.text)

    def __str__(self):
        return f'''
        ---> Sentence:
        text='{self.text}',
        word_tokens={self.word_tokens},
        word_token_count={self.word_token_count},
        num={self.num}'''
#------------------------------------

@dataclass
class TextChunk:
    sentences: list[Sentence]
    text: str = None
    word_token_count: int = None
    
    def __post_init__(self, sentence_tokenizer = SentenceTokenizerSingleton()):

        self.text = " ".join([s.text for s in self.sentences])
        self.word_token_count = sum(s.word_token_count for s in self.sentences)
        self.word_tokens = [s.word_tokens for s in self.sentences]

        sentence_word_embeddings = [s.word_embeddings for s in self.sentences]
        sentence_word_embeddings = [word_embeddings[1:-1] for word_embeddings in sentence_word_embeddings]
        self.word_embeddings = [101] + [element for each_list in sentence_word_embeddings for element in each_list] + [102]

        self.sent_embeddings = [s.sent_embedding for s in self.sentences]
        self.full_text_embedding = sentence_tokenizer.tokenizer.encode(self.text)

        self.sentences_nums = sorted([s.num for s in self.sentences])
        self.start_sentence_num = self.sentences_nums[0]
        self.end_sentence_num = self.sentences_nums[-1]
    
    def __str__(self):
        return f'''TextChunk:
        sentences={" ".join([str(s) for s in self.sentences])},
        text='{self.text}',
        word_token_count={self.word_token_count},
        sentences_nums={self.sentences_nums}'''

    def object_print(self):
        print(f"{self.__class__.__name__}:")
        for k, v in self.__dict__.items():
            print(f"  {k}: {v}")
#------------------------------------

@dataclass
class ChunkState:
    chunks: List[TextChunk] = None
    current_chunk: List[Sentence] = None
    current_token_count: int = 0
    overlap_buffer: List[Sentence] = None
    overlap_token_count: int = 0

    def __post_init__(self):
        self.chunks = self.chunks or []
        self.current_chunk = self.current_chunk or []
        self.overlap_buffer = self.overlap_buffer or []
#------------------------------------

class ChunkBuilder:
    def __init__(self, word_tokenizer = BERTWordTokenizerSingleton(), 
                       max_tokens: int = CHUNK_MAX_TOK, 
                       overlap_tokens: int = OVERLAP_TOK,
                       long_sentence_strategy: str = "split"):

        self.tokenizer = word_tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.long_strat = long_sentence_strategy
        
        if self.overlap_tokens >= self.max_tokens // 2:
            raise ValueError("Перекрытие должно быть меньше половины max_tokens")

    def _process_long_sentence(self, sent: Sentence) -> List[Sentence]:
        """Обработка предложения длиннее max_tokens"""
        
        if self.long_strat == "truncate":

            # обрезаем лишнее
            truncated_tokens = sent.word_tokens[:self.max_tokens - 2]
            truncated_text = self.tokenizer.tokenizer.convert_tokens_to_string(truncated_tokens)
            return [Sentence(truncated_text, sent.num)]
        
        elif self.long_strat == "split":
            
            # делим пополам
            half = len(sent.word_tokens) // 2
            part1 = self.tokenizer.tokenizer.convert_tokens_to_string(sent.word_tokens[:half])
            part2 = self.tokenizer.tokenizer.convert_tokens_to_string(sent.word_tokens[half:])
            return [
                Sentence(part1, sent.num),
                Sentence(part2, sent.num)
            ]

    def build_chunks(self, sentences: List[Sentence]) -> List[TextChunk]:
        
        state = ChunkState()  # Создаем контейнер состояния
        
        for sent in sentences:
            sent_tokens = sent.word_tokens
            sent_token_count = sent.word_token_count
            
            # Обработка очень длинных предложений
            if sent_token_count > self.max_tokens:
                processed = self._process_long_sentence(sent)
                for processed_sent in processed:
                    self._add_to_chunk(processed_sent, state)
                continue
            
            # Обычное предложение
            self._add_to_chunk(sent, state)
        
        # Добавляем последний чанк
        if state.current_chunk:
            state.chunks.append(state.current_chunk.copy())

        result_text_chunks = []
        for chunk in state.chunks:
            result_text_chunks.append(TextChunk(chunk))
            
        return result_text_chunks

    def _add_to_chunk(self, sent: Sentence, state: ChunkState):

        str = sent.text
        token_count = sent.word_token_count
        
        if state.current_token_count + token_count <= self.max_tokens:

            state.current_chunk.append(sent)
            state.current_token_count += token_count
            
            state.overlap_buffer.append(sent)
            state.overlap_token_count += token_count
           
            # Управление буфером перекрытия
            while state.overlap_token_count > self.overlap_tokens and len(state.overlap_buffer) > 1:
                removed_sentence = state.overlap_buffer.pop(0)
                removed_tokens = removed_sentence.word_tokens
                state.overlap_token_count -= len(removed_tokens)
        else:

            if state.current_chunk:
                state.chunks.append(state.current_chunk.copy())
            
            state.current_chunk = state.overlap_buffer.copy()
            state.current_token_count = state.overlap_token_count
            
            if token_count <= self.max_tokens:
                state.current_chunk.append(sent)
                state.current_token_count += token_count
            
            state.overlap_buffer = [sent]
            state.overlap_token_count = token_count
#------------------------------------

@dataclass
class Answer:
    text: str
    chunk: TextChunk
    confidence: float
    similarity_score = None
    start_pos: Optional[int] = None  
    end_pos: Optional[int] = None

    def attach_question(self, question: "Question"):
        self.question = question     
#------------------------------------

@dataclass
class Question:
    text: str
    answers: List[Answer] = field(default_factory=list) # По умолчанию создает пустой список для каждого экземпляра класса
    
    def __post_init__(self, sentence_tokenizer = SentenceTokenizerSingleton()):
        self.text_embedding = sentence_tokenizer.tokenizer.encode(self.text)

    def clear_answers(self):
        self.answers = []

#------------------------------------

class SimpleAnswerFinder:

    def __init__(self, qa_model = BERTEngQASingleton()):

        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model.model,
            tokenizer=qa_model.tokenizer,
            device=get_device()
        )

    def find_answers(self, 
                    question: Question, 
                    chunks: List[TextChunk], 
                    parallel: bool = False,
                    workers: int = 4) -> Question:
        
        if parallel:
            return self._find_answers_parallel(question, chunks, workers)
        return self._find_answers_sequential(question, chunks)
    
    def _find_answers_sequential(self, question: Question, chunks: List[TextChunk]) -> Question:
        for chunk in chunks:
            result = self._process_chunk(question.text, chunk)
            if result:
                question.answers.append(result)
        question.answers.sort(key=lambda x: x.confidence, reverse=True)
        return question
    
    def _find_answers_parallel(self, 
                             question: Question, 
                             chunks: List[TextChunk],
                             workers: int) -> Question:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                lambda c: self._process_chunk(question.text, c), 
                chunks
            ))
        question.answers = [r for r in results if r is not None]
        question.answers.sort(key=lambda x: x.confidence, reverse=True)
        return question
    
    def _process_chunk(self, question: str, chunk: TextChunk) -> Optional[Answer]:
        try:
            result = self.qa_pipeline(question=question, context=chunk.text)
            answer = Answer(
                text=result["answer"],
                chunk=chunk,
                confidence=result["score"],
                start_pos=result["start"],
                end_pos=result["end"]
            )
            answer.attach_question(question)
            return answer
        except Exception as e:
            print(f"Error processing chunk from source {chunk.source}: {e}")
            return None
#------------------------------------
@dataclass
class ScoredChunk:
    chunk: TextChunk
    score: float
    matched_keywords: List[str]
    keyword_scores: Dict[str, float]
    original_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ScoredAnswerFinder(SimpleAnswerFinder):
    """
    Наследник SimpleAnswerFinder, который работает с ScoredChunk и сортирует ответы по score фрагментов
    """

    def __init__(self, qa_model=BERTEngQASingleton()):
        super().__init__(qa_model)

    def find_answers(self, 
                    question: Question, 
                    scored_chunks: List[ScoredChunk], 
                    parallel: bool = False,
                    workers: int = 4) -> Question:
        """
        Находит ответы в scored chunks и сортирует их по score фрагментов
        
        Args:
            question: Вопрос для поиска ответов
            scored_chunks: Список ScoredChunk объектов с оцененными фрагментами
            parallel: Использовать многопоточность
            workers: Количество worker'ов для многопоточности
            
        Returns:
            Question с отсортированными ответами
        """
        if parallel:
            return self._find_answers_parallel(question, scored_chunks, workers)
        return self._find_answers_sequential(question, scored_chunks)
    
    def _find_answers_sequential(self, question: Question, scored_chunks: List[ScoredChunk]) -> Question:
        """Последовательная обработка scored chunks"""
        for scored_chunk in scored_chunks:
            result = self._process_scored_chunk(question.text, scored_chunk)
            if result:
                question.answers.append(result)
        
        # Сортируем ответы по score фрагментов (в порядке убывания)
        question.answers.sort(key=lambda x: x.chunk_score, reverse=True)
        return question
    
    def _find_answers_parallel(self, 
                             question: Question, 
                             scored_chunks: List[ScoredChunk],
                             workers: int) -> Question:
        """Многопоточная обработка scored chunks"""
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                lambda sc: self._process_scored_chunk(question.text, sc), 
                scored_chunks
            ))
        
        question.answers = [r for r in results if r is not None]
        
        # Сортируем ответы по score фрагментов (в порядке убывания)
        question.answers.sort(key=lambda x: x.chunk_score, reverse=True)
        return question
    
    def _process_scored_chunk(self, question: str, scored_chunk: ScoredChunk) -> Optional[Answer]:
        """
        Обрабатывает один scored chunk и создает ответ с информацией о score
        
        Args:
            question: Текст вопроса
            scored_chunk: ScoredChunk с фрагментом и его оценкой
            
        Returns:
            Answer объект с дополнительным полем chunk_score или None в случае ошибки
        """
        try:
            # Используем унаследованный метод для обработки QA
            result = self.qa_pipeline(question=question, context=scored_chunk.chunk.text)
            
            # Создаем ответ с дополнительной информацией о score фрагмента
            answer = Answer(
                text=result["answer"],
                chunk=scored_chunk,  # Оригинальный TextChunk
                confidence=result["score"],
                start_pos=result["start"],
                end_pos=result["end"]
            )
            
            # Добавляем информацию о score фрагмента
            answer.chunk_score = scored_chunk.score
            answer.matched_keywords = scored_chunk.matched_keywords
            answer.keyword_scores = scored_chunk.keyword_scores
            
            answer.attach_question(question)
            return answer
            
        except Exception as e:
            print(f"Error processing scored chunk from source {scored_chunk.chunk.source}: {e}")
            return None

    def find_and_rank_answers(self,
                             question: Question,
                             scored_chunks: List[ScoredChunk],
                             parallel: bool = False,
                             workers: int = 4,
                             combine_scores: bool = True) -> Question:
        """
        Расширенная версия с комбинированной сортировкой по confidence QA модели и score фрагмента
        
        Args:
            combine_scores: Если True, комбинирует confidence и chunk_score для финальной сортировки
        """
        # Сначала находим все ответы
        question = self.find_answers(question, scored_chunks, parallel, workers)
        
        if combine_scores and question.answers:
            # Комбинируем оценку: confidence QA модели * score фрагмента
            for answer in question.answers:
                answer.combined_score = answer.confidence * answer.chunk_score
            
            # Сортируем по комбинированной оценке
            question.answers.sort(key=lambda x: x.combined_score, reverse=True)
        
        return question


# Дополнительно: расширяем класс Answer для поддержки новых полей
class EnhancedAnswer(Answer):
    """Расширенный класс Answer с дополнительными полями для scored chunks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_score = kwargs.get('chunk_score', 0.0)
        self.combined_score = kwargs.get('combined_score', 0.0)
        self.matched_keywords = kwargs.get('matched_keywords', [])
        self.keyword_scores = kwargs.get('keyword_scores', {})
#------------------------------------

@dataclass
class ThematicAspect:
    name: str  # Название аспекта
    questions: List[Question] = field(default_factory=list)

    def get_all_answers(self)-> List[Answer]:
        return [ans for q in self.questions for ans in q.answers]

    def reset_questions(self, new_questions) -> None:
        self.questions.clear()
        self.questions = new_questions


#------------------------------------

@dataclass
class FinalAnswer:
    text: str
    confidence: float
    supporting_answers: List[Answer] = field(default_factory=list)
#------------------------------------

@dataclass
class AnswerCluster:
    answers: List[Answer]
    centroid_embedding: np.ndarray
    confidence: float

    def __str__(self):
        return f'''
        AnswerCluster:
           answers={" ; ".join([f"{a.text}" for a in self.answers])},
           confidence={self.confidence}
        '''
#------------------------------------

class AnswerAggregator:

    def __init__(self, cluster_threshold: float = ANS_CLUSTER_THRESHOLD,
                    sentence_tokenizer = SentenceTokenizerSingleton()):

        self.sentence_tokenizer = sentence_tokenizer
        self.cluster_threshold = cluster_threshold

    def cluster_answers(self,
                    answers: List[Answer]
                    ) -> List[AnswerCluster]:

        if not answers:
            return []

        # Получаем эмбеддинги для всех ответов
        texts = [ans.text for ans in answers]
        embeddings = self.sentence_tokenizer.tokenizer.encode(texts)
        
        # Кластеризация (порог в косинусной близости)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity='cosine',
            linkage='average',
            distance_threshold=self.cluster_threshold
        ).fit(embeddings)
        
        # Формируем кластеры
        clusters = []
        for cluster_id in np.unique(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_answers = [answers[i] for i in cluster_indices]
            centroid = np.mean(embeddings[cluster_indices], axis=0)
            conf = np.max([ans.confidence for ans in cluster_answers])
            
            clusters.append(AnswerCluster(
                answers=cluster_answers.copy(),
                centroid_embedding=centroid,
                confidence=conf
            ))
        
        return clusters

    def select_best_cluster(self, clusters: List[AnswerCluster]) -> Optional[AnswerCluster]:
        if not clusters:
            return None
        # Выбираем кластер с наибольшей средней уверенностью
        return max(clusters, key=lambda x: x.confidence)

    def find_most_representative(self, cluster: AnswerCluster) -> Answer:

        # Находим ответ, ближайший к центроиду кластера
        answer_embeddings = self.sentence_tokenizer.tokenizer.encode(
                                            [ans.text for ans in cluster.answers])
        distances = np.linalg.norm(answer_embeddings - cluster.centroid_embedding, axis=1)
        best_idx = np.argmin(distances)
        return cluster.answers[best_idx]

    def aggregate_with_clustering(self,
                            aspect: ThematicAspect
                            ) -> Optional[FinalAnswer]:

        # Собираем все ответы по аспекту
        all_answers = [ans for q in aspect.questions for ans in q.answers]
        
        # Кластеризуем
        self.clusters = self.cluster_answers(all_answers)
        if self.clusters:
            print("Выделенные кластеры: ")
            for cluster in self.clusters:
                print(str(cluster))
        best_cluster = self.select_best_cluster(self.clusters)
        
        if not best_cluster:
            return None
        
        # Выбираем репрезентативный ответ
        representative = self.find_most_representative(best_cluster)
        
        return FinalAnswer(
            text=representative.text,
            confidence=best_cluster.confidence,
            supporting_answers=best_cluster.answers
        )

    def aggregate_without_clustering(self,
                            aspect: ThematicAspect
                            ) -> Optional[FinalAnswer]:

        # Собираем все ответы по аспекту
        all_answers = [ans for q in aspect.questions for ans in q.answers]

        if not all_answers:
            return None

        # Векторизация ответов
        texts = [ans.text for ans in all_answers]
        embeddings = self.sentence_tokenizer.tokenizer.encode(texts, convert_to_tensor=True)
        
        # Расчет попарных сходств
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.T)

        # Находим "центральный" ответ (с max средним сходством)
        mean_similarities = torch.mean(sim_matrix, dim=1)  # Среднее сходство каждого ответа с остальными
        best_idx = torch.argmax(mean_similarities).item()
        best_answer = all_answers[best_idx]

        # Подсчет метрик
        avg_confidence = np.mean([ans.confidence for ans in all_answers])
        supporting_answers = [ans for ans in all_answers if ans.text == best_answer.text]

        return FinalAnswer(
            text=best_answer.text,
            confidence=avg_confidence,
            supporting_answers=supporting_answers
        )


#------------------------------------

class ChunkFilter:

    def __init__(self, filter_method: str = None):
        self.filter_method = "simple cosine"


    def filter_chunks(self,
                    reference_text: "Question",
                    chunks: List[TextChunk]) -> List[TextChunk]:
        print("v")
        print(self.filter_method)
        print(self.filter_method == "simple cosine")

        if self.filter_method == "simple cosine":

            print("a")
            '''
            question_embedding = torch.from_numpy(reference_text.text_embedding) 
            question_embedding.to(get_device())
            text_embeddings = torch.tensor([torch.from_numpy(chunk.full_text_embedding).to(get_device()) for chunk in chunks])
            '''
            question_embedding = torch.tensor(reference_text.text_embedding)
            text_embeddings = torch.tensor([chunk.full_text_embedding for chunk in chunks])

            similarities = cosine_similarity(
                question_embedding.unsqueeze(0),  
                text_embeddings                   
                ).numpy()

            print(similarities)

            threshold = 0.3
            relevant_indices = (similarities >= threshold).nonzero()[0]
            relevant_chunks = [(chunks[i], similarities[i]) for i in relevant_indices]

            return relevant_chunks

    def filter_chunks_by_keywords(self, 
                                chunks: List[TextChunk], 
                                keywords: List[str],
                                case_sensitive: bool = False,
                                min_keywords: int = 3) -> List[TextChunk]:
        """
        Фильтрует чанки, оставляя только те, которые содержат хотя бы одно ключевое слово
        
        Параметры:
            chunks: Список объектов TextChunk для фильтрации
            keywords: Список ключевых слов для поиска
            case_sensitive: Учитывать регистр при поиске (по умолчанию False)
            
        Возвращает:
            Список чанков, содержащих хотя бы одно ключевое слово
        """
        if not keywords or not chunks or min_keywords < 1:
            return []
            
        # Создаем regex-паттерн для поиска
        pattern = re.compile(
            r'\b(?:{})\b'.format('|'.join(map(re.escape, keywords))),
            flags=0 if case_sensitive else re.IGNORECASE
        )
        
        # Фильтруем чанки
        filtered_chunks = []
        for chunk in chunks:
            matches = pattern.findall(chunk.text)
            unique_matches = set(matches)  # Убираем дубликаты
            if len(unique_matches) >= min_keywords:
                filtered_chunks.append(chunk)
                
        return filtered_chunks

    def filter_chunks_by_keywords_lemm_stemm(self, 
                                    chunks: List[TextChunk], 
                                    keywords: List[dict],
                                    min_keywords: int = 1,
                                    case_sensitive: bool = False) -> List[TextChunk]:
        """
        Фильтрует чанки, оставляя только те, которые содержат ключевые слова.
        Совпадение считается, если найдено либо по лемме, либо по стему слова.
        
        Параметры:
            chunks: Список объектов TextChunk для фильтрации
            keywords: Список словарей формата [{"word": str, "lemma": str, "stem": str}, ...]
            case_sensitive: Учитывать регистр при поиске (по умолчанию False)
            min_keywords: Минимальное количество уникальных слов для совпадения
            
        Возвращает:
            Список чанков, содержащих достаточно ключевых слов
        """
        if not keywords or not chunks or min_keywords < 1:
            return []
        
        # Собираем все варианты (леммы и стемы) для поиска
        search_terms = set()
        for keyword in keywords:
            search_terms.add(keyword['lemma'])
            search_terms.add(keyword['stem'])
        
        # Создаем regex-паттерн
        pattern = re.compile(
            r'\b(?:{})\b'.format('|'.join(map(re.escape, search_terms))),
            flags=0 if case_sensitive else re.IGNORECASE
        )
        
        # Словарь для быстрого поиска соответствий
        term_to_word = {}
        for keyword in keywords:
            term_to_word[keyword['lemma'].lower()] = keyword['word']
            term_to_word[keyword['stem'].lower()] = keyword['word']
        
        filtered_chunks = []
        for chunk in chunks:
            # Находим все совпадения терминов в тексте
            found_terms = pattern.findall(chunk.text.lower() if not case_sensitive else chunk.text)
            
            # Преобразуем найденные термины в оригинальные слова
            found_words = set()
            for term in found_terms:
                normalized_term = term if case_sensitive else term.lower()
                if normalized_term in term_to_word:
                    found_words.add(term_to_word[normalized_term])
            
            # Проверяем количество уникальных найденных слов
            if len(found_words) >= min_keywords:
                filtered_chunks.append(chunk)

        return filtered_chunks
#------------------------------------

@dataclass
class AnswerValidator:

    similarity_threshold: float = 0.90

    def __post_init__(self, 
                      sentence_tokenizer = SentenceTokenizerSingleton()):

        self.sentence_tokenizer = sentence_tokenizer
    '''
    def validate_answers(self, 
                       answers: List[Answer], 
                       reference_answer: str) -> List[Answer]:
        
        if not answers:
            return []
            
        # Векторизация ответов и эталона
        answer_texts = [ans.text for ans in answers]
        embeddings = self.sentence_tokenizer.tokenizer.encode(answer_texts, convert_to_tensor=True)  # Получаем тензор
        ref_embedding = self.sentence_tokenizer.tokenizer.encode(reference_answer, convert_to_tensor=True)
        
        # Нормализация эмбеддингов (обязательно для корректного косинусного сходства)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        ref_embedding = torch.nn.functional.normalize(ref_embedding.unsqueeze(0), p=2, dim=1)
        
        # Расчет косинусного сходства (матричное умножение)
        similarities = torch.mm(ref_embedding, embeddings.T)[0]  # shape: [num_answers]
        
        
        # Фильтрация по порогу
        valid_indices = (similarities >= self.similarity_threshold).nonzero().flatten()
        valid_answers = [answers[i] for i in valid_indices]
        
        return valid_answers

    '''
    def validate_answers(self, 
                   answers: List[Answer], 
                   reference_answer: str,
                   strict_threshold: float = STRICT_THRESHOLD,
                   min_threshold: float = MIN_THRESHOLD,
                   top_k: int = 1) -> List[Answer]:
        """
        Валидирует ответы по отношению к эталону с использованием гибридного подхода.
        
        Args:
            answers: Список ответов-кандидатов
            reference_answer: Эталонный ответ
            strict_threshold: Строгий порог сходства (по умолчанию 0.9)
            min_threshold: Минимальный допустимый порог (по умолчанию 0.7)
            top_k: Сколько ответов возвращать, если нет проходящих strict_threshold
        
        Returns:
            Список валидных ответов с добавленным полем similarity_score
        """
        if not answers:
            return []
            
        # Векторизация ответов и эталона
        answer_texts = [ans.text for ans in answers]
        embeddings = self.sentence_tokenizer.tokenizer.encode(answer_texts, convert_to_tensor=True)
        ref_embedding = self.sentence_tokenizer.tokenizer.encode(reference_answer, convert_to_tensor=True)
        
        # Нормализация эмбеддингов
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        ref_embedding = torch.nn.functional.normalize(ref_embedding.unsqueeze(0), p=2, dim=1)
        
        # Расчет косинусного сходства
        similarities = torch.mm(ref_embedding, embeddings.T)[0].cpu().numpy()
        
        # Применяем гибридную стратегию
        valid_answers = []
        
        # 1. Сначала ищем ответы, превышающие строгий порог
        strict_indices = np.where(similarities >= strict_threshold)[0]
        if len(strict_indices) > 0:
            for idx in strict_indices:
                answers[idx].similarity_score = float(similarities[idx])
                valid_answers.append(answers[idx])
            return valid_answers
        
        # 2. Если строгих совпадений нет, берем топ-N по минимальному порогу
        # Создаем список пар (индекс, score) и сортируем по убыванию
        scored_answers = [(i, score) for i, score in enumerate(similarities)]
        scored_answers.sort(key=lambda x: x[1], reverse=True)
        
        # Фильтруем по минимальному порогу и берем топ-K
        filtered = [x for x in scored_answers if x[1] >= min_threshold]

        if len(filtered) == 0:
            return []

        top_indices = [x[0] for x in filtered[:top_k]]
        
        for idx in top_indices:
            answers[idx].similarity_score = float(similarities[idx])
            valid_answers.append(answers[idx])
        
        return valid_answers



    def extract_keywords_from_chunks(self, valid_answers: list[Answer]) -> dict:

        chunk_texts = [ans.chunk.text for ans in valid_answers if ans.chunk]

        logger.info(f"chunk_texts до фильтрации IDF:\n '{chunk_texts}'")

        #chunk_texts = filter_uniform_words(chunk_texts, IDF_THRESHOLD)
        chunk_texts = dymamic_filter_uniform_words(chunk_texts, IDF_THRESHOLD)[0]

        
        logger.info(f"chunk_texts после фильтрации IDF:\n '{chunk_texts}'")
        
        if len(chunk_texts) == 0:
            return {"keywords": [], "chunks": []}
        
        # Выберите нужный метод:
        keywords = find_keywords(self.sentence_tokenizer.tokenizer,
                                chunk_texts, chunk_texts, 
                                1, 20) 
        
        return {
            "keywords": keywords,
            "chunks": [ans.chunk for ans in valid_answers if ans.chunk]
        }



    def process(self, 
               answers: List[Answer], 
               reference_answer: str) -> dict:

        valid_answers = self.validate_answers(answers, reference_answer)
        if len(valid_answers) > 0:
            analysis = self.extract_keywords_from_chunks(valid_answers)
        else:
            analysis = None
        
        return {
            "reference_answer": reference_answer,
            "valid_answers_count": len(valid_answers),
            "valid_answers": valid_answers,
            "analysis": analysis
        }


@dataclass
class FilteringSet:
    text: str = ""
    keywords: List[str] = field(default_factory=list)


    import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class AnswerConsensusFinder:
    """
    Класс для нахождения консенсусного ответа на основе оценок фрагментов и сходства между ответами.
    """
    
    def __init__(self):
        """
        Инициализация модели для вычисления эмбеддингов.
        """
        self.model = SentenceTokenizerSingleton()
    
    def find_consensus_answer(self, 
                            answers: List[Answer],
                            chunk_weight: float = 0.6,
                            similarity_weight: float = 0.4,
                            min_similarity_threshold: float = 0.7,
                            top_k_candidates: int = 5) -> Answer:
        """
        Находит консенсусный ответ на основе комбинации оценки фрагмента и сходства с другими ответами.
        
        Args:
            answers: Список объектов Answer
            chunk_weight: Вес оценки фрагмента в финальной оценке (0-1)
            similarity_weight: Вес сходства с другими ответами (0-1)
            min_similarity_threshold: Минимальный порог сходства для учета
            top_k_candidates: Количество топ-кандидатов для анализа
            
        Returns:
            Answer: Наиболее вероятный правильный ответ
        """
        if not answers:
            return None
        
        if len(answers) == 1:
            return answers[0]
        
        # Нормализуем веса
        total_weight = chunk_weight + similarity_weight
        chunk_weight /= total_weight
        similarity_weight /= total_weight
        
        # Шаг 1: Получаем эмбеддинги для всех ответов
        answer_texts = [answer.text for answer in answers]
        embeddings = self.model.tokenizer.encode(answer_texts)
        
        # Шаг 2: Вычисляем матрицу попарных сходств
        similarity_matrix = cosine_similarity(embeddings)
        
        # Шаг 3: Вычисляем оценку консенсуса для каждого ответа
        consensus_scores = []
        
        for i, answer in enumerate(answers):
            # Базовая оценка из фрагмента (нормализованная)
            chunk_score = getattr(answer, 'chunk_score', 0.0)
            confidence_score = getattr(answer, 'confidence', 0.0)
            
            # Комбинированная оценка фрагмента
            base_score = (chunk_score + confidence_score) / 2
            
            # Оценка сходства с другими ответами
            similarity_scores = []
            for j, other_answer in enumerate(answers):
                if i != j:
                    similarity = similarity_matrix[i][j]
                    if similarity >= min_similarity_threshold:
                        # Взвешиваем сходство с важностью другого ответа
                        other_chunk_score = getattr(other_answer, 'chunk_score', 0.0)
                        other_confidence = getattr(other_answer, 'confidence', 0.0)
                        other_importance = (other_chunk_score + other_confidence) / 2
                        
                        weighted_similarity = similarity * other_importance
                        similarity_scores.append(weighted_similarity)
            
            # Средняя оценка сходства (0 если нет похожих ответов)
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Финальная оценка консенсуса
            consensus_score = (base_score * chunk_weight) + (avg_similarity * similarity_weight)
            
            consensus_scores.append(consensus_score)
        
        # Шаг 4: Выбираем ответ с максимальной оценкой консенсуса
        best_index = np.argmax(consensus_scores)
        best_answer = answers[best_index]
        
        # Добавляем информацию о консенсусе
        best_answer.consensus_score = consensus_scores[best_index]
        best_answer.similar_answers_count = sum(1 for i, score in enumerate(consensus_scores) 
                                              if i != best_index and similarity_matrix[best_index][i] >= min_similarity_threshold)
        
        return best_answer

    def find_consensus_with_clustering(self,
                                     answers: List[Answer],
                                     similarity_threshold: float = 0.75) -> Answer:
        """
        Альтернативная стратегия: кластеризация ответов и выбор из наибольшего кластера.
        """
        print(f"=== НАЧАЛО КЛАСТЕРИЗАЦИИ ОТВЕТОВ ===")
        print(f"Количество ответов для анализа: {len(answers)}")
        print(f"Порог сходства для кластеризации: {similarity_threshold}")
        
        if not answers:
            print("❌ Пустой список ответов")
            return None
            
        if len(answers) == 1:
            print("✅ Только один ответ - возвращаем его")
            return answers[0]
        
        # Получаем эмбеддинги
        answer_texts = [answer.text for answer in answers]
        print(f"\n📊 Тексты ответов для анализа:")
        for i, text in enumerate(answer_texts):
            chunk_score = getattr(answers[i], 'chunk_score', 0.0)
            confidence = getattr(answers[i], 'confidence', 0.0)
            print(f"  {i+1}. '{text[:50]}{'...' if len(text) > 50 else ''}' "
                  f"(chunk_score: {chunk_score:.3f}, confidence: {confidence:.3f})")
        
        embeddings = self.model.tokenizer.encode(answer_texts)
        print(f"✅ Получены эмбеддинги размерности: {embeddings.shape}")
        
        # Простая кластеризация по сходству (возвращает индексы кластеров)
        cluster_indices = self._cluster_answers(embeddings, similarity_threshold)
        
        print(f"\n🔍 Результаты кластеризации:")
        print(f"Образовано кластеров: {len(cluster_indices)}")
        
        # Преобразуем индексы в кластеры объектов Answer
        clusters = []
        for cluster_idx, indices in enumerate(cluster_indices):
            cluster = [answers[i] for i in indices]
            clusters.append(cluster)
            print(f"Кластер {cluster_idx+1}: {len(indices)} ответов (индексы: {indices})")
            
            # Выводим содержимое кластера
            for i, answer_idx in enumerate(indices):
                answer = answers[answer_idx]
                chunk_score = getattr(answer, 'chunk_score', 0.0)
                confidence = getattr(answer, 'confidence', 0.0)
                print(f"    {i+1}. '{answer.text[:60]}{'...' if len(answer.text) > 60 else ''}' "
                      f"(score: {chunk_score:.3f}, conf: {confidence:.3f})")
        
        # Находим самый большой кластер
        if not clusters:
            print("\n⚠️ Кластеры не образовались - возвращаем ответ с наивысшей оценкой")
            best_answer = max(answers, key=lambda x: getattr(x, 'chunk_score', 0.0))
            chunk_score = getattr(best_answer, 'chunk_score', 0.0)
            print(f"✅ Выбран ответ: '{best_answer.text}' (chunk_score: {chunk_score:.3f})")
            return best_answer
        
        largest_cluster = max(clusters, key=len)
        largest_cluster_size = len(largest_cluster)
        
        print(f"\n📈 Самый большой кластер: {largest_cluster_size} ответов")
        
        if not largest_cluster:
            print("⚠️ Самый большой кластер пуст - возвращаем ответ с наивысшей оценкой")
            best_answer = max(answers, key=lambda x: getattr(x, 'chunk_score', 0.0))
            chunk_score = getattr(best_answer, 'chunk_score', 0.0)
            print(f"✅ Выбран ответ: '{best_answer.text}' (chunk_score: {chunk_score:.3f})")
            return best_answer
        
        # В самом большом кластере выбираем ответ с наивысшей оценкой фрагмента
        print(f"\n🎯 Выбор лучшего ответа в самом большом кластере:")
        best_answer_in_cluster = max(largest_cluster, 
                                   key=lambda x: getattr(x, 'chunk_score', 0.0))
        
        chunk_score = getattr(best_answer_in_cluster, 'chunk_score', 0.0)
        confidence = getattr(best_answer_in_cluster, 'confidence', 0.0)
        
        print(f"✅ Лучший ответ в кластере: '{best_answer_in_cluster.text}'")
        print(f"   - Оценка фрагмента: {chunk_score:.3f}")
        print(f"   - Уверенность модели: {confidence:.3f}")
        print(f"   - Размер кластера: {largest_cluster_size}")
        print(f"   - Всего кластеров: {len(clusters)}")
        
        # Добавляем информацию о кластере
        best_answer_in_cluster.cluster_size = len(largest_cluster)
        best_answer_in_cluster.total_clusters = len(clusters)
        
        print(f"\n✅ ФИНАЛЬНЫЙ ВЫБОР: '{best_answer_in_cluster.text[:80]}{'...' if len(best_answer_in_cluster.text) > 80 else ''}'")
        print("=== ЗАВЕРШЕНИЕ КЛАСТЕРИЗАЦИИ ===")
        
        return best_answer_in_cluster
    
    def _cluster_answers(self, embeddings: np.ndarray, threshold: float) -> List[List[int]]:
        """
        Простая кластеризация ответов на основе косинусного сходства.
        Возвращает список кластеров, где каждый кластер - список индексов ответов.
        """
        n = len(embeddings)
        clusters = []
        visited = set()
        
        similarity_matrix = cosine_similarity(embeddings)
        
        for i in range(n):
            if i in visited:
                continue
                
            cluster = [i]
            visited.add(i)
            
            # Находим все похожие ответы
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters

    # Альтернативная упрощенная версия без кластеризации
    def find_consensus_simple(self, answers: List[Answer]) -> Answer:
        """
        Упрощенная версия: выбирает ответ с наибольшей поддержкой от других похожих ответов.
        """
        if not answers:
            return None
            
        if len(answers) == 1:
            return answers[0]
        
        # Получаем эмбеддинги
        answer_texts = [answer.text for answer in answers]
        embeddings = self.model.tokenizer.encode(answer_texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Для каждого ответа вычисляем сумму сходств с другими ответами, взвешенную по их оценкам
        support_scores = []
        
        for i, answer in enumerate(answers):
            chunk_score = getattr(answer, 'chunk_score', 0.0)
            confidence = getattr(answer, 'confidence', 0.0)
            base_score = (chunk_score + confidence) / 2
            
            # Вычисляем поддержку от других ответов
            support = 0.0
            for j, other_answer in enumerate(answers):
                if i != j:
                    other_chunk_score = getattr(other_answer, 'chunk_score', 0.0)
                    other_confidence = getattr(other_answer, 'confidence', 0.0)
                    other_score = (other_chunk_score + other_confidence) / 2
                    
                    support += similarity_matrix[i][j] * other_score
            
            # Комбинированная оценка
            combined_score = 0.7 * base_score + 0.3 * (support / (len(answers) - 1))
            support_scores.append(combined_score)
        
        # Выбираем ответ с максимальной комбинированной оценкой
        best_index = np.argmax(support_scores)
        best_answer = answers[best_index]
        best_answer.consensus_score = support_scores[best_index]
        
        return best_answer

    def extra_find_consensus_with_clustering(self,
                                     answers: List[Answer],
                                     similarity_threshold: float = 0.75,
                                     cluster_selection_strategy: str = "weighted_score",
                                     answer_selection_strategy: str = "highest_chunk_score") -> Answer:
        """
        Альтернативная стратегия: кластеризация ответов и выбор из наилучшего кластера.
        
        Args:
            cluster_selection_strategy: Стратегия выбора кластера:
                - "largest": Самый большой кластер (оригинальная стратегия)
                - "highest_avg_score": Кластер с наивысшим средним chunk_score
                - "weighted_score": Кластер с наивысшим взвешенным score (учитывает размер и качество)
                - "best_single": Кластер, содержащий ответ с наивысшим chunk_score
            answer_selection_strategy: Стратегия выбора ответа внутри кластера:
                - "highest_chunk_score": Ответ с наивысшим chunk_score (по умолчанию)
                - "highest_similarity": Ответ с наивысшим средним сходством с другими ответами в кластере
                - "combined_score": Комбинация chunk_score и сходства
        """
        print(f"=== НАЧАЛО КЛАСТЕРИЗАЦИИ ОТВЕТОВ ===")
        print(f"Количество ответов для анализа: {len(answers)}")
        print(f"Порог сходства для кластеризации: {similarity_threshold}")
        print(f"Стратегия выбора кластера: {cluster_selection_strategy}")
        print(f"Стратегия выбора ответа: {answer_selection_strategy}")
        
        if not answers:
            print("❌ Пустой список ответов")
            return None
            
        if len(answers) == 1:
            print("✅ Только один ответ - возвращаем его")
            return answers[0]
        
        # Получаем эмбеддинги
        answer_texts = [answer.text for answer in answers]
        print(f"\n📊 Тексты ответов для анализа:")
        for i, text in enumerate(answer_texts):
            chunk_score = getattr(answers[i], 'chunk_score', 0.0)
            confidence = getattr(answers[i], 'confidence', 0.0)
            print(f"  {i+1}. '{text[:50]}{'...' if len(text) > 50 else ''}' "
                  f"(chunk_score: {chunk_score:.3f}, confidence: {confidence:.3f})")
        
        embeddings = self.model.tokenizer.encode(answer_texts)
        print(f"✅ Получены эмбеддинги размерности: {embeddings.shape}")
        
        # Вычисляем матрицу сходств для всех ответов
        similarity_matrix = cosine_similarity(embeddings)
        print(f"\n🔬 Матрица сходств (первые 5x5):")
        for i in range(min(5, len(answers))):
            row = [f"{similarity_matrix[i][j]:.2f}" for j in range(min(5, len(answers)))]
            print(f"  {i}: {row}")
        
        # Простая кластеризация по сходству (возвращает индексы кластеров)
        cluster_indices = self._cluster_answers(embeddings, similarity_threshold)
        
        print(f"\n🔍 Результаты кластеризации:")
        print(f"Образовано кластеров: {len(cluster_indices)}")
        
        # Преобразуем индексы в кластеры объектов Answer
        clusters = []
        cluster_metrics = []
        
        for cluster_idx, indices in enumerate(cluster_indices):
            cluster = [answers[i] for i in indices]
            clusters.append(cluster)
            
            # Вычисляем метрики для кластера
            chunk_scores = [getattr(answer, 'chunk_score', 0.0) for answer in cluster]
            confidences = [getattr(answer, 'confidence', 0.0) for answer in cluster]
            
            avg_chunk_score = np.mean(chunk_scores) if chunk_scores else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            max_chunk_score = max(chunk_scores) if chunk_scores else 0.0
            cluster_size = len(cluster)
            
            # Вычисляем среднее сходство внутри кластера
            intra_cluster_similarities = []
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    if i < j:  # Избегаем дублирования и диагонали
                        intra_cluster_similarities.append(similarity_matrix[idx_i][idx_j])
            
            avg_intra_similarity = np.mean(intra_cluster_similarities) if intra_cluster_similarities else 0.0
            
            # Взвешенная оценка кластера (размер * среднее качество)
            weighted_score = cluster_size * avg_chunk_score
            
            cluster_metrics.append({
                'cluster': cluster,
                'size': cluster_size,
                'avg_chunk_score': avg_chunk_score,
                'avg_confidence': avg_confidence,
                'max_chunk_score': max_chunk_score,
                'weighted_score': weighted_score,
                'avg_intra_similarity': avg_intra_similarity,
                'indices': indices,
                'similarity_matrix': similarity_matrix
            })
            
            print(f"\nКластер {cluster_idx+1}:")
            print(f"  Размер: {cluster_size} ответов")
            print(f"  Средний chunk_score: {avg_chunk_score:.3f}")
            print(f"  Средняя уверенность: {avg_confidence:.3f}")
            print(f"  Максимальный chunk_score: {max_chunk_score:.3f}")
            print(f"  Среднее сходство внутри кластера: {avg_intra_similarity:.3f}")
            print(f"  Взвешенная оценка: {weighted_score:.3f}")
            print(f"  Индексы: {indices}")
            
            # Выводим содержимое кластера
            for i, answer in enumerate(cluster):
                chunk_score = getattr(answer, 'chunk_score', 0.0)
                confidence = getattr(answer, 'confidence', 0.0)
                print(f"    {i+1}. '{answer.text[:60]}{'...' if len(answer.text) > 60 else ''}' "
                      f"(score: {chunk_score:.3f}, conf: {confidence:.3f})")
        
        # Находим наилучший кластер по выбранной стратегии
        if not clusters:
            print("\n⚠️ Кластеры не образовались - возвращаем ответ с наивысшей оценкой")
            best_answer = max(answers, key=lambda x: getattr(x, 'chunk_score', 0.0))
            chunk_score = getattr(best_answer, 'chunk_score', 0.0)
            print(f"✅ Выбран ответ: '{best_answer.text}' (chunk_score: {chunk_score:.3f})")
            return best_answer
        
        selected_cluster_metrics = None
        
        if cluster_selection_strategy == "largest":
            selected_cluster_metrics = max(cluster_metrics, key=lambda x: x['size'])
            print(f"\n📈 Стратегия 'largest': выбран самый большой кластер (размер: {selected_cluster_metrics['size']})")
            
        elif cluster_selection_strategy == "highest_avg_score":
            selected_cluster_metrics = max(cluster_metrics, key=lambda x: x['avg_chunk_score'])
            print(f"\n📈 Стратегия 'highest_avg_score': выбран кластер с наивысшим средним score ({selected_cluster_metrics['avg_chunk_score']:.3f})")
            
        elif cluster_selection_strategy == "weighted_score":
            selected_cluster_metrics = max(cluster_metrics, key=lambda x: x['weighted_score'])
            print(f"\n📈 Стратегия 'weighted_score': выбран кластер с наивысшей взвешенной оценкой ({selected_cluster_metrics['weighted_score']:.3f})")
            
        elif cluster_selection_strategy == "best_single":
            global_best_score = max(getattr(answer, 'chunk_score', 0.0) for answer in answers)
            for metrics in cluster_metrics:
                if metrics['max_chunk_score'] == global_best_score:
                    selected_cluster_metrics = metrics
                    break
            if selected_cluster_metrics is None:
                selected_cluster_metrics = max(cluster_metrics, key=lambda x: x['max_chunk_score'])
            print(f"\n📈 Стратегия 'best_single': выбран кластер с наивысшим отдельным ответом (score: {selected_cluster_metrics['max_chunk_score']:.3f})")
        
        elif cluster_selection_strategy == "highest_cohesion":
            # Новая стратегия: кластер с наивысшим внутренним сходством
            selected_cluster_metrics = max(cluster_metrics, key=lambda x: x['avg_intra_similarity'])
            print(f"\n📈 Стратегия 'highest_cohesion': выбран наиболее сплоченный кластер (сходство: {selected_cluster_metrics['avg_intra_similarity']:.3f})")
        
        else:
            selected_cluster_metrics = max(cluster_metrics, key=lambda x: x['weighted_score'])
            print(f"\n📈 Стратегия по умолчанию 'weighted_score': выбран кластер с наивысшей взвешенной оценкой ({selected_cluster_metrics['weighted_score']:.3f})")
        
        selected_cluster = selected_cluster_metrics['cluster']
        cluster_indices_list = selected_cluster_metrics['indices']
        cluster_similarity_matrix = selected_cluster_metrics['similarity_matrix']
        
        # Выбор лучшего ответа в кластере по выбранной стратегии
        print(f"\n🎯 Выбор лучшего ответа в выбранном кластере (стратегия: {answer_selection_strategy}):")
        
        if answer_selection_strategy == "highest_chunk_score":
            # Оригинальная стратегия: наивысший chunk_score
            best_answer_in_cluster = max(selected_cluster, 
                                       key=lambda x: getattr(x, 'chunk_score', 0.0))
            print(f"  Использована стратегия 'highest_chunk_score'")
            
        elif answer_selection_strategy == "highest_similarity":
            # Новая стратегия: ответ с наивысшим средним сходством с другими ответами в кластере
            best_answer_in_cluster = self._select_answer_by_similarity(
                selected_cluster, cluster_indices_list, cluster_similarity_matrix)
            print(f"  Использована стратегия 'highest_similarity'")
            
        elif answer_selection_strategy == "combined_score":
            # Комбинированная стратегия: chunk_score и сходство
            best_answer_in_cluster = self._select_answer_by_combined_score(
                selected_cluster, cluster_indices_list, cluster_similarity_matrix)
            print(f"  Использована стратегия 'combined_score'")
            
        else:
            best_answer_in_cluster = max(selected_cluster, 
                                       key=lambda x: getattr(x, 'chunk_score', 0.0))
            print(f"  Использована стратегия по умолчанию 'highest_chunk_score'")
        
        chunk_score = getattr(best_answer_in_cluster, 'chunk_score', 0.0)
        confidence = getattr(best_answer_in_cluster, 'confidence', 0.0)
        
        print(f"✅ Лучший ответ в кластере: '{best_answer_in_cluster.text}'")
        print(f"   - Оценка фрагмента: {chunk_score:.3f}")
        print(f"   - Уверенность модели: {confidence:.3f}")
        print(f"   - Размер кластера: {selected_cluster_metrics['size']}")
        print(f"   - Средний chunk_score кластера: {selected_cluster_metrics['avg_chunk_score']:.3f}")
        print(f"   - Среднее сходство в кластере: {selected_cluster_metrics['avg_intra_similarity']:.3f}")
        print(f"   - Всего кластеров: {len(clusters)}")
        print(f"   - Стратегия выбора кластера: {cluster_selection_strategy}")
        print(f"   - Стратегия выбора ответа: {answer_selection_strategy}")
        
        # Добавляем информацию о кластере
        best_answer_in_cluster.cluster_size = selected_cluster_metrics['size']
        best_answer_in_cluster.total_clusters = len(clusters)
        best_answer_in_cluster.cluster_avg_score = selected_cluster_metrics['avg_chunk_score']
        best_answer_in_cluster.cluster_avg_similarity = selected_cluster_metrics['avg_intra_similarity']
        best_answer_in_cluster.selection_strategy = f"{cluster_selection_strategy}+{answer_selection_strategy}"
        
        print(f"\n✅ ФИНАЛЬНЫЙ ВЫБОР: '{best_answer_in_cluster.text[:80]}{'...' if len(best_answer_in_cluster.text) > 80 else ''}'")
        print("=== ЗАВЕРШЕНИЕ КЛАСТЕРИЗАЦИИ ===")
        
        return best_answer_in_cluster

    def _select_answer_by_similarity(self, cluster: List[Answer], indices: List[int], 
                                   similarity_matrix: np.ndarray) -> Answer:
        """
        Выбирает ответ с наивысшим средним сходством с другими ответами в кластере.
        """
        print(f"  Поиск ответа с максимальным средним сходством в кластере:")
        
        best_answer = None
        best_avg_similarity = -1
        
        for i, answer in enumerate(cluster):
            original_index = indices[i]
            similarities = []
            
            for j, other_answer in enumerate(cluster):
                if i != j:
                    other_original_index = indices[j]
                    similarity = similarity_matrix[original_index][other_original_index]
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            chunk_score = getattr(answer, 'chunk_score', 0.0)
            
            print(f"    Ответ {i+1}: '{answer.text[:40]}...' - среднее сходство: {avg_similarity:.3f}, chunk_score: {chunk_score:.3f}")
            
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_answer = answer
        
        print(f"  🎯 Выбран ответ со средним сходством: {best_avg_similarity:.3f}")
        return best_answer

    def _select_answer_by_combined_score(self, cluster: List[Answer], indices: List[int],
                                       similarity_matrix: np.ndarray) -> Answer:
        """
        Выбирает ответ на основе комбинации chunk_score и сходства.
        """
        print(f"  Поиск ответа по комбинированной оценке:")
        
        best_answer = None
        best_combined_score = -1
        
        for i, answer in enumerate(cluster):
            original_index = indices[i]
            
            # Вычисляем среднее сходство
            similarities = []
            for j, other_answer in enumerate(cluster):
                if i != j:
                    other_original_index = indices[j]
                    similarity = similarity_matrix[original_index][other_original_index]
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            chunk_score = getattr(answer, 'chunk_score', 0.0)
            
            # Комбинированная оценка (50% chunk_score + 50% сходство)
            combined_score = 0.5 * chunk_score + 0.5 * avg_similarity
            
            print(f"    Ответ {i+1}: '{answer.text[:40]}...' - combined: {combined_score:.3f} "
                  f"(chunk: {chunk_score:.3f}, similarity: {avg_similarity:.3f})")
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_answer = answer
        
        print(f"  🎯 Выбран ответ с комбинированной оценкой: {best_combined_score:.3f}")
        return best_answer
