from models_processing.system_data_setting import get_CUDA_num, get_device
import sys
import os
from pathlib import Path
from transformers import *
from sentence_transformers import SentenceTransformer
import torch
from enum import Enum, auto
#--------------------------------------------------

#---------------------------------------------------- 
#       Загрузка путей


current_path = Path(__file__).absolute()

bert_qa_dir = current_path.parent / "models" / "bert_eng_qa_model"
bert_qa_dir = str(bert_qa_dir)

rubert_qa_dir = current_path.parent / "models" / "rubert_ru_qa_model"
rubert_qa_dir = str(rubert_qa_dir)

#tinyroberta_qa_dir = current_path.parent / "models" / "bert_eng_qa_tinyroberta_model"
#tinyroberta_qa_dir = current_path.parent / "models" / "bert_eng_qa_distilledroberta_model"
tinyroberta_qa_dir = current_path.parent / "models" / "bert_eng_qa_baseroberta_model"
tinyroberta_qa_dir = str(tinyroberta_qa_dir)

bert_senttok_dir = current_path.parent / "models" / "eng_sentence_transformer_model"
bert_senttok_dir = str(bert_senttok_dir)

distbert_senttok_dir = current_path.parent / "models" / "eng_sentence_transformer_distilbert_model"
distbert_senttok_dir = str(distbert_senttok_dir)

#---------------------------------------------------- 

class SupportedLanguage(Enum):
    RUS = auto()
    ENG = auto()

class SupportedModels(Enum):
    BERT = auto()
    DistilBERT = auto()
    RoBERTa = auto()

#--------------------------------------------------


try:

    cuda_num = get_CUDA_num()
    device =  get_device()
    # lng_code = SupportedLanguage.ENG
    lng_code = SupportedLanguage.RUS
    # qa_model_type = SupportedModels.RoBERTa
    qa_model_type = SupportedModels.BERT
    sent_model_type = SupportedModels.BERT

except Exception as e:

    raise e

os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
#----------------------------------------------------

#---------------------------------------------------
concrete_tokenizer = None
concrete_sentence_tokenizer = None
concrete_sentence_model = None
concrete_qa_model = None
concrete_ner_model = None
#---------------------------------------------------

def get_lang_code():
    return lng_code

def set_concrete_tokenizer(lang_code=lng_code, qa_model_type=qa_model_type):
    global concrete_tokenizer
    if lang_code==SupportedLanguage.RUS:
        if qa_model_type==SupportedModels.BERT:
            concrete_tokenizer = AutoTokenizer.from_pretrained(rubert_qa_dir)
    elif lang_code==SupportedLanguage.ENG:
        if qa_model_type==SupportedModels.BERT:
            concrete_tokenizer = BertTokenizer.from_pretrained(bert_qa_dir)
        elif qa_model_type==SupportedModels.RoBERTa:
            concrete_tokenizer = AutoTokenizer.from_pretrained(tinyroberta_qa_dir)
    else:
        concrete_tokenizer = None

def set_concrete_sentence_tokenizer(lang_code=lng_code):
    global concrete_sentence_tokenizer
    global concrete_sentence_model
    if lang_code==SupportedLanguage.RUS:
        concrete_sentence_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        concrete_sentence_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
    else:
        concrete_sentence_tokenizer = None
        concrete_sentence_model = None

def set_concrete_QA_model(lang_code=lng_code, qa_model_type=qa_model_type):
    global concrete_qa_model
    if lang_code==SupportedLanguage.RUS:
        concrete_qa_model = None
        #concrete_qa_model = build_model('squad_ru_bert', download=True)
        concrete_qa_model = AutoModelForQuestionAnswering.from_pretrained(rubert_qa_dir).to(device)
    # elif lang_code==SupportedLanguage.ENG:
    #     if qa_model_type==SupportedModels.BERT:
    #         concrete_qa_model = None
    #         concrete_qa_model = BertForQuestionAnswering.from_pretrained(bert_qa_dir).to(device)
    #     elif qa_model_type==SupportedModels.RoBERTa:
    #         concrete_qa_model = None
    #         concrete_qa_model = AutoModelForQuestionAnswering.from_pretrained(tinyroberta_qa_dir, output_attentions=True).to(device)
    # else:
    #     concrete_qa_model = None

def set_concrete_NER_model(lang_code=lng_code):
    global concrete_ner_model
    if lang_code==SupportedLanguage.RUS:
        concrete_ner_model = None
        #concrete_ner_model = build_model('ner_ontonotes_bert_mult', download=True)
    elif lang_code==SupportedLanguage.ENG:
        concrete_ner_model = None
        #concrete_ner_model = build_model('ner_ontonotes_bert_mult', download=True)    
    else:
        concrete_ner_model = None

#---------------------------------------------------

def get_concrete_tokenizer():

    if concrete_tokenizer is None:
        set_concrete_tokenizer()

    return concrete_tokenizer
    

def get_concrete_sentence_tokenizer():

    if lng_code==SupportedLanguage.ENG:
        if sent_model_type==SupportedModels.BERT:
            model = SentenceTransformer(bert_senttok_dir, device = device)
            return model, None
        elif sent_model_type==SupportedModels.DistilBERT:
            model = SentenceTransformer(distbert_senttok_dir, device = device)
            return model, None


    if (concrete_sentence_tokenizer is None) and \
       (concrete_sentence_model is None):
       set_concrete_sentence_tokenizer()

    return concrete_sentence_tokenizer, concrete_sentence_model


def get_concrete_QA_model():

    if concrete_qa_model is None:
        set_concrete_QA_model()

    return concrete_qa_model


def get_concrete_NER_model():

    if concrete_ner_model is None:
        set_concrete_NER_model()

    return concrete_ner_model

#---------------------------------------------------

class SentenceTokenizer():

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_sentence_embeddings(self, sentences_array):
        encoded_input = self.tokenizer(sentences_array, 
                                        return_tensors="pt", 
                                        truncation=True, padding=True, max_length = 512 )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']).numpy()
        return sentence_embeddings

#---------------------------------------------------

def convert_ner_category_to_tag(ner_category):

    if ner_category == "PERSON":
        return "PERSON"
    if ner_category == "NORP":
        return "NORP"
    elif ner_category == "GPE":
        return "GPE"
    elif ner_category == "LOCATION":
        return "LOC"
    elif ner_category == "FACILITY":
        return "FAC"
    elif ner_category == "ORGANIZATION":
        return "ORG"
    elif ner_category == "PRODUCT":
        return "PRODUCT"
    elif ner_category == "EVENT":
        return "EVENT"
    elif ner_category == "WORK OF ART":
        return "WORK_OF_ART"
    elif ner_category == "LAW":
        return "LAW"
    elif ner_category == "LANGUAGE":
        return "LANGUAGE"
    elif ner_category == "DATE":
        return "DATE"
    elif ner_category == "TIME":
        return "TIME"
    elif ner_category == "PERCENT":
        return "PERCENT"
    elif ner_category == "MONEY":
        return "MONEY"
    elif ner_category == "QUANTITY":
        return "QUANTITY"
    if ner_category == "ORDINAL":
        return "ORDINAL"
    if ner_category == "CARDINAL":
        return "CARDINAL"
    else:
         return "O"

# ---------------------------------------------------------