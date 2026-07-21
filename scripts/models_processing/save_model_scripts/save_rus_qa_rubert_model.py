from transformers import AutoTokenizer, AutoModelForQuestionAnswering

save_dir = "../models/rubert_ru_qa_model/"

# Загрузка модели и токенизатора
model_name = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"

# Загрузка модели и токенизатора
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Сохранение в указанную папку
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
