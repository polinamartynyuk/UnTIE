from transformers import AutoTokenizer, AutoModelForQuestionAnswering

save_dir = "../models/bert_eng_qa_distilledroberta_model/"

# Загрузка модели и токенизатора
model_name = "deepset/roberta-base-squad2-distilled"

# Загрузка модели и токенизатора
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Сохранение в указанную папку
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
