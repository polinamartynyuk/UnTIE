from transformers import BertForQuestionAnswering, BertTokenizer

save_dir = "../models/bert_eng_qa_model/"

# Загрузка модели и токенизатора
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Сохранение в указанную папку
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)