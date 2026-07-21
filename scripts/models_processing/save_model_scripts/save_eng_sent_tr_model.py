from sentence_transformers import SentenceTransformer

save_dir = "../models/eng_sentence_transformer_model/"

# Загрузка модели и токенизатора
model_name = "bert-large-nli-mean-tokens"
sentence_model = SentenceTransformer(model_name)

# Сохранение в указанную папку
sentence_model.save(save_dir)