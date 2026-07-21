from sentence_transformers import SentenceTransformer

save_dir = "../models/eng_sentence_transformer_distilbert_model/"

# Загрузка модели
model_name = "sentence-transformers/msmarco-distilbert-cos-v5"
sentence_model = SentenceTransformer(model_name)

# Сохранение в указанную папку
sentence_model.save(save_dir)