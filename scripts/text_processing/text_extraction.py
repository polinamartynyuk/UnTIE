import re
from typing import List

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])" 

class TextProcesser():

    def split_into_sentences(self, text: str) -> List[str]:
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        text = re.sub (digits + "[.]" + digits, "\\1<prd> \\2 ", text)
        if '"' in text: text = text.replace('."','".')
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]

        corrected_sentences = []
        for sentence in sentences:
            if sentence != ".":
                corrected_sentences.append(sentence)

        return corrected_sentences

# Алфавит (кириллица + латиница для смешанных текстов)
ru_alphabets = "([А-Яа-яA-Za-z])"
# Русские префиксы (титулы, обращения)
ru_prefixes = "(г-жа|г-н|т|д|проф|акад|докт|инж|ген|полк|кап|лейт|серж|мл|ст|старш)[.]"
# Русские суффиксы (организации, сокращения)
ru_suffixes = "(ООО|ЗАО|ОАО|ИП|РФ|США|ЕС|мл|млрд|тыс|ст|мл|млад|старш)"
# Русские стартеры (начало предложения)
ru_starters = "(Он\s|Она\s|Оно\s|Они\s|Мы\s|Вы\s|Я\s|Ты\s|Но\s|Однако\s|Что\s|Это\s|Здесь\s|Там\s|Где\s|Который\s|Какой\s|Так\s|И\s|А\s|Но\s|Или)"
# Аббревиатуры (русские и латинские)
ru_acronyms = "([А-Я][.][А-Я][.](?:[А-Я][.])?)|([A-Z][.][A-Z][.](?:[A-Z][.])?)"
# Сайты (те же)
ru_websites = "[.](com|net|org|io|gov|me|edu|ru|рф|su)"
# Цифры
digits = "([0-9])"
# Русские сокращения (специфичные)
rus_abbreviations = "(и[.]т[.]д|и[.]т[.]п|т[.]е|т[.]к|см|ул|д|кв|корп|стр|рис|табл|гл|п|пп|ст|статьи|№)"


# class RuTextProcessor():
    
#     def split_into_sentences(self, text: str) -> List[str]:
#         """
#         Разделение текста на предложения с учётом русского языка
#         """
#         text = " " + text + "  "
#         text = text.replace("\n", " ")
        
#         # Защита русских префиксов от ложного разбиения
#         text = re.sub(ru_prefixes, "\\1<prd>", text)
        
#         # Защита сайтов
#         text = re.sub(ru_websites, "<prd>\\1", text)
        
#         # Защита русских аббревиатур (и т.д., т.е., и т.п.)
#         text = re.sub(rus_abbreviations, lambda m: m.group().replace(".", "<prd>"), text)
        
#         # Защита Ph.D. и подобных
#         if "Ph.D" in text: 
#             text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        
#         # Защита инициалов (А.С. Пушкин)
#         text = re.sub("\s" + ru_alphabets + "[.] ", " \\1<prd> ", text)
        
#         # Защита аббревиатур перед стартерами
#         text = re.sub(ru_acronyms + " " + ru_starters, "\\1<stop> \\2", text)
        
#         # Защита инициалов (А.С.П.)
#         text = re.sub(ru_alphabets + "[.]" + ru_alphabets + "[.]" + ru_alphabets + "[.]", 
#                      "\\1<prd>\\2<prd>\\3<prd>", text)
#         text = re.sub(ru_alphabets + "[.]" + ru_alphabets + "[.]", 
#                      "\\1<prd>\\2<prd>", text)
        
#         # Защита суффиксов
#         text = re.sub(" " + ru_suffixes + "[.] " + ru_starters, " \\1<stop> \\2", text)
#         text = re.sub(" " + ru_suffixes + "[.]", " \\1<prd>", text)
        
#         # Защита одиночных букв с точкой
#         text = re.sub(" " + ru_alphabets + "[.]", " \\1<prd>", text)
        
#         # Защита чисел с точкой (3.14, 1.5)
#         text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
        
#         # Защита кавычек
#         if '"' in text: 
#             text = text.replace('."', '".')
#         if "\"" in text: 
#             text = text.replace(".\"", "\".")
#         if "!" in text: 
#             text = text.replace('!"', '"!')
#         if "?" in text: 
#             text = text.replace('?"', '"?')
        
#         # Разметка концов предложений
#         text = text.replace(".", ".<stop>")
#         text = text.replace("?", "?<stop>")
#         text = text.replace("!", "!<stop>")
#         text = text.replace("…", "…<stop>")  # Многоточие
        
#         # Восстановление защищённых точек
#         text = text.replace("<prd>", ".")
        
#         # Разделение по маркерам
#         sentences = text.split("<stop>")
#         sentences = sentences[:-1]  # Убрать последний пустой
#         sentences = [s.strip() for s in sentences]
        
#         # Фильтрация пустых и невалидных
#         corrected_sentences = []
#         for sentence in sentences:
#             if sentence and sentence not in [".", "?", "!", "…"]:
#                 corrected_sentences.append(sentence)
        
#         return corrected_sentences
