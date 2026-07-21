import pandas as pd
import json
from typing import List, Dict, Union, Optional

def save_dataframe_to_json(df: pd.DataFrame, file_path: str, orient: str = 'records') -> None:
    """
    Сохраняет DataFrame в JSON файл
    
    Параметры:
        df: DataFrame для сохранения
        file_path: путь к файлу для сохранения
        orient: формат сохранения ('records', 'split', 'index', 'columns', 'values')
                по умолчанию 'records' - список словарей
    """
    df.to_json(file_path, orient=orient, force_ascii=False, indent=4)
    parts = file_path.split('/')
    short_path = f"{parts[-2]}/{parts[-1]}"
    print(f"The dataset was successfully saved to {short_path}")

def load_dataframe_from_json(file_path: str, 
                   orient: str = 'records') -> pd.DataFrame:
    """
    Простая загрузка JSON в DataFrame без дополнительных проверок
    
    Параметры:
        file_path: путь к JSON файлу
        orient: формат JSON ('records', 'split', 'index', 'columns', 'values')
    
    Возвращает:
        Загруженный DataFrame
    
    Исключения:
        FileNotFoundError: если файл не существует
        ValueError: если JSON невалидный
    """
    try:
        return pd.read_json(file_path, orient=orient, lines=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except ValueError as e:
        try: 
            return pd.read_json(file_path)
        except Exception as e:
            raise ValueError(f"Error while loading JSON: {str(e)}")


def get_empty_answer_records(df: pd.DataFrame, 
                           answer_columns: List[str] = ['first_answer', 'corrected_answer']) -> pd.DataFrame:
    """
    Возвращает записи, где не заполнены указанные колонки с ответами
    
    Параметры:
        df: исходный DataFrame
        answer_columns: список колонок для проверки
        
    Возвращает:
        DataFrame с записями, где хотя бы одна из указанных колонок пуста
    """
    # Проверяем, что указанные колонки существуют в DataFrame
    missing_cols = [col for col in answer_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"В DataFrame отсутствуют колонки: {missing_cols}")
    
    # Создаем маску для поиска пустых значений
    mask = df[answer_columns].isna().any(axis=1) | (df[answer_columns].eq('')).any(axis=1)
    
    return df[mask].copy()

def replace_underscores_in_array_column(df: pd.DataFrame,
                                        column_name: str,
                                        new_column_name: Optional[str] = None,
                                        inplace: bool = False
                                    ) -> pd.DataFrame:
    """
    Заменяет нижние подчёркивания на пробелы в массивах строк указанной колонки.
    
    Параметры:
        df: Исходный DataFrame
        column_name: Название колонки с массивами строк
        new_column_name: Название новой колонки (если None, будет использовано исходное имя + '_cleaned')
        inplace: Если True, модифицирует исходный DataFrame
    
    Возвращает:
        DataFrame с новой колонкой (если inplace=False)
    """
    if not inplace:
        df = df.copy()
    
    if new_column_name is None:
        new_column_name = f"{column_name}_cleaned"
    
    def process_array(arr: List[str]) -> List[str]:
        """Обрабатывает массив строк, заменяя _ на пробелы"""
        if not isinstance(arr, list):
            return arr
        return [s.replace('_', ' ') for s in arr]
    
    # Применяем функцию к каждой строке колонки
    df[new_column_name] = df[column_name].apply(process_array)
    
    return df