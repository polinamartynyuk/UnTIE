import os
import logging
from pathlib import Path
#--------------------------------------------------
#       Настройка путей

current_path = Path(__file__).absolute()
logs_path = current_path.parent / "logs"
os.makedirs(logs_path, exist_ok=True)

#--------------------------------------------------

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Настройка логгера с защитой от дублирования обработчиков."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Проверяем, есть ли уже нужный обработчик
    handler_exists = any(
        isinstance(h, logging.FileHandler) and h.baseFilename.endswith(log_file)
        for h in logger.handlers
    )

    if not handler_exists:
        # Создаём обработчик с перезаписью (mode='w')
        file_handler = logging.FileHandler(
            os.path.join(logs_path, log_file),
            mode='w'
        )
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger