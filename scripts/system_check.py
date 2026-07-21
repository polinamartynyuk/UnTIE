import torch
from io import StringIO
import sys


def CUDA_state_print():

    # Проверяем, доступен ли CUDA
    print("CUDA доступна:", torch.cuda.is_available())

    # Количество доступных GPU
    gpu_count = torch.cuda.device_count()
    print("Количество GPU:", gpu_count)

    # Информация о каждой карте
    for i in range(gpu_count):
        print(f"\nGPU #{i}:")
        print("Название:", torch.cuda.get_device_name(i))
        print("Вычислительная способность (CUDA Capability):", torch.cuda.get_device_capability(i))
        print("Объем памяти:", round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2), "GB")


def CUDA_state_print_limited():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    CUDA_state_print()
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    print(output[:37])