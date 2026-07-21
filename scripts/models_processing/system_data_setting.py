import torch
from typing import List
import numpy as np
#------------------------------

CUDA_num = 1


#------------------------------

print("------------------------------------")
print("---> Choosing resource")
print("---> Result of torch.cuda.is_available():")
print(torch.cuda.is_available())
print("---> Final resource")
device = ("cuda:" + str(CUDA_num)) if torch.cuda.is_available() else "cpu"
print((f"-----> CUDA: {str(CUDA_num)}")  if torch.cuda.is_available() else "-----> CPU")
print(f"-----> device: {device}")
print("------------------------------------")


#------------------------------  

def get_CUDA_num() -> int:

    return CUDA_num


def get_device() -> str:

    return device
