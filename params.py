'''Storing parameters'''
import torch 
SEQ_LENGTH = 7
device = torch.device("cpu")

# device = ''
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
# else:
#     device = torch.device("cpu")
#     print("No GPU available, using CPU.")
