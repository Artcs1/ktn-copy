import torch

weights = torch.load("/home/artcs/Desktop/KernelTransformerNetwork-master1/models/pascal2_2.transform.pt")
    
for i in weights:
    print(i)