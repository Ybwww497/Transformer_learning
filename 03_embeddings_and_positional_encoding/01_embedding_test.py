import torch
import torch.nn as nn

embedding = nn.Embedding(10, 4)
token_ids = torch.tensor([[1, 3, 5],[2, 4, 6]])
x = embedding(token_ids)

print(token_ids.shape)
print(x.shape)



import torch 
import torch.nn as nn

embedding = nn.Embedding(10, 4)
token_ids = torch.tensor([[1, 3, 5],[2, 4, 6]])
x = embedding(token_ids)

print("\ntoken_ids.shape =",token_ids.shape)
print("x.shape =",x.shape)
print("embedding.weight.shape =",embedding.weight.shape)

print("\nembedding(token_ids)[0, 0] =")
print(x[0, 0])

print("\nembedding.weight[1] =")
print(embedding.weight[1])
