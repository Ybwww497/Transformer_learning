import torch

X = torch.tensor([[1., 0., 1., 0.],[1., 1., 0., 0.],[0., 0., 1., 1.]])
scores = X @ X.transpose(0, 1)
weights = torch.softmax(scores, dim = -1)
out = weights @ X

print("X.shape =", X.shape)
print("scores.shape =", scores.shape)
print("scores =", scores)
print("weights =", weights)
print("out =", out)
print("out.shape =", out.shape)