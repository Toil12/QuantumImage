import torch
from torchvision import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')