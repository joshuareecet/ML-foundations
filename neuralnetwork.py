import torch
from torch import nn
from utils.setup import device

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.ReLU(),
			nn.Linear(512,10)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

if __name__ == "__main__":
	model = NeuralNetwork().to(device)
	print(model)

	X = torch.rand(1,28,28, device = device)
	print(X)
	logits = model(X)
	pred_prob = nn.Softmax(1)(logits)
	y_pred = pred_prob.argmax(1)
	print(f"Predicted class {y_pred}, full prob {pred_prob}")