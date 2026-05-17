import torch
from torch import nn
from torch.utils.data import DataLoader

from data import (FMNIST_training_data, FMNIST_test_data)
from neuralnetwork import NeuralNetwork
from utils.setup import device, models_dir

# Training hyperparameters -----------------------------------------------------------------------------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 5
MODEL_NAME = "FMNIST_MODEL_V1"
model_name = models_dir / MODEL_NAME

# Loading Data -----------------------------------------------------------------------------------------------------------------
train_dataloader = DataLoader(FMNIST_training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(FMNIST_test_data, batch_size=BATCH_SIZE)

# Train and test loops ---------------------------------------------------------------------------------------------------------
def train_loop(dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X,Y) in enumerate(dataloader):
		X, Y = X.to(device), Y.to(device)
		pred = model(X)
		loss = loss_fn(pred, Y)
		
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if batch % 100 == 0:
			curr_loss, cur_progress = loss.item(), batch * BATCH_SIZE + len(X)
			print(f"Current loss: {curr_loss:>7f}	|	[{cur_progress:>5d} / {size:>5d}]")

def test_loop(dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module):
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			Y = model(X)
			test_loss += loss_fn(Y, y).item()
			correct += (Y.argmax(1) == y).type(torch.float).sum().item()
	
	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
	model = NeuralNetwork().to(device)
	if model_name.exists():
		model.load_state_dict(torch.load(model_name,weights_only=True))
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)

	for t in range(EPOCHS):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loop(train_dataloader, model, loss_fn, optimizer)
		test_loop(test_dataloader, model, loss_fn)
	print("Done!")
	torch.save(model.state_dict(), model_name)