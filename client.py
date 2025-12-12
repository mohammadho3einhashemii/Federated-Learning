import flwr as fl
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import sys
from model import Net

# ----------------- تنظیمات اولیه -----------------
client_id = int(sys.argv[1])   # شماره کلاینت از آرگومان خط فرمان
num_clients = 3                
batch_size = 32

# ----------------- تنظیمات دستگاه -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ----------------- تابع ارزیابی -----------------
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

# ----------------- آماده‌سازی داده -----------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

# تقسیم داده‌ها به طور مساوی بین کلاینت‌ها
num_samples_per_client = len(train_dataset) // num_clients
start_idx = client_id * num_samples_per_client
end_idx = (client_id + 1) * num_samples_per_client
train_indices = list(range(start_idx, end_idx))

train_subset = Subset(train_dataset, train_indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------- کلاس کلاینت -----------------
class SimpleClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(model.parameters(), parameters):
            p.data = torch.tensor(new_p, device=device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        return self.get_parameters(None), len(train_subset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = evaluate(model, test_loader)
        return float(1 - acc), len(test_loader.dataset), {"accuracy": acc}

# ----------------- اجرای کلاینت -----------------
if __name__ == "__main__":
    print(f"Client {client_id} initialized with {len(train_subset)} samples.")
    fl.client.start_numpy_client(server_address="localhost:8080", client=SimpleClient())
