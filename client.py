import flwr as fl
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import sys
from model import Net
from torchvision import transforms
from PIL import Image

# ----------------- تنظیمات اولیه -----------------
client_id = int(sys.argv[1])                                                                   #شماره کلاینت را در ورودی ترمینال از کاربر می گیرد
num_clients = 3                                                                                #تعداد کلاینت ها
alpha = 0.3                                                                                    # پارامتر Dirichlet برای حالت نا همگونی داده ها،هر چه این پارامتر بزرگتر ، داده ها همگن تر و هر چه کوچتر، داده ها ها همگن تر
batch_size = 32                                

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()                                                              #تابع خطا
optimizer = optim.SGD(model.parameters(), lr=0.01)                                             #تابع بهینه ساز ، Learning rate = 0.01

# ----------------- دیتاست -----------------
class Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = data.iloc[:, 1:].values.astype("float32") / 255.0
        self.y = data.iloc[:, 0].values.astype("int64")
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(28,28)
        y = self.y[idx]
        x = Image.fromarray((x*255).astype(np.uint8))                                         
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ----------------- تابع تقسیم‌بندی Dirichlet -----------------یعنی هر کلاینت ترکیبی از کلاس ها را دارد اما با نسبت های متفاوت
def dirichlet_partition(dataset, num_clients, alpha=0.3):
    labels = np.array(dataset.y)
    num_classes = len(np.unique(labels))
    idx_per_class = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        np.random.shuffle(idx_per_class[c])
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_per_class[c])).astype(int)[:-1]
        splits = np.split(idx_per_class[c], proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])
    return [np.array(indices) for indices in client_indices]

# ----------------- Feature Skew ----------------- یعنی مثلا یک کلاینت در داده های تصویری، داده هاش تار هستند، یک کلاینت داده هاش زیاد روشن هستند
def get_base_transform(client_id):
    if client_id == 0:
        return transforms.Compose([transforms.RandomRotation(20), transforms.ToTensor()])           
    elif client_id == 1:
        return transforms.Compose([transforms.GaussianBlur(kernel_size=(3,3)), transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.ColorJitter(brightness=0.5), transforms.ToTensor()])

# ----------------- Concept Drift ----------------- یعنی در هر راند داده های آموزشی کلاینت ها متفاوت خواهد شد و تغییر می کند
def get_drift_transform(round_num, base_transform):
    if round_num % 3 == 0:
        extra = transforms.RandomRotation(10)
    elif round_num % 3 == 1:
        extra = transforms.ColorJitter(contrast=0.3)
    else:
        extra = transforms.GaussianBlur(kernel_size=(5,5))
    return transforms.Compose([extra] + base_transform.transforms)

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
base_transform = get_base_transform(client_id)
train_dataset = Dataset("fashion-mnist/fashion-mnist_train.csv", transform=base_transform)
test_dataset = Dataset("fashion-mnist/fashion-mnist_test.csv", transform=transforms.ToTensor())

client_partitions = dirichlet_partition(train_dataset, num_clients, alpha)                    #توزیع Dirichlet 
quantity_ratios = [1.0, 0.5, 0.1]                                                             # client0: 100%, client1: 50%, client2: 10%
client_indices = client_partitions[client_id]                                                 ###
num_samples = int(len(client_indices) * quantity_ratios[client_id])                           ###---------Quantity skew----------
client_indices = np.random.choice(client_indices, num_samples, replace=False)                 ###یعنی هر کلاینت تعداد متفاوتی از داده ها دارند، مثلا یکی 2000 و یکی هم 20000

train_subset = Subset(train_dataset, client_indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------- کلاس کلاینت -----------------
class HybridClient(fl.client.NumPyClient):
    def get_parameters(self, config):                                                         #ارسال پارامتر های اموزش دیده شده به سرور
        return [val.detach().cpu().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):                                                     #دریافت پارمتر های مدل اصلی از سرور
        for p, new_p in zip(model.parameters(), parameters):
            p.data = torch.tensor(new_p, device=device)

    def fit(self, parameters, config):                                                        #شروع فرایند آموزش
        self.set_parameters(parameters)                                                      
        model.train()
        round_num = int(config.get("server_round", config.get("round", 0)))

        drifted_transform = get_drift_transform(round_num, base_transform)
        train_dataset_drift = Dataset("fashion-mnist/fashion-mnist_train.csv", transform=drifted_transform)
        train_subset_drift = Subset(train_dataset_drift, client_indices)
        train_loader_drift = DataLoader(train_subset_drift, batch_size=batch_size, shuffle=True)

        acc_before = evaluate(model, test_loader)                                              #ارزیابی قبل مدل
        
        num_epochs = 1                                                                         #تعداد Epoch
        for epoch in range(num_epochs):
            for data, target in train_loader_drift:                                  
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        acc_after = evaluate(model, test_loader)                                               #اریابی بعد مدل

        print(f"Client {client_id} | Round {round_num} | acc before={acc_before:.3f}, after={acc_after:.3f}")
        return self.get_parameters(None), len(train_subset_drift), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = evaluate(model, test_loader)
        return float(1 - acc), len(test_loader), {"accuracy": acc}

# ----------------- اجرای کلاینت -----------------
if __name__ == "__main__":
    print(f"Client {client_id} initialized with {len(train_subset)} samples ({quantity_ratios[client_id]*100:.0f}% of base).")
    #برای مقیاس لوکال از localhost استفاده می کنیم
    #برای مقیاس واقعی باید از IP واقعی سرور استفاده کنیم
    fl.client.start_numpy_client(server_address="localhost:8080", client=HybridClient())
