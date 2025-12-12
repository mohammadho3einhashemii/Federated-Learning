import flwr as fl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from model import Net
from typing import Dict, List, Tuple

# ----------------- دیتاست تست -----------------
class Dataset_test(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.X = data.iloc[:, 1:].values.astype("float32") / 255.0
        self.y = data.iloc[:, 0].values.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

test_dataset = Dataset_test("fashion-mnist/fashion-mnist_test.csv")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = Net().to(device)

# ----------------- تابع ارزیابی -----------------
#محسابه دقت مدل روی دیتاست تست( می تواند توسط سرور یا کلاینت اجرا شود ) 
def evaluate_model(model):                                                        
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

# ----------------- ذخیره تاریخچه دقت -----------------
history: Dict[str, List[Tuple[int, float]]] = {"server_accuracy": []}

# ----------------- تابع evaluate_fn -----------------
 #ارزیابی مدل کلی سرور بعد از هر راند(فقط در سرور اجرا می شود ) 
def evaluate_fn(server_round: int, parameters, config) -> Tuple[float, Dict[str, float]]:
    for p, new_p in zip(global_model.parameters(), parameters):
        p.data = torch.tensor(new_p, device=device)
    acc = evaluate_model(global_model)
    print(f"\n=== Round {server_round} Server Evaluation ===")
    print(f"Server accuracy on test set: {acc:.4f}\n")
    history["server_accuracy"].append((server_round, acc))
    return 1-acc, {"accuracy": acc}

# ----------------- استراتژی FedAvg -----------------
#از هر نوع استراتژی دیگر هم بسته به شرایط، نوع داده، سخت افزار و ... می توان استفاده کرد
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,                                                   #چه درصدی از کلاینت ها به صورت تصادفی ، برای آموزش انتخاب شوند(100% کلاینت ها در اینجا)
    fraction_evaluate=1.0,                                              
    min_fit_clients=3,                                                   #حداقل تعداد کلاینت ها برای آموزش در هر راند
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_fn=evaluate_fn
)

server_config = fl.server.ServerConfig(num_rounds=5)                  #تعداد راند ها 

# ----------------- اجرای سرور -----------------
fl.server.start_server(
    #برای اجرای لوکال از localhost استفاده می کنیم
    #برای مقیاس واقعی کلاینت ها ، باید از 0.0.0.0 یا IP خود سرور استفاده کنیم
    server_address="localhost:8080",                                     
    strategy=strategy,
    config=server_config
)

print("\n=== Summary of Server Accuracy ===")
for rnd, acc in history["server_accuracy"]:
    print(f"Round {rnd}: {acc:.4f}")
