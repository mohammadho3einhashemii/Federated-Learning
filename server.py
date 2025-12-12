import flwr as fl
from model import Net                                                                                   #از فایل مدل، کلاس نت را برای استفاده از مدل ایمپروت کردم
import torch
from torchvision import datasets, transforms                                                             #این کتابخانه مخصوص کار با داده های تصویری
from typing import Dict, Tuple, List

# دیتاست تست برای ارزیابی مدل نهایی
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = Net().to(device)

# تابع ارزیابی
def evaluate_model(model: torch.nn.Module) -> float:
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

# ذخیره تاریخچه دقت
history: Dict[str, List[Tuple[int, float]]] = {"server_accuracy": []}

# تابع evaluate_fn برای چاپ دقت سرور
def evaluate_fn(server_round: int, parameters, config) -> Tuple[float, Dict[str, float]]:
    # بروز رسانی مدل سرور با پارامترهای دریافت شده
    for p, new_p in zip(global_model.parameters(), parameters):
        p.data = torch.tensor(new_p, device=device)
    acc = evaluate_model(global_model)
    print(f"\n=== Round {server_round} Server Evaluation ===")
    print(f"Server accuracy on test set: {acc:.4f}\n")
    history["server_accuracy"].append((server_round, acc))
    return 1-acc, {"accuracy": acc}

# استراتژی FedAvg
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,                                                                                         #نسبت کلاینت هایی که در هر دور آموزش شرکت می کنند(مثلا 1 یعنی همه کلاینت ها شرکت کنند)
    fraction_evaluate=1.0,                   
    min_fit_clients=3,                                                                                        # حداقل تعداد کلاینت هایی که بایددر یک دور آموزش شرکت کند
    min_evaluate_clients=3,      
    min_available_clients=3,                                                                                  #حدقال تعداد کلاینت های متصل به سرور برای شروع فرایند آموزش
    evaluate_fn=evaluate_fn
)

# کانفیگ سرور
server_config = fl.server.ServerConfig(num_rounds=3)                                                          # مشخص کردن تعداد راند های آموزش

# اجرای سرور
fl.server.start_server(
    server_address="localhost:8080",
    strategy=strategy,
    config=server_config
)

# چاپ تاریخچه نهایی
print("\n=== Summary of Server Accuracy ===")
for rnd, acc in history["server_accuracy"]:
    print(f"Round {rnd}: {acc:.4f}")
