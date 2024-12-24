
import torch
from torch.optim import SGD

torch.manual_seed(42)

dataset = [
    (torch.tensor([1, 0, 0.7, 0.8, 0.6, 0.7, 0.9, 0.5, 0.8]), torch.tensor([1.0])),
    (torch.tensor([0, 1, 0.4, 0.7, 0.9, 0.8, 0.7, 0.4, 0.6]), torch.tensor([1.0])),
    (torch.tensor([1, 0, 0.5, 0.6, 0.5, 0.6, 0.8, 0.6, 0.7]), torch.tensor([0.0])),
    (torch.tensor([0, 1, 0.3, 0.4, 0.8, 0.7, 0.6, 0.3, 0.5]), torch.tensor([0.0])),
    (torch.tensor([1, 1, 0.9, 0.9, 0.7, 0.9, 0.9, 0.8, 0.9]), torch.tensor([1.0])),
    (torch.tensor([0, 0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.2, 0.3]), torch.tensor([0.0])),
    (torch.tensor([1, 0, 0.6, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6]), torch.tensor([1.0])),
    (torch.tensor([0, 1, 0.3, 0.5, 0.6, 0.5, 0.6, 0.3, 0.4]), torch.tensor([0.0])),
    (torch.tensor([1, 1, 0.8, 0.9, 0.9, 0.8, 0.9, 0.7, 0.8]), torch.tensor([1.0]))
]

w_technical = torch.rand((1, 3), requires_grad=True)
w_emotional = torch.rand((1, 3), requires_grad=True)
w_metadata = torch.rand((1, 3), requires_grad=True)
# Смещение
bias = torch.tensor([0.5], requires_grad=True)

# Оптимизатор
optimizer = SGD([w_technical, w_emotional, w_metadata, bias], lr=0.01)

# Функция предсказания
def predict_likelihood(track_features: torch.Tensor) -> torch.Tensor:
    technical = track_features[2:5] @ w_technical.T
    emotional = track_features[5:8] @ w_emotional.T
    metadata = track_features[:2].unsqueeze(0) @ w_metadata[:, :2].T + track_features[8:] @ w_metadata[:, 2:].T
    return technical + emotional + metadata + bias

# Функция потерь
def loss_function(y_pred, y_actual):
    return torch.nn.functional.mse_loss(y_pred, y_actual)

# Обучение модели
for epoch in range(100):  # 100 эпох обучения
    total_loss = 0
    for track, label in dataset:
        optimizer.zero_grad()
        predicted = predict_likelihood(track) # Предсказание
        loss = loss_function(predicted, label) # Вычисление ошибки
        total_loss += loss.item()
        loss.backward() # Обратное распространение ошибки
        optimizer.step() # Шаг оптимизации

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Тестирование модели
print("\nFinal Weights:")
print("Technical Weights:", w_technical)
print("Emotional Weights:", w_emotional)
print("Metadata Weights:", w_metadata)
print("Bias:", bias)