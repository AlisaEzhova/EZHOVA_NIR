import torch
import sys
sys.path.append('.')

from app.recommender import MatrixFactorization

# Создаём модель
model = MatrixFactorization(num_users=610, num_items=9724, embedding_dim=50)
model.eval()

# Создаём фиктивные входные данные
dummy_users = torch.randint(0, 610, (1,))
dummy_items = torch.randint(0, 9724, (1,))

# Экспорт в ONNX с параметром dynamo=False (используем старый экспортёр)
torch.onnx.export(
    model, 
    (dummy_users, dummy_items), 
    "model.onnx", 
    verbose=True,
    input_names=['users', 'items'],
    output_names=['predictions'],
    dynamo=False  # <-- ЭТО КЛЮЧЕВОЙ ПАРАМЕТР!
)

print("Файл model.onnx создан!")