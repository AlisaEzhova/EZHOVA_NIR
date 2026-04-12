print("Начинаем обучение...")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Добавляем путь к проекту
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.recommender import MatrixFactorization

print("Все библиотеки импортированы")

# Проверка данных
if not os.path.exists('data/ratings.csv'):
    print("Файл data/ratings.csv не найден!")
    exit(1)

print("Данные найдены")

# Загрузка
df = pd.read_csv('data/ratings.csv')
print(f"Загружено {len(df)} оценок")

# Маппинги
user_ids = df['userId'].unique()
item_ids = df['movieId'].unique()
user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

df['user_id'] = df['userId'].map(user_to_idx)
df['item_id'] = df['movieId'].map(item_to_idx)

print(f"Пользователей: {len(user_ids)}, товаров: {len(item_ids)}")

# Разделение
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

num_users = len(user_ids)
num_items = len(item_ids)

# Модель
model = MatrixFactorization(num_users, num_items, embedding_dim=50)
print(f"Модель создана: {num_users} пользователей, {num_items} товаров")

# Обучение
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = nn.MSELoss()

train_users = torch.tensor(train_df['user_id'].values)
train_items = torch.tensor(train_df['item_id'].values)
train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)

val_users = torch.tensor(val_df['user_id'].values)
val_items = torch.tensor(val_df['item_id'].values)
val_ratings = torch.tensor(val_df['rating'].values, dtype=torch.float32)

print("Запускаем обучение...")

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    preds = model(train_users, train_items)
    loss = criterion(preds, train_ratings)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = model(val_users, val_items)
        rmse = torch.sqrt(criterion(val_preds, val_ratings))
    
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}, val_rmse={rmse.item():.4f}")

# Сохранение
torch.save({
    'model_state_dict': model.state_dict(),
    'user_to_idx': user_to_idx,
    'item_to_idx': item_to_idx,
    'num_users': num_users,
    'num_items': num_items
}, 'recommender_model.pt')

print("Модель сохранена в recommender_model.pt")
print(f"Итоговый RMSE: {rmse.item():.4f}")