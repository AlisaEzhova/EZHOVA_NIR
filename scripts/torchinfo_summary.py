"""
Скрипт для вывода архитектуры Neural BPR через torchinfo.
Запуск: python scripts/torchinfo_summary.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class FastBPR(nn.Module):
    """Нейронная сеть с BPR-функцией потерь для рекомендаций."""

    def __init__(self, num_users: int, num_items: int, dim: int = 16):
        """
        Args:
            num_users: количество пользователей
            num_items: количество товаров
            dim: размерность эмбеддингов
        """
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
        """
        Прямой проход для BPR.
        
        Args:
            u: ID пользователей [batch_size]
            i: ID позитивных товаров [batch_size]
            j: ID негативных товаров [batch_size]
            
        Returns:
            Кортеж из скоров для позитивных и негативных товаров
        """
        u_emb = self.user_emb(u)
        i_emb = self.item_emb(i)
        j_emb = self.item_emb(j)
        return (u_emb * i_emb).sum(dim=1), (u_emb * j_emb).sum(dim=1)


if __name__ == "__main__":
    # Параметры из реального эксперимента
    NUM_USERS = 1_382_085
    NUM_ITEMS = 760_794
    EMBEDDING_DIM = 16
    BATCH_SIZE = 2048

    print("=" * 70)
    print("Архитектура Neural BPR модели (torchinfo)")
    print("=" * 70)
    print(f"Параметры: users={NUM_USERS:,}, items={NUM_ITEMS:,}, dim={EMBEDDING_DIM}")
    print()

    model = FastBPR(NUM_USERS, NUM_ITEMS, EMBEDDING_DIM)

    # Вывод torchinfo
    summary(
        model,
        input_data=[
            torch.randint(0, NUM_USERS, (BATCH_SIZE,)),   # u
            torch.randint(0, NUM_ITEMS, (BATCH_SIZE,)),   # i
            torch.randint(0, NUM_ITEMS, (BATCH_SIZE,))    # j
        ],
        col_names=["input_size", "output_size", "num_params"],
        depth=2,
        verbose=1
    )

    print("\n" + "=" * 70)
    print("Экспорт модели в ONNX для Netron")
    print("=" * 70)

    # Экспорт в ONNX
    dummy_u = torch.randint(0, NUM_USERS, (1,))
    dummy_i = torch.randint(0, NUM_ITEMS, (1,))
    dummy_j = torch.randint(0, NUM_ITEMS, (1,))

    torch.onnx.export(
        model,
        (dummy_u, dummy_i, dummy_j),
        "neural_bpr_model.onnx",
        input_names=["user_ids", "pos_item_ids", "neg_item_ids"],
        output_names=["pos_scores", "neg_scores"],
        dynamic_axes={
            "user_ids": {0: "batch_size"},
            "pos_item_ids": {0: "batch_size"},
            "neg_item_ids": {0: "batch_size"}
        },
        opset_version=11
    )
    print("✅ Модель экспортирована в neural_bpr_model.onnx")
    print("   Загрузите этот файл на https://netron.app для визуализации")
