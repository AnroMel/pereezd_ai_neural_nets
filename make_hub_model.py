# make_hub_model.py
import torch
import sys
import pathlib

FIXED = "best_barrier_rail_v1_fixed.pt"   # твой текущий файл
HUB   = "best_barrier_rail_v1_hub.pt"     # новый файл для torch.hub.load

# Добавляем локальный yolov5 в sys.path, чтобы нашёлся модуль "models"
y5_root = pathlib.Path(__file__).parent / "yolov5"
if str(y5_root.resolve()) not in sys.path:
    sys.path.append(str(y5_root.resolve()))
    print("Добавил в sys.path:", y5_root)

print("Загружаю фиксированную модель:", FIXED)
model = torch.load(FIXED, map_location="cpu")
print("Тип модели:", type(model))

ckpt = {"model": model}  # YOLOv5 ждёт словарь с ключом 'model'

print("Сохраняю hub-совместимый чекпоинт:", HUB)
torch.save(ckpt, HUB)

print("✅ Готово! Используй", HUB, "в качестве MODEL_PATH")
