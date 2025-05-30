import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import GoogLeNet_Weights
import numpy as np


def test():
    # 参数
    data_dir = "dataset_split/test"
    model_path = "best_GoogleNet.pth"
    num_classes = 4
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像预处理（要与训练一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 加载测试集
    test_dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 推理与统计
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 打印分类报告和混淆矩阵
    print("📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    test()
