import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNet_Weights


def train():
    # 设置参数
    data_dir = "dataset_split"
    num_classes = 4
    batch_size = 128
    num_epochs = 30
    learning_rate = 0.001
    model_save_path = "best_GoogleNet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 构建模型
    model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器（每10轮将LR减半）
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels)

        train_acc = train_correct.double() / len(train_dataset)

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        val_acc = val_correct.double() / len(val_dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print("✅ Saved best model.")

        scheduler.step()

    print("🎯 Training finished.")


if __name__ == "__main__":
    train()
