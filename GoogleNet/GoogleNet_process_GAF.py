import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.core.audio import samplerate
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pywt
import os
import shutil
import random
from pyts.image import GramianAngularField
from scipy.signal import resample


# ========== STEP 1: 心音预处理 ==========

def read_m4a_and_clip(file_path, start_time=0, end_time=5):
    data, sample_rate = librosa.load(file_path, sr=None)
    if data.ndim == 2:
        data = data[:, 0]
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    return sample_rate, data[start_sample:end_sample]


def generate_reference_signal(sample_rate, frequency, duration):
    t = np.arange(0, duration, 1 / sample_rate)
    return np.cos(2 * np.pi * frequency * t)


def mix_signal(received_signal, reference_signal):
    return received_signal * reference_signal


def low_pass_filter(signal, cutoff_frequency, sample_rate, order=4):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def extract_heart_signal(received_signal, reference_frequency, sample_rate, cutoff_frequency, duration):
    reference_signal = generate_reference_signal(sample_rate, reference_frequency, duration)
    mixed = mix_signal(received_signal, reference_signal)
    filtered = low_pass_filter(mixed, cutoff_frequency, sample_rate)
    return filtered


def ica_extract(signal, n_components=1):
    ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
    reshaped = signal.reshape(-1, 1)
    return ica.fit_transform(reshaped).squeeze()


# ========== STEP 2: GAF图像生成 ==========


def save_gaf_image(signal, save_path, image_size=224, method='summation'):
    # 归一化信号到 [-1, 1]
    signal = signal - np.mean(signal)
    signal = signal / (np.max(np.abs(signal)) + 1e-8)

    # 确保是二维输入
    signal_2d = signal.reshape(1, -1)

    # 使用 GASF 或 GADF 方法
    gaf = GramianAngularField(method=method, image_size=image_size)
    gaf_image = gaf.fit_transform(signal_2d)[0]

    # 保存图像
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.imshow(gaf_image, cmap='rainbow', origin='upper')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# ========== STEP 3: 图像数据集定义 ==========

class CWTImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.class_map = {}
        self.transform = transform

        for idx, cls in enumerate(sorted(os.listdir(root_dir))):
            self.class_map[cls] = idx
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith(".png"):
                    self.samples.append(os.path.join(cls_path, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ========== STEP 4: 预处理+生成图像 ==========

def generate_gaf_images_from_audio(input_dir, output_dir, ref_freq=21000):
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(input_dir):  # 分类目录，例如 Normal / Abnormal
        label_path = os.path.join(input_dir, label)
        save_label_dir = os.path.join(output_dir, label)
        os.makedirs(save_label_dir, exist_ok=True)

        count = 0
        for fname in os.listdir(label_path):
            if not fname.endswith(".wav"):
                continue
            file_path = os.path.join(label_path, fname)
            sr, raw = read_m4a_and_clip(file_path)
            #filtered = extract_heart_signal(raw, ref_freq, sr, 25, duration=5)
            #heart = ica_extract(filtered)
            heart = raw

            #降采样到 250Hz
            #resample_rate = 1000
            #resampled = samplerate.resample(heart, resample_rate / sr, converter_type='sinc_best')
            #heart_downsampled = resampled / np.max(np.abs(resampled))
            #segment = heart_downsampled[1024:2048]

            for i in range(0, 5):
                segment = heart[i*1000: (i+1)*1000]
                save_name = os.path.join(save_label_dir, f"{fname[:-4]}_{count}.png")
                save_gaf_image(segment, save_name)
                count += 1


# ========== STEP 5: 划分数据 ==========
def data_split(source_root, target_root):
    # 比例
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 初始化目标文件夹
    splits = ['train', 'val', 'test']
    class_names = os.listdir(source_root)
    for split in splits:
        for cls in class_names:
            os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

    # 对每个类别进行划分
    for cls in class_names:
        class_path = os.path.join(source_root, cls)
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits_indices = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split in splits:
            for img in splits_indices[split]:
                src = os.path.join(class_path, img)
                dst = os.path.join(target_root, split, cls, img)
                shutil.copyfile(src, dst)

    print("✅ 数据集划分完成，已保存在：", target_root)


# ========== STEP 6: 模型训练(已经转移至脚本) ==========

def train_model(data_dir, num_epochs=10, batch_size=16, lr=1e-4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CWTImageDataset(data_dir, transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = models.googlenet(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(set(dataset.labels)))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.cuda(), torch.tensor(labels).cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch + 1}: Loss={total_loss:.4f}, Acc={acc:.4f}")


# ========== MAIN CALL ==========

if __name__ == "__main__":
    # 1. 从音频生成 GAF 图像
    generate_gaf_images_from_audio("heart_data", "Vit_data_gaf")

    # 2. 划分测试集验证集与训练集
    data_split("Vit_data_gaf", "dataset_split")
