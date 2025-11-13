import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
import random

# SMILES 标记器
class SmilesTokenizer:
    def __init__(self):
        self.chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefghiklmnoprstuy$"
        self.vocab = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.vocab)

    def encode(self, smiles, max_length=64):
        encoded = [self.vocab.get(c, 0) for c in smiles]
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded = encoded + [0] * (max_length - len(encoded))
        return encoded

def randomize_smiles(smiles, random_type='restricted'):
    """返回SMILES的随机化表达（增强数据）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, doRandom=True, canonical=False)

# 数据集定义
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, results, tokenizer, max_length=64, augment_times=0):
        self.smiles_list = []
        self.results = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for smiles, label in zip(smiles_list, results):
            self.smiles_list.append(smiles)
            self.results.append(label)
            # 数据增强：为每个SMILES生成augment_times个随机SMILES
            for _ in range(augment_times):
                rand_smiles = randomize_smiles(smiles)
                self.smiles_list.append(rand_smiles)
                self.results.append(label)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.results[idx]
        input_ids = torch.tensor(self.tokenizer.encode(smiles, self.max_length), dtype=torch.long)
        return {'input_ids': input_ids, 'label': torch.tensor(label, dtype=torch.float)}

# Transformer模型定义
class SimpleTransformerRegressor(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_length=64, dropout=0.1, act=nn.ReLU):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, activation=act.__name__.lower()
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.max_length = max_length

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x).squeeze(-1)
        return out

# 训练函数
def train_model(model, train_loader, val_loader, epochs=400, device='cuda'):  # 训练轮数增至300
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 学习率减小为1e-4
    loss_fn = nn.SmoothL1Loss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    df = pd.read_csv('data1.csv')  # 替换为你的数据文件路径
    smiles_list = df['SMILES'].tolist()
    results = df['Td/K'].tolist()
    
    # 数据标准化
    mean_result = np.mean(results)
    std_result = np.std(results)
    results_norm = [(x - mean_result) / std_result for x in results]
    
    # 初始化分词器
    tokenizer = SmilesTokenizer()
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    all_val_labels = []
    all_val_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(smiles_list)):
        print(f"\nFold {fold+1}/5")
        
        # 准备数据
        train_smiles = [smiles_list[i] for i in train_idx]
        val_smiles = [smiles_list[i] for i in val_idx]
        train_results = [results_norm[i] for i in train_idx]
        val_results = [results_norm[i] for i in val_idx]
        
        # 创建数据加载器
        train_dataset = SmilesDataset(train_smiles, train_results, tokenizer)
        val_dataset = SmilesDataset(val_smiles, val_results, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # 初始化模型
        model = SimpleTransformerRegressor(
            vocab_size=tokenizer.vocab_size,
            d_model=256,
            nhead=8,
            num_layers=3  # 更浅的模型
        ).to(device)
        
        # 训练模型
        train_losses, val_losses = train_model(
            model, 
            train_loader, 
            val_loader,
            epochs=400,
            device=device
        )

       
        # 评估模型
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].cpu().numpy()
                outputs = model(input_ids).cpu().numpy()
                val_preds.extend(outputs)
                val_labels.extend(labels)

        # 反标准化预测结果
        val_preds = np.array(val_preds) * std_result + mean_result
        val_labels = np.array(val_labels) * std_result + mean_result

        # 保存每一折的验证集结果
        all_val_labels.append(val_labels)
        all_val_preds.append(val_preds)

        # 计算metrics
        mse = mean_squared_error(val_labels, val_preds)
        r2 = r2_score(val_labels, val_preds)
        mae = mean_absolute_error(val_labels, val_preds)
        fold_results.append({'MSE': mse, 'R2': r2, 'MAE': mae})
        
        print(f"Fold {fold+1} Results:")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
    
    # 打印平均结果
    avg_mse = np.mean([x['MSE'] for x in fold_results])
    avg_r2 = np.mean([x['R2'] for x in fold_results])
    avg_mae = np.mean([x['MAE'] for x in fold_results])
    print("\nAverage Results:")
    print(f"MSE: {avg_mse:.4f}")
    print(f"R2: {avg_r2:.4f}")
    print(f"MAE: {avg_mae:.4f}")

    # 随机选取一折，画出真实值 vs 预测值散点图
    random_fold = random.randint(0, 4)
    print(f'Randomly selected fold: {random_fold+1}')
    y_true = all_val_labels[random_fold]
    y_pred = all_val_preds[random_fold]
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'True vs Predicted')
    plt.tight_layout()
    plt.savefig(f'dantonhdao_true_vs_pred.png')
    plt.close()

def plot_residuals(y_true, y_pred, name):
    residuals = y_true - y_pred

    # 设置图形大小
    plt.figure(figsize=(8, 6))

    # 创建直方图
    n, bins, patches = plt.hist(residuals, bins=50, density=True)

    # 根据柱体高度调整颜色深浅（越高越深）
    norm = plt.Normalize(vmin=min(n), vmax=max(n))  # 归一化高度
    cmap = plt.cm.Blues  # 选择颜色映射

    for count, patch in zip(n, patches):
        color = cmap(norm(count))  # 根据高度设置颜色
        plt.setp(patch, 'facecolor', color)

    # 添加标签和标题
    plt.xlabel('Residual (True - Predicted)', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.title(f'{name} Residual Distribution', fontsize=16)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'{name}_residuals.png')
    plt.close()


if __name__ == "__main__":
    main()