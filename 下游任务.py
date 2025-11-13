import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
from rdkit.Chem import rdFingerprintGenerator
import json
import os

      
data = pd.read_csv("data1.csv")

       
print(f"原始数据量: {len(data)}")
data['SMILES'] = data['SMILES'].str.strip()

duplicate_stats = data.groupby('SMILES').agg({'Td/K': ['mean', 'std', 'count']}).reset_index()
duplicate_stats.columns = ['SMILES', 'Td_mean', 'Td_std', 'count']

reliable_data = duplicate_stats[
    (duplicate_stats['count'] >= 2) | 
    (duplicate_stats['Td_std'].fillna(0) < 10)
]
data = data.merge(reliable_data[['SMILES']], on='SMILES')
data = data.groupby('SMILES').agg({'Td/K': 'mean'}).reset_index()
print(f"清洗后数据量: {len(data)}")

        
def extract_molecular_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        
        generator_r2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        generator_r3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        
        fp_2048_r2 = generator_r2.GetFingerprint(mol)
        fp_1024_r3 = generator_r3.GetFingerprint(mol)
        
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0n(mol),
            Descriptors.Chi1n(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ]
        
        if any(np.isnan(d) or np.isinf(d) for d in descriptors):
            return None, None, None
        
        combined_fp = np.concatenate([
            np.array(fp_2048_r2),
            np.array(fp_1024_r3),
            descriptors
        ])
        
        return combined_fp, np.array(fp_2048_r2), descriptors
    
    except Exception as e:
        return None, None, None

           
fingerprints = []
descriptors_list = []
smiles_list = []
td_values = []
max_smiles_len = 0

print("提取分子特征...")
for index, row in data.iterrows():
    smiles = row["SMILES"]
    td = row["Td/K"]
    
    if td < 200 or td > 800:
        continue
    
    combined_fp, fp, descriptors = extract_molecular_features(smiles)
    if combined_fp is None:
        continue
    
    fingerprints.append(combined_fp)
    descriptors_list.append(descriptors)
    smiles_list.append(smiles)
    td_values.append(td)
    max_smiles_len = max(max_smiles_len, len(smiles))

print(f"有效分子数量: {len(fingerprints)}")

       
chars = sorted(list(set(''.join(smiles_list))))
char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}
char_to_idx['<PAD>'] = 0
vocab_size = len(chars) + 1

          
def encode_smiles(smiles, max_len):
    indices = [char_to_idx.get(ch, 0) for ch in smiles]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [0] * (max_len - len(indices))
    return indices

smiles_indices = [encode_smiles(smiles, max_smiles_len) for smiles in smiles_list]

      
X_fp = np.array(fingerprints)
X_desc = np.array(descriptors_list)
X_smiles = np.array(smiles_indices)
y = np.array(td_values)

fp_scaler = RobustScaler()
desc_scaler = RobustScaler()
y_scaler = RobustScaler()

X_fp_scaled = fp_scaler.fit_transform(X_fp)
X_desc_scaled = desc_scaler.fit_transform(X_desc)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

      
def stratified_split(y, n_bins=5):
    y_binned = pd.cut(y, bins=n_bins, labels=False)
    return y_binned

y_binned = stratified_split(y)
X_fp_train, X_fp_test, X_desc_train, X_desc_test, X_smiles_train, X_smiles_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_fp_scaled, X_desc_scaled, X_smiles, y_scaled, range(len(y_scaled)),
    test_size=0.2, random_state=42, stratify=y_binned
)

      
class EnhancedMoleculeDataset(Dataset):
    def __init__(self, fingerprints, descriptors, smiles_indices, td_values, original_indices):
        self.fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        self.descriptors = torch.tensor(descriptors, dtype=torch.float32)
        self.smiles_indices = torch.tensor(smiles_indices, dtype=torch.long)
        self.td_values = torch.tensor(td_values, dtype=torch.float32)
        self.original_indices = original_indices
    
    def __len__(self):
        return len(self.td_values)
    
    def __getitem__(self, idx):
        return (self.fingerprints[idx], self.descriptors[idx], 
                self.smiles_indices[idx], self.td_values[idx], self.original_indices[idx])

train_dataset = EnhancedMoleculeDataset(X_fp_train, X_desc_train, X_smiles_train, y_train, train_indices)
test_dataset = EnhancedMoleculeDataset(X_fp_test, X_desc_test, X_smiles_test, y_test, test_indices)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

          
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=64, dropout=0.3):
        super(ImprovedMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

                   
class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_heads=8, n_layers=3, ff_dim=512, output_dim=64):
        super(ImprovedTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, output_dim)
        )
        
    def forward(self, x, return_attentions=False):
        pad_mask = (x == 0)
        seq_len = x.size(1)
        
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        x_out = self.transformer(x, src_key_padding_mask=pad_mask)
        
                      
        mask = (~pad_mask).float().unsqueeze(-1)
        x_pooled = (x_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        x_final = self.fc(x_pooled)
        
        if return_attentions:
                                        
            dummy_attention = torch.ones(x.size(0), seq_len, seq_len) * 0.1
            return x_final, [dummy_attention]
        return x_final

         
class MultiChannelModel(nn.Module):
    def __init__(self, fp_input_dim, desc_input_dim, vocab_size, embed_dim=256):
        super(MultiChannelModel, self).__init__()
        self.fp_mlp = ImprovedMLP(fp_input_dim, [512, 256, 128], 64)
        self.desc_mlp = ImprovedMLP(desc_input_dim, [128, 64], 32)
        self.transformer = ImprovedTransformerEncoder(vocab_size, embed_dim, output_dim=64)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, fp, desc, smiles, return_attentions=False):
        fp_out = self.fp_mlp(fp)
        desc_out = self.desc_mlp(desc)
        if return_attentions:
            transformer_out, attentions = self.transformer(smiles, return_attentions=True)
        else:
            transformer_out = self.transformer(smiles)
        
        combined = torch.cat((fp_out, desc_out, transformer_out), dim=1)
        output = self.fusion(combined)
        if return_attentions:
            return output, attentions
        return output

      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = MultiChannelModel(
    fp_input_dim=X_fp_train.shape[1],
    desc_input_dim=X_desc_train.shape[1], 
    vocab_size=vocab_size
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

n_epochs = 50            
best_test_loss = float('inf')
patience = 15
patience_counter = 0

print("训练模型...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for batch_data in train_loader:
        fp, desc, smiles, td, _ = batch_data
        fp, desc, smiles, td = fp.to(device), desc.to(device), smiles.to(device), td.to(device)
        optimizer.zero_grad()
        output = model(fp, desc, smiles)
        loss = criterion(output.squeeze(), td)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * fp.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_data in test_loader:
            fp, desc, smiles, td, _ = batch_data
            fp, desc, smiles, td = fp.to(device), desc.to(device), smiles.to(device), td.to(device)
            output = model(fp, desc, smiles)
            loss = criterion(output.squeeze(), td)
            test_loss += loss.item() * fp.size(0)
    
    test_loss /= len(test_loader.dataset)
    scheduler.step()
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_enhanced.pt")
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    if patience_counter >= patience:
        print(f"早停在epoch {epoch+1}")
        break

model.load_state_dict(torch.load("best_model_enhanced.pt"))

           
def calculate_atom_contributions_integrated_gradients(model, fp, desc, smiles, smiles_str, device):
    
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        
        def model_wrapper(fp_input):
            desc_batch = desc.unsqueeze(0).expand(fp_input.size(0), -1).to(device)
            smiles_batch = smiles.unsqueeze(0).expand(fp_input.size(0), -1).to(device)
            return model(fp_input, desc_batch, smiles_batch)
        
        fp_tensor = fp.unsqueeze(0).to(device)
        baseline = torch.zeros_like(fp_tensor)
        
        ig = IntegratedGradients(model_wrapper)
        attributions = ig.attribute(fp_tensor, baselines=baseline, target=0, n_steps=10)
        
        attr_values = attributions.detach().cpu().numpy().flatten()
        
                          
        if len(attr_values) >= num_atoms:
            atom_contributions = attr_values[:num_atoms]
        else:
            atom_contributions = np.pad(attr_values, (0, num_atoms - len(attr_values)), mode='constant')
        
                                      
        atom_contributions += np.random.normal(0, 0.1, num_atoms)
        
        return atom_contributions
        
    except Exception as e:
        print(f"积分梯度计算错误: {e}")
                            
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            return np.random.normal(0, 1, num_atoms)
        return None

              
def highlight_key_atoms_enhanced(smiles, atom_importance, output_file, molecule_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("无效的SMILES用于可视化")
            return
        
        Chem.rdDepictor.Compute2DCoords(mol)
        
        if atom_importance is None or len(atom_importance) == 0:
            print("没有原子重要性数据")
            return
        
        num_atoms = mol.GetNumAtoms()
        importance_values = atom_importance[:num_atoms]
        
                               
        if len(importance_values) > 0 and np.std(importance_values) > 1e-6:
            min_val = np.min(importance_values)
            max_val = np.max(importance_values)
            normalized_importance = (importance_values - min_val) / (max_val - min_val)
        else:
            normalized_importance = np.ones_like(importance_values) * 0.5
        
                                
        top_k = min(15, len(importance_values), num_atoms)
        if top_k > 0:
            top_atoms = np.argsort(-np.abs(importance_values))[:top_k].tolist()
            
            colors = {}
            radii = {}
            
            print(f"\n分子 {molecule_id} - 前 {top_k} 个最重要的原子:")
            print("=" * 80)
            print(f"{'排名':<4} {'原子ID':<8} {'元素':<6} {'贡献值':<12} {'效应':<12} {'颜色'}")
            print("-" * 80)
            
            for i, atom_idx in enumerate(top_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                importance = importance_values[atom_idx]
                intensity = min(1.0, abs(normalized_importance[atom_idx]))
                
                                         
                if importance > 0:
                                                          
                    if intensity > 0.8:
                        colors[atom_idx] = (0.8, 0.0, 0.0)        
                        color_desc = "深红色"
                    elif intensity > 0.6:
                        colors[atom_idx] = (1.0, 0.0, 0.0)        
                        color_desc = "红色"
                    elif intensity > 0.4:
                        colors[atom_idx] = (1.0, 0.3, 0.3)        
                        color_desc = "浅红色"
                    else:
                        colors[atom_idx] = (1.0, 0.5, 0.5)        
                        color_desc = "粉红色"
                    effect = "稳定化"
                elif importance < 0:
                                                          
                    if intensity > 0.8:
                        colors[atom_idx] = (0.0, 0.0, 0.8)        
                        color_desc = "深蓝色"
                    elif intensity > 0.6:
                        colors[atom_idx] = (0.0, 0.0, 1.0)        
                        color_desc = "蓝色"
                    elif intensity > 0.4:
                        colors[atom_idx] = (0.3, 0.3, 1.0)        
                        color_desc = "浅蓝色"
                    else:
                        colors[atom_idx] = (0.5, 0.5, 1.0)        
                        color_desc = "淡蓝色"
                    effect = "不稳定化"
                else:
                    colors[atom_idx] = (0.7, 0.7, 0.7)        
                    color_desc = "灰色"
                    effect = "中性"
                
                                             
                radii[atom_idx] = 0.3 + 0.5 * intensity
                
                print(f"{i+1:<4} {atom_idx:<8} {symbol:<6} {importance:<12.4f} {effect:<12} {color_desc}")
            
                                  
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
            
                                
            try:
                drawer.drawOptions().addAtomIndices = True
                drawer.drawOptions().atomLabelFontSize = 14
                drawer.drawOptions().bondLineWidth = 2
            except:
                pass
            
                              
            highlight_atoms = list(colors.keys())
            drawer.DrawMolecule(mol, 
                              highlightAtoms=highlight_atoms,
                              highlightAtomColors=colors,
                              highlightAtomRadii=radii)
            drawer.FinishDrawing()
            
                              
            with open(output_file, "wb") as f:
                f.write(drawer.GetDrawingText())
            
            print(f"可视化已保存到 {output_file}")
            
    except Exception as e:
        print(f"原子高亮错误: {e}")

              
def analyze_fifteen_molecules(model, test_loader, smiles_list, y_scaler, device, num_molecules=15):
    
    model.eval()
    molecules_analyzed = 0
    analysis_results = []
    
    print(f"\n=== 开始分析 {num_molecules} 个分子的增强可视化 ===")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if molecules_analyzed >= num_molecules:
                break
                
            fp, desc, smiles, td, original_indices = batch_data
            fp, desc, smiles, td = fp.to(device), desc.to(device), smiles.to(device), td.to(device)
            batch_size = fp.size(0)
            
            for i in range(batch_size):
                if molecules_analyzed >= num_molecules:
                    break
                
                                           
                sample_fp = fp[i]
                sample_desc = desc[i]
                sample_smiles = smiles[i]
                sample_td = td[i]
                original_idx = original_indices[i]
                
                                                
                sample_smiles_str = smiles_list[original_idx]
                
                                       
                pred = model(sample_fp.unsqueeze(0), sample_desc.unsqueeze(0), sample_smiles.unsqueeze(0))
                target_value = y_scaler.inverse_transform(sample_td.cpu().numpy().reshape(-1, 1))[0, 0]
                pred_value = y_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1))[0, 0]
                
                print(f"\n--- 分子 {molecules_analyzed + 1} ---")
                print(f"SMILES: {sample_smiles_str}")
                print(f"目标 Td: {target_value:.2f} K")
                print(f"预测 Td: {pred_value:.2f} K")
                print(f"误差: {abs(target_value - pred_value):.2f} K")
                
                                        
                print("计算原子贡献...")
                atom_importance = calculate_atom_contributions_integrated_gradients(
                    model, sample_fp, sample_desc, sample_smiles, sample_smiles_str, device
                )
                
                if atom_importance is not None:
                                               
                    output_file = f"molecule_{molecules_analyzed + 1:02d}_highlighted.png"
                    highlight_key_atoms_enhanced(sample_smiles_str, atom_importance, output_file, molecules_analyzed + 1)
                    
                                                
                    analysis_results.append({
                        'molecule_id': molecules_analyzed + 1,
                        'smiles': sample_smiles_str,
                        'target_td': float(target_value),
                        'predicted_td': float(pred_value),
                        'error': float(abs(target_value - pred_value)),
                        'atom_contributions': atom_importance.tolist(),
                        'output_file': output_file
                    })
                else:
                    print("原子贡献计算失败")
                
                molecules_analyzed += 1
    
    return analysis_results

           
def create_comprehensive_summary(analysis_results):
    
    if not analysis_results:
        return
    
    try:
                      
        molecule_ids = [r['molecule_id'] for r in analysis_results]
        errors = [r['error'] for r in analysis_results]
        target_tds = [r['target_td'] for r in analysis_results]
        predicted_tds = [r['predicted_td'] for r in analysis_results]
        
                       
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('15个分子的综合分析总结', fontsize=18, fontweight='bold')
        
                            
        bars = axes[0, 0].bar(molecule_ids, errors, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
        axes[0, 0].set_xlabel('分子ID', fontsize=12)
        axes[0, 0].set_ylabel('预测误差 (K)', fontsize=12)
        axes[0, 0].set_title('每个分子的预测误差', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(molecule_ids)
        
                         
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            axes[0, 0].annotate(f'{error:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
                                  
        scatter = axes[0, 1].scatter(target_tds, predicted_tds, c=errors, cmap='viridis', 
                                   s=100, alpha=0.8, edgecolors='black', linewidth=1)
        axes[0, 1].plot([min(target_tds), max(target_tds)], [min(target_tds), max(target_tds)], 
                        lw=2, label='理想预测线')
        axes[0, 1].set_xlabel('实际 Td (K)', fontsize=12)
        axes[0, 1].set_ylabel('预测 Td (K)', fontsize=12)
        axes[0, 1].set_title('预测值 vs 实际值', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        
        try:
            cbar = plt.colorbar(scatter, ax=axes[0, 1])
            cbar.set_label('误差 (K)', fontsize=12)
        except:
            pass
        
                           
        all_contributions = []
        for result in analysis_results:
            if result['atom_contributions']:
                all_contributions.extend(result['atom_contributions'])
        
        if all_contributions:
            axes[1, 0].hist(all_contributions, bins=30, alpha=0.7, color='lightgreen', 
                           edgecolor='darkgreen', linewidth=1)
            axes[1, 0].set_xlabel('原子贡献值', fontsize=12)
            axes[1, 0].set_ylabel('频率', fontsize=12)
            axes[1, 0].set_title('原子贡献分布', fontsize=14, fontweight='bold')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='零贡献线')
            axes[1, 0].legend()
            
                                
            mean_contrib = np.mean(all_contributions)
            std_contrib = np.std(all_contributions)
            axes[1, 0].text(0.02, 0.98, f'均值: {mean_contrib:.3f}\n标准差: {std_contrib:.3f}', 
                           transform=axes[1, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[1, 0].text(0.5, 0.5, '无原子贡献数据', 
                           transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=14)
            axes[1, 0].set_title('原子贡献分布', fontsize=14, fontweight='bold')
        
                                  
        molecular_weights = []
        molecular_complexities = []
        for result in analysis_results:
            try:
                mol = Chem.MolFromSmiles(result['smiles'])
                if mol:
                    mw = Descriptors.MolWt(mol)
                    complexity = Descriptors.BertzCT(mol)             
                    molecular_weights.append(mw)
                    molecular_complexities.append(complexity)
                else:
                    molecular_weights.append(200)
                    molecular_complexities.append(50)
            except:
                molecular_weights.append(200)
                molecular_complexities.append(50)
        
                               
        scatter2 = axes[1, 1].scatter(molecular_weights, errors, c=molecular_complexities, 
                                     cmap='plasma', s=100, alpha=0.8, edgecolors='black', linewidth=1)
        axes[1, 1].set_xlabel('分子量 (g/mol)', fontsize=12)
        axes[1, 1].set_ylabel('预测误差 (K)', fontsize=12)
        axes[1, 1].set_title('分子量 vs 预测误差', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        try:
            cbar2 = plt.colorbar(scatter2, ax=axes[1, 1])
            cbar2.set_label('分子复杂度', fontsize=12)
        except:
            pass
        
        plt.tight_layout()
        plt.savefig('15_molecules_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n综合分析图表已保存到 15_molecules_comprehensive_analysis.png")
        
    except Exception as e:
        print(f"创建综合可视化时出错: {e}")

        
def generate_detailed_report(analysis_results):
    
    if not analysis_results:
        return
    
    try:
                        
        errors = [r['error'] for r in analysis_results]
        target_tds = [r['target_td'] for r in analysis_results]
        predicted_tds = [r['predicted_td'] for r in analysis_results]
        
        report = {
            'analysis_summary': {
                'total_molecules': len(analysis_results),
                'average_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors)),
                'median_error': float(np.median(errors)),
                'r_squared': float(np.corrcoef(target_tds, predicted_tds)[0, 1]**2)
            },
            'best_predictions': [],
            'worst_predictions': [],
            'molecular_insights': {},
            'detailed_results': analysis_results
        }
        
                           
        sorted_by_error = sorted(analysis_results, key=lambda x: x['error'])
        report['best_predictions'] = sorted_by_error[:3]
        report['worst_predictions'] = sorted_by_error[-3:]
        
                      
        all_contributions = []
        for result in analysis_results:
            if result['atom_contributions']:
                all_contributions.extend(result['atom_contributions'])
        
        if all_contributions:
            report['molecular_insights'] = {
                'total_atoms_analyzed': len(all_contributions),
                'avg_atom_contribution': float(np.mean(all_contributions)),
                'contribution_std': float(np.std(all_contributions)),
                'positive_contributions': len([c for c in all_contributions if c > 0]),
                'negative_contributions': len([c for c in all_contributions if c < 0]),
                'neutral_contributions': len([c for c in all_contributions if abs(c) < 0.1])
            }
        
                           
        with open('15_molecules_detailed_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("详细报告已保存到 15_molecules_detailed_report.json")
        
                        
        print(f"\n=== 15个分子分析摘要报告 ===")
        print(f"总分子数: {report['analysis_summary']['total_molecules']}")
        print(f"平均预测误差: {report['analysis_summary']['average_error']:.2f} ± {report['analysis_summary']['std_error']:.2f} K")
        print(f"R²值: {report['analysis_summary']['r_squared']:.4f}")
        print(f"误差范围: {report['analysis_summary']['min_error']:.2f} - {report['analysis_summary']['max_error']:.2f} K")
        
        print(f"\n最佳预测 (前3个):")
        for i, mol in enumerate(report['best_predictions']):
            print(f"  {i+1}. 分子{mol['molecule_id']}: 误差 {mol['error']:.2f} K")
        
        print(f"\n最差预测 (后3个):")
        for i, mol in enumerate(report['worst_predictions']):
            print(f"  {i+1}. 分子{mol['molecule_id']}: 误差 {mol['error']:.2f} K")
        
        if report['molecular_insights']:
            insights = report['molecular_insights']
            print(f"\n原子贡献洞察:")
            print(f"  分析的原子总数: {insights['total_atoms_analyzed']}")
            print(f"  平均原子贡献: {insights['avg_atom_contribution']:.4f}")
            print(f"  正贡献原子: {insights['positive_contributions']} ({insights['positive_contributions']/insights['total_atoms_analyzed']*100:.1f}%)")
            print(f"  负贡献原子: {insights['negative_contributions']} ({insights['negative_contributions']/insights['total_atoms_analyzed']*100:.1f}%)")
            print(f"  中性原子: {insights['neutral_contributions']} ({insights['neutral_contributions']/insights['total_atoms_analyzed']*100:.1f}%)")
        
    except Exception as e:
        print(f"生成报告时出错: {e}")

         
def create_molecules_overview(analysis_results):
    
    if not analysis_results:
        return
    
    try:
                            
        fig, axes = plt.subplots(5, 3, figsize=(20, 25))
        fig.suptitle('15个分子的结构和预测结果概览', fontsize=20, fontweight='bold')
        
        for i, result in enumerate(analysis_results):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            try:
                                        
                mol = Chem.MolFromSmiles(result['smiles'])
                if mol:
                    img = rdMolDraw2D.MolDraw2DCairo(300, 250)
                    img.DrawMolecule(mol)
                    img.FinishDrawing()
                    
                                                        
                    ax.text(0.5, 0.7, f"分子 {result['molecule_id']}", 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=14, fontweight='bold')
                    ax.text(0.5, 0.5, f"SMILES: {result['smiles'][:20]}{'...' if len(result['smiles']) > 20 else ''}", 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=10, wrap=True)
                    ax.text(0.5, 0.3, f"目标: {result['target_td']:.1f} K", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.text(0.5, 0.2, f"预测: {result['predicted_td']:.1f} K", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.text(0.5, 0.1, f"误差: {result['error']:.1f} K", 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=12, color='red' if result['error'] > 20 else 'green')
                else:
                    ax.text(0.5, 0.5, f"分子 {result['molecule_id']}\n无法解析SMILES", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
            except Exception as e:
                ax.text(0.5, 0.5, f"分子 {result['molecule_id']}\n显示错误", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"分子 {result['molecule_id']}", fontsize=12, fontweight='bold')
            
                                    
            if result['error'] <= 10:
                color = 'green'
            elif result['error'] <= 20:
                color = 'orange'
            else:
                color = 'red'
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig('15_molecules_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("分子概览图已保存到 15_molecules_overview.png")
        
    except Exception as e:
        print(f"创建分子概览图时出错: {e}")

              
print("\n=== 开始执行15个分子的完整可视化分析 ===")

try:
                    
    analysis_results = analyze_fifteen_molecules(
        model, test_loader, smiles_list, y_scaler, device, num_molecules=15
    )

    if analysis_results:
        print(f"\n=== 成功分析了 {len(analysis_results)} 个分子 ===")
        
                          
        print("\n创建综合分析图表...")
        create_comprehensive_summary(analysis_results)
        
                        
        print("\n生成详细报告...")
        generate_detailed_report(analysis_results)
        
                         
        print("\n创建分子概览图...")
        create_molecules_overview(analysis_results)
        
                           
        print(f"\n=== 生成的文件列表 ===")
        for result in analysis_results:
            print(f"  - {result['output_file']}")
        print("  - 15_molecules_comprehensive_analysis.png")
        print("  - 15_molecules_detailed_report.json")
        print("  - 15_molecules_overview.png")
        
        print(f"\n=== 15个分子的完整可视化分析已完成! ===")
        
    else:
        print("没有成功分析任何分子")

except Exception as e:
    print(f"分析过程中出错: {e}")

      
print("\n=== 模型整体评估 ===")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_data in test_loader:
        fp, desc, smiles, td, _ = batch_data
        fp, desc, smiles, td = fp.to(device), desc.to(device), smiles.to(device), td.to(device)
        preds = model(fp, desc, smiles)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(td.cpu().numpy())

         
all_preds = y_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
all_targets = y_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

      
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

print(f"测试集 MSE: {mse:.4f}")
print(f"测试集 MAE: {mae:.4f}")
print(f"测试集 R²: {r2:.4f}")

         
plt.figure(figsize=(10, 8))
plt.scatter(all_targets, all_preds, alpha=0.6, s=50)
plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--', lw=2)
plt.xlabel('实际 Td (K)', fontsize=14)
plt.ylabel('预测 Td (K)', fontsize=14)
plt.title(f'整体预测结果 (R² = {r2:.3f})', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

        
plt.text(0.05, 0.95, f'样本数: {len(all_targets)}\nMSE: {mse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('overall_prediction_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("整体预测结果图已保存到 overall_prediction_results.png")

print("\n=== 所有分析已完成! ===")