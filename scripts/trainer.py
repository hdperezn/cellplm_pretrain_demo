import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
import os

# Local CellPLM modules
from CellPLM.model.omicsformer import OmicsFormer

# === Dataset Loader ===
class AnnDataset(Dataset):
    def __init__(self, adata, gene_list):
        self.adata = adata
        self.gene_list = gene_list
        self.gene_idx = [i for i, g in enumerate(adata.var_names) if g in gene_list]
        self.X = adata.X[:, self.gene_idx].todense() if hasattr(adata.X, 'todense') else adata.X[:, self.gene_idx]

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        return {
            'x': torch.FloatTensor(self.X[idx].A1 if hasattr(self.X[idx], 'A1') else self.X[idx]),
            'batch': 0  # can be adapted if adata.obs["batch"] exists
        }

# === Load data ===
adata = sc.read_h5ad("data/demo_train.h5ad")
gene_list = list(adata.var_names)
dataset = AnnDataset(adata, gene_list)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# === Build model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OmicsFormer(
    gene_list=gene_list,
    enc_mod='transformer',
    enc_hid=128,
    enc_layers=3,
    post_latent_dim=64,
    dec_mod='mlp',
    dec_hid=128,
    dec_layers=2,
    out_dim=len(gene_list),
    batch_num=1,
    latent_mod='gmvae',
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Training Loop ===
EPOCHS = 20
os.makedirs("ckpt", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x = batch['x'].to(device)
        x_dict = {
            'x': x,
            'batch': torch.zeros(x.size(0), dtype=torch.long).to(device)
        }

        x_dict = model.mask_model.apply_mask(x_dict)

        output, loss = model(x_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), f"ckpt/epoch_{epoch+1}.pt")
