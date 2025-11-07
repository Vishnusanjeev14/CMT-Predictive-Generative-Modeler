"""
tabular_cgan.py

Conditional GAN for tabular CMT synthetic data generation.

- Input: ../data/synthetic/cmt_synthetic_dataset.csv
- Output: ../data/synthetic/cmt_synthetic_generated.csv (and model checkpoints in ./checkpoints/)
- Designed for mixed-type tabular data: categorical (gene_variant_id, mutation_type),
  numeric (age_of_onset, motor_score, sensory_score), and label (disease_severity).
- Generator outputs numeric values and categorical logits; categorical values are sampled
  with argmax (or soft sampling during training / temperature annealing can be added).

Usage (example):
    python src/tabular_cgan.py --data ../data/synthetic/cmt_synthetic_dataset.csv \
        --out ../data/synthetic/cmt_synthetic_generated.csv --epochs 400 --batch 256

Notes:
- Requires: torch, numpy, pandas, scikit-learn
- For small GPUs or CPU-only, reduce --batch and --latent-dim, or reduce epochs.
"""

import os
import argparse
import math
import random
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Dataset utilities
# ----------------------------
class CMTTabularDataset(Dataset):
    def __init__(self, df, enc_cat=None, scaler=None, fit=True):
        """
        df: pandas DataFrame with columns:
            gene_variant_id, mutation_type, age_of_onset, motor_score, sensory_score, disease_severity
        enc_cat: OneHotEncoder instance (if None and fit=True, it will be created and fit)
        scaler: StandardScaler for numeric columns (same logic)
        """
        self.df = df.copy().reset_index(drop=True)
        self.cat_cols = ["gene_variant_id", "mutation_type"]
        self.num_cols = ["age_of_onset", "motor_score", "sensory_score"]
        self.label_col = "disease_severity"
        # map label to int
        self.label_map = {"mild": 0, "moderate": 1, "severe": 2}
        if fit:
            self.enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.enc.fit(self.df[self.cat_cols])
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.num_cols].values.astype(float))
        else:
            assert enc_cat is not None and scaler is not None
            self.enc = enc_cat
            self.scaler = scaler

        # Preprocess and store tensors
        cat_enc = self.enc.transform(self.df[self.cat_cols])  # dense array
        num_scaled = self.scaler.transform(self.df[self.num_cols].values.astype(float))
        labels = self.df[self.label_col].map(self.label_map).values
        # We'll also store raw categories for inverse mapping if needed
        self.cat_enc = cat_enc.astype(np.float32)
        self.num_scaled = num_scaled.astype(np.float32)
        self.labels = labels.astype(np.int64)

        # For easier training, produce combined arrays:
        self.x_real = np.hstack([self.num_scaled, self.cat_enc]).astype(np.float32)
        # Sizes for generator/discriminator architecture
        self.num_features = self.num_scaled.shape[1]
        self.cat_dim = self.cat_enc.shape[1]
        self.input_dim = self.num_features + self.cat_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "num": torch.from_numpy(self.num_scaled[idx]),
            "cat": torch.from_numpy(self.cat_enc[idx]),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "x": torch.from_numpy(self.x_real[idx])
        }


# ----------------------------
# Models (Generator & Discriminator)
# ----------------------------
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, num_features, cat_dim, hidden_dims=(256, 256)):
        """
        Generator outputs:
          - numeric predictions (num_features, continuous)
          - categorical logits (cat_dim, we'll split into categories at inference)
        We output a single vector of length num_features + cat_dim (numeric first).
        """
        super().__init__()
        input_dim = latent_dim + condition_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, num_features + cat_dim))
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, z, cond):
        # z: (B, latent_dim), cond: (B, condition_dim)
        x = torch.cat([z, cond], dim=1)
        out = self.net(x)
        # split numeric and categorical logits
        return out  # caller will split numeric / cat logits

class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims=(256, 256)):
        """
        Discriminator receives concatenated real/fake vector (numeric + one-hot cats)
        and condition (one-hot disease severity) optionally appended.
        Outputs single logit (real/fake).
        """
        super().__init__()
        layers = []
        prev = input_dim + condition_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x, cond):
        # x: (B, input_dim), cond: (B, condition_dim)
        h = torch.cat([x, cond], dim=1)
        return self.net(h).squeeze(1)


# ----------------------------
# Training / helper functions
# ----------------------------
def sample_noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim, device=device)

def label_to_onehot(labels, num_classes=3, device='cpu'):
    # labels: LongTensor (B,)
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float().to(device)

def postprocess_generated(num_pred, cat_logits, scaler, enc):
    """
    num_pred: numpy array (B, num_features) in scaled space -> inverse scale
    cat_logits: numpy array (B, cat_dim) logits -> convert to categories by argmax within groups
    enc: OneHotEncoder fitted on the categorical columns in order [gene_variant_id, mutation_type]
         we need to know how many columns correspond to each category. We'll recover feature names.
    Returns: DataFrame with decoded categorical and numeric columns.
    """
    # inverse scale numeric
    num_unscaled = scaler.inverse_transform(num_pred)
    # decode categorical by argmax: we need the splits
    feature_names = enc.get_feature_names_out()
    # OneHotEncoder encodes categories in the order of enc.categories_
    cats = []
    idx = 0
    decoded = {}
    for col_idx, categories in enumerate(enc.categories_):
        k = len(categories)
        logits_block = cat_logits[:, idx:idx + k]
        argmax = np.argmax(logits_block, axis=1)
        decoded[col_idx] = [categories[a] for a in argmax]
        idx += k
    # built DataFrame
    df = pd.DataFrame()
    # numeric names assumed in order
    df["age_of_onset"] = num_unscaled[:, 0].round().astype(int)
    df["motor_score"] = np.clip(num_unscaled[:, 1].round().astype(int), 0, 100)
    df["sensory_score"] = np.clip(num_unscaled[:, 2].round().astype(int), 0, 100)
    # map decoded cats
    # enc.categories_ order corresponds to input cat_cols = ["gene_variant_id", "mutation_type"]
    df["gene_variant_id"] = decoded[0]
    df["mutation_type"] = decoded[1]
    return df

def compute_gradient_penalty(D, real_samples, fake_samples, cond, device):
    """
    Optional: gradient penalty for WGAN-GP style stabilization.
    Not used in vanilla GAN, but kept here if you want to enable it.
    """
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates, cond)
    grads = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones_like(d_interpolates),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(grads.size(0), -1)
    grad_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

# ----------------------------
# Main training function
# ----------------------------
def train_cgan(
    df_path,
    out_path,
    epochs=400,
    batch_size=256,
    latent_dim=64,
    lr=2e-4,
    device=None,
    checkpoint_dir="checkpoints",
    save_every=100
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_path)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(df_path)
    # Build dataset & encoders
    dataset = CMTTabularDataset(df, fit=True)
    enc = dataset.enc
    scaler = dataset.scaler
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # shapes
    num_features = dataset.num_features
    cat_dim = dataset.cat_dim
    input_dim = dataset.input_dim  # numeric + cat one-hot dims
    condition_dim = 3  # mild/moderate/severe one-hot
    # Instantiate models
    G = Generator(latent_dim=latent_dim, condition_dim=condition_dim,
                  num_features=num_features, cat_dim=cat_dim).to(device)
    D = Discriminator(input_dim=input_dim, condition_dim=condition_dim).to(device)
    # Optimizers
    optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # Loss
    adversarial_loss = nn.BCEWithLogitsLoss()
    # training loop
    print(f"Starting CGAN training on device={device} with {len(dataset)} samples.")
    for epoch in range(1, epochs + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        for batch in dataloader:
            real_x = torch.cat([batch["num"], batch["cat"]], dim=1).to(device)  # (B, input_dim)
            labels = batch["label"].to(device)
            cond = label_to_onehot(labels, num_classes=3, device=device)  # (B,3)
            bsize = real_x.size(0)

            # --------------------------------------------
            # Train Discriminator: maximize log(D(x|c)) + log(1 - D(G(z|c)))
            # --------------------------------------------
            optim_D.zero_grad()
            # real
            valid = torch.ones(bsize, device=device)
            fake = torch.zeros(bsize, device=device)
            d_real_logits = D(real_x, cond)
            d_real_loss = adversarial_loss(d_real_logits, valid)
            # fake
            z = sample_noise(bsize, latent_dim, device)
            g_out = G(z, cond)  # (B, num_features + cat_dim)
            num_pred = g_out[:, :num_features]
            cat_logits = g_out[:, num_features:]  # logits (we will treat as one-hot-like)
            # For discriminator input, convert cat_logits to softmax probabilities
            cat_probs = torch.nn.functional.softmax(cat_logits, dim=1)
            fake_x = torch.cat([num_pred, cat_probs], dim=1).detach()
            d_fake_logits = D(fake_x, cond)
            d_fake_loss = adversarial_loss(d_fake_logits, fake)
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optim_D.step()

            # --------------------------------------------
            # Train Generator: minimize log(1 - D(G(z|c))) <-> maximize log(D(G(z|c)))
            # --------------------------------------------
            optim_G.zero_grad()
            z2 = sample_noise(bsize, latent_dim, device)
            g_out2 = G(z2, cond)
            num_pred2 = g_out2[:, :num_features]
            cat_logits2 = g_out2[:, num_features:]
            cat_probs2 = torch.nn.functional.softmax(cat_logits2, dim=1)
            fake_x2 = torch.cat([num_pred2, cat_probs2], dim=1)
            d_fake_logits2 = D(fake_x2, cond)
            g_loss = adversarial_loss(d_fake_logits2, valid)
            g_loss.backward()
            optim_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        g_loss_epoch /= len(dataloader)
        d_loss_epoch /= len(dataloader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}]  D_loss: {d_loss_epoch:.4f}  G_loss: {g_loss_epoch:.4f}")

        # save checkpoints
        if epoch % save_every == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "G_state_dict": G.state_dict(),
                "D_state_dict": D.state_dict(),
                "optim_G": optim_G.state_dict(),
                "optim_D": optim_D.state_dict(),
                "enc": enc,
                "scaler": scaler,
            }, checkpoint_dir / f"cgan_checkpoint_epoch{epoch}.pt")
    # --- end training

    # --- Generate synthetic samples by sampling conditioned on each severity evenly
    print("Generation: Sampling synthetic dataset...")
    n_per_class =  int( (len(dataset)) )  # generate same as original size per class spread below
    samples = []
    # We'll generate same total number as original dataset
    total_gen = len(dataset)
    for class_idx in range(3):
        n_class = total_gen // 3
        labels_fake = torch.full((n_class,), class_idx, dtype=torch.long, device=device)
        cond_fake = label_to_onehot(labels_fake, num_classes=3, device=device)
        z = sample_noise(n_class, latent_dim, device)
        with torch.no_grad():
            g_out = G(z, cond_fake).cpu().numpy()
        num_pred = g_out[:, :num_features]
        cat_logits = g_out[:, num_features:]
        # convert cat_logits to numpy
        num_pred_np = num_pred.astype(np.float32)
        cat_logits_np = cat_logits.astype(np.float32)
        df_gen = postprocess_generated(num_pred_np, cat_logits_np, scaler, enc)
        df_gen["disease_severity"] = ["mild", "moderate", "severe"][class_idx]
        samples.append(df_gen)

    df_out = pd.concat(samples, ignore_index=True)
    # clip realistic bounds
    df_out["age_of_onset"] = df_out["age_of_onset"].clip(5, 95)
    # shuffle
    df_out = df_out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Generated dataset saved to: {out_path}")

    # Save some metadata
    meta = {
        "source_dataset": str(df_path),
        "generated_dataset": str(out_path),
        "n_generated": len(df_out),
        "checkpoint_dir": str(checkpoint_dir),
        "latent_dim": latent_dim,
        "epochs": epochs
    }
    with open(out_path.parent / "generation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return df_out

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Conditional GAN for tabular CMT data")
    p.add_argument("--data", type=str, required=True, help="Path to synthetic CSV (realistic training data)")
    p.add_argument("--out", type=str, required=True, help="Output CSV path for generated data")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_cgan(
        df_path=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        latent_dim=args.latent_dim,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )