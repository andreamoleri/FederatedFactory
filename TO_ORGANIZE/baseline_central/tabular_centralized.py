import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==========================================
# 1. Models: Residual Spherical Encoder
#    (Must match FL script exactly)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=64):
        super(MLPEncoder, self).__init__()
        hidden_dim = max(128, input_dim * 8)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim)
        )
        self.head = nn.Linear(hidden_dim, latent_dim, bias=False)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        z = self.head(x)
        return F.normalize(z, p=2, dim=1)


# ==========================================
# 2. Optimized Loss & Math
# ==========================================
class DeepSVDDLoss(nn.Module):
    def __init__(self, center):
        super().__init__()
        self.register_buffer('center', F.normalize(center, p=2, dim=0))

    def forward(self, features):
        return torch.mean(torch.sum((features - self.center) ** 2, dim=1))


@torch.jit.script
def slerp(low, high, val: float):
    omega = torch.sum(low * high, dim=1, keepdim=True)
    omega = torch.clamp(omega, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(omega)
    sin_theta = torch.sin(theta)

    mask = (sin_theta > 1e-6).float()
    scale_0 = torch.sin((1.0 - val) * theta) / (sin_theta + 1e-8)
    scale_1 = torch.sin(val * theta) / (sin_theta + 1e-8)

    linear_0 = 1.0 - val
    linear_1 = val

    s0 = mask * scale_0 + (1.0 - mask) * linear_0
    s1 = mask * scale_1 + (1.0 - mask) * linear_1
    return s0 * low + s1 * high


def geodesic_mixup(z, alpha=0.2):
    if z.size(0) < 2: return z
    lam = float(np.random.beta(alpha, alpha))
    perm_indices = torch.randperm(z.size(0), device=z.device)
    z_perm = z[perm_indices]
    return slerp(z, z_perm, lam)


# ==========================================
# 3. Optimized Data Loading
# ==========================================
class GPUDataset(Dataset):
    def __init__(self, X, y, normal_class, device, train=True):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        mask = (y == normal_class)

        if train:
            # CENTRALIZED: Load ALL normal data
            self.data = X[mask].to(device)
            self.labels = torch.zeros(len(self.data), dtype=torch.long, device=device)
        else:
            self.data = X.to(device)
            self.labels = torch.where(mask, 0, 1).long().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


_DATA_CACHE = {}


def load_data_source(dataset_name):
    if dataset_name in _DATA_CACHE:
        return _DATA_CACHE[dataset_name]

    if dataset_name == 'iris':
        d = load_iris()
        X, y = StandardScaler().fit_transform(d.data), d.target
        meta = [X.shape[1]]
    elif dataset_name == 'breast_cancer':
        d = load_breast_cancer()
        X, y = StandardScaler().fit_transform(d.data), d.target
        meta = [X.shape[1]]
    elif dataset_name == 'titanic':
        X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        X = X[features].copy()
        X['age'] = X['age'].fillna(X['age'].median())
        X['fare'] = X['fare'].fillna(X['fare'].median())
        le = LabelEncoder()
        X['sex'] = le.fit_transform(X['sex'].astype(str))
        X = RobustScaler().fit_transform(X)
        y = y.astype(int).values
        meta = [X.shape[1]]
    elif dataset_name == 'adult':
        X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())
        X = RobustScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        meta = [X.shape[1]]
    elif dataset_name == 'covertype':
        X, y = fetch_openml("covertype", version=4, as_frame=True, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        meta = [X.shape[1]]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    _DATA_CACHE[dataset_name] = (X, y, meta)
    return X, y, meta


# ==========================================
# 4. Centralized Training Logic
# ==========================================
def run_centralized_experiment(seed, dataset_name, normal_class, device):
    # 1. Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # 2. Data Loading (MUST MATCH FL SCRIPT SPLITS)
    X_full, y_full, meta = load_data_source(dataset_name)
    X_train, X_rest, y_train, y_rest = train_test_split(X_full, y_full, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    # 3. Setup
    train_ds = GPUDataset(X_train, y_train, normal_class, device, True)
    val_ds = GPUDataset(X_val, y_val, normal_class, device, False)
    test_ds = GPUDataset(X_test, y_test, normal_class, device, False)

    if len(train_ds) == 0: return 0.5, 0.0, 0.0, 0.0, 0.0

    # Batch sizes: High for speed/stability in centralized
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=4096)
    test_loader = DataLoader(test_ds, batch_size=4096)

    # 4. Model & Init
    latent_dim = 64
    model = MLPEncoder(input_dim=meta[0], latent_dim=latent_dim).to(device)

    # Data-Dependent Initialization (using a subset of train data)
    model.eval()
    with torch.no_grad():
        features = []
        for x, _ in train_loader:
            features.append(model(x))
            if len(features) * 1024 > 5000: break

        if features:
            center = torch.mean(torch.cat(features, dim=0), dim=0)
            center = F.normalize(center, p=2, dim=0)
        else:
            center = F.normalize(torch.randn(latent_dim, device=device), p=2, dim=0)

    loss_fn = DeepSVDDLoss(center)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 5. Training (Fairness: 50 Epochs is comparable to FL's multiple rounds)
    epochs = 50
    model.train()

    for epoch in range(epochs):
        for x, _ in train_loader:
            if x.size(0) < 2: continue
            optimizer.zero_grad(set_to_none=True)
            z = model(x)
            # Mixup augmentation
            if np.random.random() > 0.3:
                z = geodesic_mixup(z, alpha=0.4)
            loss = loss_fn(z)
            loss.backward()
            optimizer.step()

    # 6. Evaluation
    model.eval()

    # Get Val threshold
    scores, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            z = model(x)
            scores.append(torch.sum((z - center) ** 2, dim=1))
            labels.append(y)

    val_scores = torch.cat(scores).cpu().numpy()
    val_labels = torch.cat(labels).cpu().numpy()

    if len(np.unique(val_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
    else:
        best_thresh = np.percentile(val_scores, 95)

    # Get Test Metrics
    scores, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            z = model(x)
            scores.append(torch.sum((z - center) ** 2, dim=1))
            labels.append(y)

    test_scores = torch.cat(scores).cpu().numpy()
    test_labels = torch.cat(labels).cpu().numpy()
    preds = (test_scores > best_thresh).astype(int)

    try:
        auc = roc_auc_score(test_labels, test_scores)
    except:
        auc = 0.5

    return (auc,
            accuracy_score(test_labels, preds),
            precision_score(test_labels, preds, zero_division=0),
            recall_score(test_labels, preds, zero_division=0),
            f1_score(test_labels, preds, zero_division=0))


# ==========================================
# 5. Main Loop
# ==========================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    seeds = [1, 6, 31, 42, 53]  # Keep consistent with FL script
    datasets = [
        ('iris', range(3)),
        ('breast_cancer', range(2)),
        ('titanic', range(2)),
        ('adult', range(2)),       # Uncomment if needed
        ('covertype', range(7))    # Uncomment if needed
    ]

    print(f"Running CENTRALIZED BASELINE on {device}...")

    # Cache data
    for d_name, _ in datasets:
        load_data_source(d_name)

    final_results = []

    total_steps = sum(len(classes) for _, classes in datasets) * len(seeds)
    pbar = tqdm(total=total_steps, desc="Processing")

    for d_name, d_classes in datasets:

        # --- UPDATED: Added lists for Precision (d_pre) and Recall (d_rec) ---
        d_auc, d_acc, d_pre, d_rec, d_f1 = [], [], [], [], []

        for cls in d_classes:
            for seed in seeds:
                auc, acc, pre, rec, f1 = run_centralized_experiment(seed, d_name, cls, device)
                d_auc.append(auc)
                d_acc.append(acc)
                d_pre.append(pre)  # <--- Collect Precision
                d_rec.append(rec)  # <--- Collect Recall
                d_f1.append(f1)
                pbar.update(1)

        # --- UPDATED: Calculate Mean/Std for Precision and Recall ---
        row = {
            "Dataset": d_name.upper(),
            "AUROC": f"{np.mean(d_auc):.4f} ± {np.std(d_auc):.4f}",
            "Accuracy": f"{np.mean(d_acc):.4f} ± {np.std(d_acc):.4f}",
            "Precision": f"{np.mean(d_pre):.4f} ± {np.std(d_pre):.4f}", # <--- Add to row
            "Recall": f"{np.mean(d_rec):.4f} ± {np.std(d_rec):.4f}",     # <--- Add to row
            "F1-Score": f"{np.mean(d_f1):.4f} ± {np.std(d_f1):.4f}"
        }
        final_results.append(row)

    pbar.close()

    print("\n" + "=" * 50)
    print("  CENTRALIZED BASELINE (Upper Bound)")
    print("=" * 50)
    df = pd.DataFrame(final_results)
    print(df.to_string(index=False))
    print("=" * 50)
    df.to_csv("centralized_upper_bound.csv", index=False)