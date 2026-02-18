import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm


# ==========================================
# 1. Models: Residual Spherical Encoder
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


def linear_mixup(z, alpha=0.2):
    """
    Standard Linear Interpolation (Euclidean Mixup).
    Replaces Geodesic SLERP.
    """
    if z.size(0) < 2: return z
    lam = float(np.random.beta(alpha, alpha)) 
    perm_indices = torch.randperm(z.size(0), device=z.device)
    z_perm = z[perm_indices]
    
    # Linear convex combination
    return (1.0 - lam) * z + lam * z_perm


# ==========================================
# 3. Optimized Data Loading (Full VRAM)
# ==========================================
class GPUDataset(Dataset):
    """
    Moves entire dataset to GPU memory upon initialization.
    Eliminates PCIe transfer overhead during training.
    """

    def __init__(self, X, y, normal_class, device, train=True):
        # Convert to tensor and move to device immediately
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        # Filtering on CPU first to save transfer (optional, but cleaner)
        mask = (y == normal_class)

        if train:
            self.data = X[mask].to(device)
            # Dummy labels for train to save memory, we don't use them
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

    # (Loading logic remains identical to your script)
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
# 4. Federated Logic
# ==========================================
def create_client_loaders(dataset, num_clients, batch_size, method, alpha):
    # Dataset is already on GPU, so we just need indices
    indices = np.arange(len(dataset))
    n_samples = len(indices)
    np.random.shuffle(indices)

    if method == 'iid' or n_samples < num_clients:
        splits = np.array_split(indices, num_clients)
    else:
        alpha = float(alpha)
        eps = 1e-12
        props = np.random.dirichlet(np.full(num_clients, max(alpha, eps)))
        props = np.clip(props, eps, None)
        props = props / props.sum()

        if n_samples >= num_clients:
            counts = np.ones(num_clients, dtype=np.int64)
            remaining = n_samples - num_clients
            extra = np.random.multinomial(remaining, props)
            counts += extra
        else:
            counts = np.zeros(num_clients, dtype=np.int64)
            counts[:n_samples] = 1

        split_pts = np.cumsum(counts)[:-1]
        splits = np.split(indices, split_pts)

    loaders = []
    for idxs in splits:
        if len(idxs) > 0:
            sub = torch.utils.data.Subset(dataset, idxs)
            # num_workers=0 is FASTEST when data is already on GPU
            loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders


def run_single_seed(seed, dataset_name, normal_class, device, split_method, alpha, num_clients):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # 1. Load Data (CPU)
    X_full, y_full, meta = load_data_source(dataset_name)
    X_train, X_rest, y_train, y_rest = train_test_split(X_full, y_full, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    # 2. Move to GPU (FastTensorDataset)
    train_ds = GPUDataset(X_train, y_train, normal_class, device, True)

    train_bs = 1024
    eval_bs = 4096

    client_loaders = create_client_loaders(train_ds, num_clients, train_bs, split_method, alpha)
    val_loader = DataLoader(GPUDataset(X_val, y_val, normal_class, device, False), batch_size=eval_bs)
    test_loader = DataLoader(GPUDataset(X_test, y_test, normal_class, device, False), batch_size=eval_bs)

    if not client_loaders: return 0.5, 0.0, 0.0, 0.0, 0.0

    latent_dim = 64
    model = MLPEncoder(input_dim=meta[0], latent_dim=latent_dim).to(device)

    # --- Initial Center Calculation ---
    model.eval()
    with torch.no_grad():
        local_means = []
        for loader in client_loaders:
            client_features = []
            for x, _ in loader:
                client_features.append(model(x))
            if client_features:
                client_features = torch.cat(client_features, dim=0)
                local_means.append(torch.mean(client_features, dim=0))

        if local_means:
            center = torch.stack(local_means).mean(dim=0)
            center = F.normalize(center, p=2, dim=0)
        else:
            center = F.normalize(torch.randn(latent_dim, device=device), p=2, dim=0)

    loss_fn = DeepSVDDLoss(center)
    client_net = copy.deepcopy(model)

    # ==========================================
    # FedAvg Logic (Replaces One-Shot Ensemble)
    # ==========================================
    rounds = 10  # Standard FedAvg iterations
    
    for _ in range(rounds):
        global_state = model.state_dict()
        local_weights = []
        client_sizes = []

        # Train Clients
        for loader in client_loaders:
            # Load global weights
            client_net.load_state_dict(global_state)
            client_net.train()
            
            # Reset Optimizer per round/client
            opt = torch.optim.AdamW(client_net.parameters(), lr=5e-4, weight_decay=1e-3)
            
            # Local Epochs (fewer than One-Shot because we iterate rounds)
            local_epochs = 2 

            for _ in range(local_epochs):
                for x, _ in loader:
                    if x.size(0) < 2: continue
                    opt.zero_grad(set_to_none=True)
                    z = client_net(x)
                    
                    # Apply Linear Mixup (Replaces Geodesic)
                    if np.random.random() > 0.3:
                        z = linear_mixup(z, alpha=0.4)
                        
                    loss = loss_fn(z)
                    loss.backward()
                    opt.step()

            local_weights.append({k: v.clone() for k, v in client_net.state_dict().items()})
            client_sizes.append(len(loader.dataset))

        # Aggregation (FedAvg)
        total_samples = sum(client_sizes)
        if total_samples > 0:
            avg_state = {}
            for k, v in global_state.items():
                avg_state[k] = torch.zeros_like(v, dtype=torch.float32)

            for w, size in zip(local_weights, client_sizes):
                scale = size / total_samples
                for k in w:
                    avg_state[k] += w[k] * scale

            model.load_state_dict(avg_state)

    # ==========================================
    # Evaluation (Global Model, not Ensemble)
    # ==========================================
    model.eval()

    # 1. Validation to find Threshold
    scores, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            z = model(x)
            s = torch.sum((z - center) ** 2, dim=1) # Distance to center
            scores.append(s)
            labels.append(y)
    
    val_scores = torch.cat(scores).cpu().numpy()
    val_labels = torch.cat(labels).cpu().numpy()

    if len(np.unique(val_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
    else:
        best_thresh = np.percentile(val_scores, 95)

    # 2. Test Set Evaluation
    scores, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            z = model(x)
            s = torch.sum((z - center) ** 2, dim=1)
            scores.append(s)
            labels.append(y)

    test_scores = torch.cat(scores).cpu().numpy()
    test_labels = torch.cat(labels).cpu().numpy()

    preds = (test_scores > best_thresh).astype(int)

    try:
        auc = roc_auc_score(test_labels, test_scores)
    except:
        auc = 0.5

    # 5. Return Results
    return (auc,
            accuracy_score(test_labels, preds),
            precision_score(test_labels, preds, zero_division=0),
            recall_score(test_labels, preds, zero_division=0),
            f1_score(test_labels, preds, zero_division=0))

# ==========================================
# 5. Main Execution
# ==========================================
def evaluate_scenarios():
    # Force CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Running on CPU (will be slow).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # Enable CUDA Benchmarking for constant input sizes
        torch.backends.cudnn.benchmark = True

    seeds = [1, 6, 31, 42, 53]
    scenarios = [
        ('IID', 'iid', 0.0),
        ('Dirichlet (0.5)', 'dirichlet', 0.5),
        ('Dirichlet (0.1)', 'dirichlet', 0.1),
        ('Dirichlet (0.01)', 'dirichlet', 0.01),
    ]
    # Datasets from Code 1
    datasets = [
        ('iris', range(3)),
        ('breast_cancer', range(2)),
        ('titanic', range(2)),
        ('adult', range(2)),
        ('covertype', range(7))
    ]

    print(f"Running OPTIMIZED FedAvg Linear SVDD on {device}...")

    # Pre-load data to cache
    for d_name, _ in datasets:
        print(f"Loading {d_name} into RAM...")
        load_data_source(d_name)

    all_results = {name: [] for name, _ in datasets}

    # Progress bar logic
    total_tasks = len(scenarios) * sum(len(c) for _, c in datasets) * len(seeds)
    pbar = tqdm(total=total_tasks, desc="Total Progress")

    # SEQUENTIAL EXECUTION (Faster than threads due to GIL/GPU Context overhead)
    for s_name, s_method, s_alpha in scenarios:
        for d_name, d_classes in datasets:

            # Temporary storage to aggregate across ALL classes for this dataset
            class_metrics = {'auc': [], 'acc': [], 'pre': [], 'rec': [], 'f1': []}

            for cls in d_classes:
                metrics_per_seed = []
                for seed in seeds:
                    res = run_single_seed(seed, d_name, cls, device, s_method, s_alpha, 3)
                    metrics_per_seed.append(res)
                    pbar.update(1)

                # Aggregate results for THIS class across all seeds
                aucs = [r[0] for r in metrics_per_seed]
                accs = [r[1] for r in metrics_per_seed]
                pres = [r[2] for r in metrics_per_seed]
                recs = [r[3] for r in metrics_per_seed]
                f1s = [r[4] for r in metrics_per_seed]

                # Store the mean of this class into the dataset accumulator
                class_metrics['auc'].append(np.mean(aucs))
                class_metrics['acc'].append(np.mean(accs))
                class_metrics['pre'].append(np.mean(pres))
                class_metrics['rec'].append(np.mean(recs))
                class_metrics['f1'].append(np.mean(f1s))

            # Helper to format Mean ± Std across ALL classes
            def fmt(arr):
                return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

            # Append ONE row per Scenario/Dataset (aggregated over all classes)
            all_results[d_name].append({
                "Scenario": s_name,
                "AUROC": fmt(class_metrics['auc']),
                "Accuracy": fmt(class_metrics['acc']),
                "Precision": fmt(class_metrics['pre']),
                "Recall": fmt(class_metrics['rec']),
                "F1-Score": fmt(class_metrics['f1'])
            })

    pbar.close()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    for d_name, _ in datasets:
        print(f"\n\n{'#' * 30}\n  TABLE: {d_name.upper()} Results\n{'#' * 30}")
        if all_results[d_name]:
            df = pd.DataFrame(all_results[d_name])
            print(df.to_string(index=False))
            df.to_csv(f"{d_name}_results.csv", index=False)
        else:
            print("No results.")


if __name__ == "__main__":
    evaluate_scenarios()
