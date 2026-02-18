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
# 1. Models: Simple MLP (No Residuals)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4, latent_dim=64):
        super(SimpleMLP, self).__init__()
        # Matches the hidden dimension logic of the previous model for fair capacity comparison
        hidden_dim = max(128, input_dim * 8)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim, bias=False) 
        )

    def forward(self, x):
        z = self.net(x)
        # Deep SVDD still requires projection to the hypersphere
        return F.normalize(z, p=2, dim=1)

# ==========================================
# 2. Standard Deep SVDD Loss
# ==========================================
class DeepSVDDLoss(nn.Module):
    def __init__(self, center):
        super().__init__()
        self.register_buffer('center', F.normalize(center, p=2, dim=0))

    def forward(self, features):
        return torch.mean(torch.sum((features - self.center) ** 2, dim=1))

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
# 4. Federated Averaging with Momentum Logic
# ==========================================
def create_client_loaders(dataset, num_clients, batch_size, method, alpha):
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
            loaders.append(DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=0))
    return loaders

def fedavg_aggregate(global_model, client_weights, client_sizes, server_velocity, momentum):
    """
    Aggrega i pesi basandosi sulla dimensione del dataset del client (FedAvg Standard).
    Formula: W_avg = Sum(W_k * (n_k / n_total))
    """
    global_dict = global_model.state_dict()
    
    # 1. Calcolo del numero totale di campioni
    total_samples = sum(client_sizes)
    
    # 2. Calcolo dei coefficienti di peso (n_k / N)
    # Creiamo un tensore per facilitare le operazioni vettoriali
    coeffs = torch.tensor([n / total_samples for n in client_sizes])
    
    # Dizionario per i nuovi pesi mediati
    avg_weights = {}

    for k in global_dict.keys():
        # Stack dei pesi di tutti i client [num_clients, shape_param...]
        stacked = torch.stack([w[k] for w in client_weights], dim=0)
        
        # Spostiamo i coefficienti sullo stesso device dei pesi
        coeffs = coeffs.to(stacked.device)
        
        # 3. Reshape dei coefficienti per il broadcasting
        # Se stacked ha shape [10, 64, 64], coeffs deve diventare [10, 1, 1]
        view_shape = [len(client_weights)] + [1] * (stacked.ndim - 1)
        coeffs_view = coeffs.view(*view_shape)
        
        # 4. Somma pesata (Weighted Sum) invece di Mean
        # W_avg = sum(W_i * coeff_i)
        avg_weights[k] = (stacked * coeffs_view).sum(dim=0)

    # --- Da qui in poi la logica del Momentum rimane identica ---
    for k in global_dict.keys():
        delta = avg_weights[k] - global_dict[k]
        
        if momentum > 0.0:
            if k not in server_velocity:
                server_velocity[k] = torch.zeros_like(delta)
            
            server_velocity[k] = momentum * server_velocity[k] + delta
            global_dict[k] += server_velocity[k]
        else:
            global_dict[k] = avg_weights[k]
            
    global_model.load_state_dict(global_dict)
    return global_model


def run_single_seed(seed, dataset_name, normal_class, device, split_method, alpha, num_clients, momentum):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    # 1. Load Data
    X_full, y_full, meta = load_data_source(dataset_name)
    X_train, X_rest, y_train, y_rest = train_test_split(X_full, y_full, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    # 2. Move to GPU
    train_ds = GPUDataset(X_train, y_train, normal_class, device, True)
    train_bs = 1024
    eval_bs = 4096

    client_loaders = create_client_loaders(train_ds, num_clients, train_bs, split_method, alpha)
    val_loader = DataLoader(GPUDataset(X_val, y_val, normal_class, device, False), batch_size=eval_bs)
    test_loader = DataLoader(GPUDataset(X_test, y_test, normal_class, device, False), batch_size=eval_bs)

    if not client_loaders: return 0.5, 0.0, 0.0, 0.0, 0.0

    latent_dim = 64
    # CHANGED: Use SimpleMLP instead of ResBlock/MLPEncoder
    global_model = SimpleMLP(input_dim=meta[0], latent_dim=latent_dim).to(device)

    # --- Initial Center Calculation ---
    global_model.eval()
    with torch.no_grad():
        all_features = []
        for loader in client_loaders:
            for x, _ in loader:
                all_features.append(global_model(x))
                break 
        
        if all_features:
            center = torch.cat(all_features, dim=0).mean(dim=0)
            center = F.normalize(center, p=2, dim=0)
        else:
            center = F.normalize(torch.randn(latent_dim, device=device), p=2, dim=0)
    
    loss_fn = DeepSVDDLoss(center)

    # ==========================================
    # FEDAVG LOOP
    # ==========================================
    n_rounds = 10
    local_epochs = 5
    lr = 5e-4
    
    # Server Momentum State
    server_velocity = {} 
        
    # ... (dentro run_single_seed) ...

    for round_idx in range(n_rounds):
        local_weights = []
        local_sizes = []  # <--- NUOVA LISTA per salvare i n_samples
        
        global_state = copy.deepcopy(global_model.state_dict())
        
        for loader in client_loaders:
            client_net = copy.deepcopy(global_model)
            client_net.load_state_dict(global_state)
            client_net.train()
            
            # Salviamo il numero di campioni di questo client
            # Nota: loader.dataset è il Subset, quindi len() è corretto
            n_samples = len(loader.dataset) # <--- CATTURA DIMENSIONE
            
            optimizer = torch.optim.AdamW(client_net.parameters(), lr=lr, weight_decay=1e-3)
            
            for _ in range(local_epochs):
                for x, _ in loader:
                    if x.size(0) < 2: continue
                    optimizer.zero_grad(set_to_none=True)
                    z = client_net(x)
                    loss = loss_fn(z)
                    loss.backward()
                    optimizer.step()
            
            local_weights.append(client_net.state_dict())
            local_sizes.append(n_samples) # <--- SALVA DIMENSIONE
        
        # Passiamo local_sizes alla nuova funzione
        fedavg_aggregate(global_model, local_weights, local_sizes, server_velocity, momentum)

    # ==========================================
    # EVALUATION
    # ==========================================
    def get_scores(loader, model, center):
        model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for x, y in loader:
                z = model(x)
                dist = torch.sum((z - center) ** 2, dim=1)
                scores.append(dist)
                labels.append(y)
        return torch.cat(scores).cpu().numpy(), torch.cat(labels).cpu().numpy()

    val_scores, val_labels = get_scores(val_loader, global_model, center)
    test_scores, test_labels = get_scores(test_loader, global_model, center)

    if len(np.unique(val_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
    else:
        best_thresh = np.percentile(val_scores, 95)

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
# 5. Main Execution
# ==========================================
def evaluate_scenarios():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    seeds = [1, 6, 31, 42, 53]
    scenarios = [
        ('IID', 'iid', 0.0),
        ('Dirichlet (0.5)', 'dirichlet', 0.5),
        ('Dirichlet (0.1)', 'dirichlet', 0.1),
        ('Dirichlet (0.01)', 'dirichlet', 0.01),
    ]
    datasets = [
        ('iris', range(3)),
        ('breast_cancer', range(2)),
        ('titanic', range(2)),
        ('adult', range(2)),
        ('covertype', range(7)) 
    ]
    
    momentum_settings = [0.0, 0.9]

    # Set Pandas options globally once
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)

    for momentum in momentum_settings:
        m_label = "NO MOMENTUM" if momentum == 0.0 else f"MOMENTUM {momentum}"
        print(f"\n\n{'='*50}")
        print(f"STARTING EXPERIMENT: SIMPLE MLP + FEDAVG ({m_label})")
        print(f"{'='*50}")

        # Pre-load data
        for d_name, _ in datasets:
            load_data_source(d_name)

        # Store results in a list to create one big DataFrame later if needed
        # But we will print incrementally
        
        total_tasks = len(scenarios) * sum(len(c) for _, c in datasets) * len(seeds)
        pbar = tqdm(total=total_tasks, desc=f"Progress ({m_label})")

        # --- RESTRUCTURED LOOP FOR INCREMENTAL PRINTING ---
        # We iterate Datasets FIRST, so we can print the table for a dataset 
        # as soon as all its scenarios are done.
        
        for d_name, d_classes in datasets:
            dataset_results = [] # Store results just for this dataset
            
            for s_name, s_method, s_alpha in scenarios:
                
                class_metrics = {'auc': [], 'acc': [], 'pre': [], 'rec': [], 'f1': []}

                for cls in d_classes:
                    metrics_per_seed = []
                    for seed in seeds:
                        res = run_single_seed(seed, d_name, cls, device, s_method, s_alpha, 3, momentum)
                        metrics_per_seed.append(res)
                        pbar.update(1)

                    # Extract and mean over seeds
                    aucs = [r[0] for r in metrics_per_seed]
                    accs = [r[1] for r in metrics_per_seed]
                    pres = [r[2] for r in metrics_per_seed]
                    recs = [r[3] for r in metrics_per_seed]
                    f1s = [r[4] for r in metrics_per_seed]

                    class_metrics['auc'].append(np.mean(aucs))
                    class_metrics['acc'].append(np.mean(accs))
                    class_metrics['pre'].append(np.mean(pres))
                    class_metrics['rec'].append(np.mean(recs))
                    class_metrics['f1'].append(np.mean(f1s))

                def fmt(arr):
                    return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

                dataset_results.append({
                    "Scenario": s_name,
                    "Momentum": momentum,
                    "AUROC": fmt(class_metrics['auc']),
                    "Accuracy": fmt(class_metrics['acc']),
                    "Precision": fmt(class_metrics['pre']),
                    "Recall": fmt(class_metrics['rec']),
                    "F1-Score": fmt(class_metrics['f1'])
                })
            
            # --- PRINT IMMEDIATELY AFTER DATASET FINISHES ---
            print(f"\n\n--- {d_name.upper()} RESULTS ({m_label}) ---")
            df = pd.DataFrame(dataset_results)
            print(df) # Use standard print(df) with options set
            
            # Save CSV incrementally
            fname = f"{d_name}_fedavg_simplemlp_mom{str(momentum).replace('.','')}.csv"
            df.to_csv(fname, index=False)

        pbar.close()

if __name__ == "__main__":
    evaluate_scenarios()
