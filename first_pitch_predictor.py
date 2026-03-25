import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pitch_data import get_transformed_data
import pandas as pd
import numpy as np
import torch.optim as optim

class first_pitch_predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_pitches, n_zones, n_targets):
        super(first_pitch_predictor, self).__init__()
        self.n_pitches = n_pitches
        self.n_zones = n_zones
        self.n_targets = n_targets
        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

        self.pitch_type_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, n_pitches),
        )

        self.strike_zone_model = nn.Sequential(
            nn.Linear(hidden_dim + n_pitches, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, n_zones),
        )

        # Outputs mean and log_var for each continuous target
        self.pitch_mean_model = nn.Sequential(
            nn.Linear(hidden_dim + n_pitches + n_zones, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, n_targets)
        )

        self.pitch_logvar_model = nn.Sequential(
            nn.Linear(hidden_dim + n_pitches + n_zones, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, n_targets)
        )

    def forward(self, x):
        """Training forward: conditions on soft probs for differentiability."""
        x = self.batch_norm(x)
        shared = self.shared_encoder(x)

        pitch_type_logits = self.pitch_type_model(shared)
        pitch_type_probs = torch.softmax(pitch_type_logits, dim=-1)

        sz_input = torch.cat([shared, pitch_type_probs], dim=-1)
        strike_zone_logits = self.strike_zone_model(sz_input)
        strike_zone_probs = torch.softmax(strike_zone_logits, dim=-1)

        cont_input = torch.cat([shared, pitch_type_probs, strike_zone_probs], dim=-1)
        pitch_mean = self.pitch_mean_model(cont_input)
        pitch_logvar = self.pitch_logvar_model(cont_input)
        pitch_logvar = torch.clamp(pitch_logvar, min=-10, max=2)

        return pitch_type_logits, strike_zone_logits, pitch_mean, pitch_logvar

    def _get_conditional_outputs(self, shared, pitch_type_onehot):
        """Get zone and continuous distributions conditioned on a specific pitch type."""
        sz_input = torch.cat([shared, pitch_type_onehot], dim=-1)
        strike_zone_logits = self.strike_zone_model(sz_input)
        return strike_zone_logits

    def sample(self, x, n_samples=1):
        """Sample from predicted distributions with proper conditional chain:
        pitch_type ~ Cat(softmax(logits))
        zone | pitch_type ~ Cat(softmax(zone_logits(shared, pitch_type)))
        continuous | pitch_type, zone ~ N(mu(shared, pt, zone), sigma^2(shared, pt, zone))
        """
        self.eval()
        with torch.no_grad():
            x = self.batch_norm(x)
            shared = self.shared_encoder(x)

            pitch_type_logits = self.pitch_type_model(shared)
            pitch_type_probs = torch.softmax(pitch_type_logits, dim=-1)

            samples = []
            for _ in range(n_samples):
                sampled_pt = torch.multinomial(pitch_type_probs, 1).squeeze(-1)
                pt_onehot = torch.zeros_like(pitch_type_probs)
                pt_onehot.scatter_(1, sampled_pt.unsqueeze(-1), 1.0)

                sz_input = torch.cat([shared, pt_onehot], dim=-1)
                strike_zone_logits = self.strike_zone_model(sz_input)
                strike_zone_probs = torch.softmax(strike_zone_logits, dim=-1)
                sampled_sz = torch.multinomial(strike_zone_probs, 1).squeeze(-1)
                sz_onehot = torch.zeros_like(strike_zone_probs)
                sz_onehot.scatter_(1, sampled_sz.unsqueeze(-1), 1.0)

                cont_input = torch.cat([shared, pt_onehot, sz_onehot], dim=-1)
                pitch_mean = self.pitch_mean_model(cont_input)
                pitch_logvar = self.pitch_logvar_model(cont_input)
                pitch_logvar = torch.clamp(pitch_logvar, min=-10, max=2)
                std = torch.exp(0.5 * pitch_logvar)
                sampled_continuous = pitch_mean + std * torch.randn_like(std)

                samples.append({
                    'pitch_type': sampled_pt,
                    'zone': sampled_sz,
                    'continuous': sampled_continuous
                })
        return samples
    
def gaussian_nll_loss(mean, logvar, target):
    """Negative log-likelihood for diagonal Gaussian."""
    var = torch.exp(logvar)
    return 0.5 * (logvar + (target - mean) ** 2 / var).mean()

def train_first_pitch_model(model, X, y_pitch_type, y_strike_zone, y_pitch, X_val, y_pitch_type_val, y_strike_zone_val, y_pitch_val, epochs=10, batch_size=256, lr=1e-3):
    import copy
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), 
        torch.tensor(y_pitch_type, dtype=torch.long), 
        torch.tensor(y_strike_zone, dtype=torch.long), 
        torch.tensor(y_pitch, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion_pitch_type = nn.CrossEntropyLoss()
    criterion_strike_zone = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_nll = float('inf')
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_pt = 0
        total_sz = 0
        total_p = 0
        for batch_X, batch_y_pt, batch_y_sz, batch_y_p in loader:
            batch_X, batch_y_pt, batch_y_sz, batch_y_p = batch_X.to(device), batch_y_pt.to(device), batch_y_sz.to(device), batch_y_p.to(device)
            
            optimizer.zero_grad()
            
            pitch_type_logits, strike_zone_logits, pitch_mean, pitch_logvar = model(batch_X)
            
            loss_pt = criterion_pitch_type(pitch_type_logits, batch_y_pt)
            loss_sz = criterion_strike_zone(strike_zone_logits, batch_y_sz)
            loss_p = gaussian_nll_loss(pitch_mean, pitch_logvar, batch_y_p)
            
            loss = loss_pt + loss_sz + loss_p
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pt += loss_pt.item()
            total_sz += loss_sz.item()
            total_p += loss_p.item()
        
        scheduler.step()
        n = len(loader)
        val_loss = get_val_error(model, X_val, y_pitch_type_val, y_strike_zone_val, y_pitch_val)
        model.train()
        
        val_nll = val_loss[4]
        marker = ''
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = copy.deepcopy(model.state_dict())
            marker = ' *'
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f} (PT: {total_pt/n:.4f}, SZ: {total_sz/n:.4f}, NLL: {total_p/n:.4f}) | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f'Validation - Pitch Type Acc: {val_loss[0]:.2%}, Strike Zone Acc: {val_loss[1]:.2%}, Cont NLL: {val_nll:.4f}{marker}')
    
    print(f'\nRestoring best model (val NLL: {best_val_nll:.4f})')
    model.load_state_dict(best_state)
    return model

def get_val_error(model, X_val, y_pitch_type_val, y_strike_zone_val, y_pitch_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_pitch_type_val = torch.tensor(y_pitch_type_val, dtype=torch.long).to(device)
        y_strike_zone_val = torch.tensor(y_strike_zone_val, dtype=torch.long).to(device)
        y_pitch_val = torch.tensor(y_pitch_val, dtype=torch.float32).to(device)
        
        pitch_type_logits, strike_zone_logits, pitch_mean, pitch_logvar = model(X_val)
        
        criterion_pitch_type = nn.CrossEntropyLoss()
        criterion_strike_zone = nn.CrossEntropyLoss()
        
        loss_pt = criterion_pitch_type(pitch_type_logits, y_pitch_type_val)
        loss_sz = criterion_strike_zone(strike_zone_logits, y_strike_zone_val)
        loss_nll = gaussian_nll_loss(pitch_mean, pitch_logvar, y_pitch_val)

        accuracy_pt = (pitch_type_logits.argmax(dim=-1) == y_pitch_type_val).float().mean().item()
        accuracy_sz = (strike_zone_logits.argmax(dim=-1) == y_strike_zone_val).float().mean().item()
        
    return accuracy_pt, accuracy_sz, loss_pt.item(), loss_sz.item(), loss_nll.item()

if __name__ == "__main__":
    train = {}
    test = {}
    train['game_context'] = pd.read_csv('train_game_context_2020-05-12_2025-08-01.csv')
    train['first_pitch'] = pd.read_csv('train_first_pitch_2020-05-12_2025-08-01.csv')
    test['game_context'] = pd.read_csv('test_game_context_2025-08-01_2025-11-03.csv')
    test['first_pitch'] = pd.read_csv('test_first_pitch_2025-08-01_2025-11-03.csv')
    X_df = pd.get_dummies(train['game_context'].drop(columns=['game_date', 'game_id']), drop_first=True).astype(float)
    train_columns = X_df.columns
    X = X_df.values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X = (X - X_mean) / X_std
    y = train['first_pitch']
    y['pitch_type'] = y['pitch_type'].fillna(y['pitch_type'].mode()[0])
    y.fillna(y.groupby('pitch_type').transform('mean'), inplace=True)

    pitch_type_categories = pd.CategoricalDtype(categories=sorted(y['pitch_type'].unique()))
    zone_categories = pd.CategoricalDtype(categories=sorted(y['zone'].dropna().unique()))
    y_pitch_type = y['pitch_type'].astype(pitch_type_categories).cat.codes.values
    y_strike_zone = y['zone'].astype(zone_categories).cat.codes.values
    y_pitch = y.drop(columns=['pitch_type', 'zone']).values

    y_pitch[np.isnan(y_pitch)] = np.nanmean(y_pitch)

    y_pitch_tensor = torch.tensor(y_pitch, dtype=torch.float32)
    pitch_mean = y_pitch_tensor.mean(dim=0)
    pitch_std = y_pitch_tensor.std(dim=0).clamp(min=1e-8)
    y_pitch_normalized = (y_pitch_tensor - pitch_mean) / pitch_std

    num_zones = len(zone_categories.categories)
    num_pitch_types = len(pitch_type_categories.categories)

    test_X_df = pd.get_dummies(test['game_context'].drop(columns=['game_date', 'game_id']), drop_first=True).astype(float)
    test_X_df = test_X_df.reindex(columns=train_columns, fill_value=0)
    test_X = test_X_df.values
    test_X = (test_X - X_mean) / X_std
    test_y = test['first_pitch']
    test_y['pitch_type'] = test_y['pitch_type'].fillna(test_y['pitch_type'].mode()[0])
    test_y.fillna(test_y.groupby('pitch_type').transform('mean'), inplace=True)
    test_y_pitch_type = test_y['pitch_type'].astype(pitch_type_categories).cat.codes.values
    test_y_strike_zone = test_y['zone'].astype(zone_categories).cat.codes.values
    test_y_pitch = test_y.drop(columns=['pitch_type', 'zone']).values
    test_y_pitch_tensor = torch.tensor(test_y_pitch, dtype=torch.float32)
    test_y_pitch_normalized = (test_y_pitch_tensor - pitch_mean) / pitch_std

    model = first_pitch_predictor(input_dim=X.shape[1],
                                hidden_dim=256,
                                n_pitches=num_pitch_types,
                                n_targets=y_pitch.shape[1],
                                n_zones = num_zones)

    # Print base rates for reference
    from collections import Counter
    pt_counts = Counter(y_pitch_type)
    most_common_pt = pt_counts.most_common(1)[0]
    print(f"Pitch type base rate: {most_common_pt[1]/len(y_pitch_type):.2%} (class {most_common_pt[0]})")
    sz_counts = Counter(y_strike_zone)
    most_common_sz = sz_counts.most_common(1)[0]
    print(f"Strike zone base rate: {most_common_sz[1]/len(y_strike_zone):.2%} (class {most_common_sz[0]})")

    train_first_pitch_model(model, X, y_pitch_type, y_strike_zone, y_pitch_normalized, test_X, test_y_pitch_type, test_y_strike_zone, test_y_pitch_normalized, epochs=50, batch_size=256, lr=1e-4)
    torch.save(model.state_dict(), 'first_pitch_predictor.pth')

    acc = get_val_error(model, test_X, test_y_pitch_type, test_y_strike_zone, test_y_pitch_normalized)
    print(f"Validation Accuracy - Pitch Type: {acc[0]:.4%}, Strike Zone: {acc[1]:.4%}")
    print(f"Validation Loss - Pitch Type: {acc[2]:.4f}, Strike Zone: {acc[3]:.4f}, Cont NLL: {acc[4]:.4f}")