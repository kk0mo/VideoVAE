import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import glob
import os
from collections import Counter
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

def process_csv_data(normalize=True):
    # Read data and check dimensions
    csv_files = sorted(glob.glob(os.path.expanduser('~/Documents/video_data_process/DanceGMR/5s/unitree_g1/*/csv/*.csv')))
    print(f"Found {len(csv_files)} CSV files")
    shapes = []
    all_data = []
    original_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=None)
        all_data.append(df.values)
        shapes.append(df.shape)
    
    num_frames_per_seq, num_features = shapes[0]

    # Check if all CSV files have the same dimensions
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        print(f"\nWARNING: CSV files have different shapes: {unique_shapes}")
        shape_counts = Counter(shapes)
        most_common_shape, count = shape_counts.most_common(1)[0]
        print(f"Using most common shape: {most_common_shape} (found in {count} files)")
        num_frames_per_seq, num_features = most_common_shape
        filtered_data = []
        filtered_files = []
        excluded_files = []
        for csv_file, shape, data in zip(csv_files, shapes, all_data):
            if shape == most_common_shape:
                filtered_data.append(data)
                filtered_files.append(csv_file)
            else:
                excluded_files.append(csv_file)
        all_data = filtered_data
        csv_files = filtered_files

        print(f"\nExcluded {len(excluded_files)} files with inconsistent shapes:")
    
    # Normalize each sequence individually
    normalized_data = []
    stats = []
    
    if normalize:
        print(f"\nNormalizing each sequence individually:")
        for i, seq in enumerate(all_data):
            original_data.append(seq.copy())
            seq_mean = seq.mean()
            seq_std = seq.std()
            seq_norm = (seq - seq_mean) / seq_std
            normalized_data.append(seq_norm)
            stats.append({'mean': seq_mean, 'std': seq_std})
    else:
        normalized_data = all_data
        original_data = all_data
        stats = [{'mean': 0.0, 'std': 1.0} for _ in all_data]
    
    return normalized_data, original_data, csv_files, stats, num_frames_per_seq, num_features


class LSTMVAE(nn.Module):
    def __init__(self, num_features, num_frames, latent_dim=32, hidden_dim=128):
        super().__init__()
        
        self.num_features = num_features
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0
        )
        
        self.fc_out = nn.Linear(hidden_dim, num_features)
    
    def encode(self, x):
        _, (h_n, c_n) = self.encoder_lstm(x)
        h = h_n[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        batch_size = z.shape[0]
        h = self.fc_decode(z)
        h = h.unsqueeze(1).repeat(1, self.num_frames, 1)
        lstm_out, _ = self.decoder_lstm(h)
        output = self.fc_out(lstm_out)
        return output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class KLAnnealer:
    def __init__(self, total_epochs, n_cycles=4):
        self.total_epochs = total_epochs
        self.n_cycles = n_cycles
        self.cycle_length = total_epochs // n_cycles
    
    def get_beta(self, epoch):
        cycle_progress = (epoch % self.cycle_length) / self.cycle_length
        return min(cycle_progress, 1.0)

def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

def compress_sequence(model, sequence, mean, std):
    """Compress entire sequence to single latent vector"""
    model.eval()
    with torch.no_grad():
        seq_norm = (sequence - mean) / std
        x = torch.FloatTensor(seq_norm).unsqueeze(0).to(device)  # ✅ Move to device
        mu, _ = model.encode(x)
        return mu.cpu().squeeze(0).numpy()  # ✅ Move back to CPU for numpy

def decompress_sequence(model, latent, mean, std, shape):
    """Decompress latent vector back to sequence"""
    model.eval()
    with torch.no_grad():
        z = torch.FloatTensor(latent).unsqueeze(0).to(device)  # ✅ Move to device
        recon = model.decode(z)
        recon = recon.cpu().squeeze(0).numpy()  # ✅ Move back to CPU for numpy
        return recon * std + mean


def augment_data(normalized_data, augmentation_factor=5, noise_levels=[0.05, 0.1, 0.15]):
    """Augment training data by adding noise"""
    augmented = []
    for seq in normalized_data:
        augmented.append(seq)
        for _ in range(augmentation_factor - 1):
            noise_std = np.random.choice(noise_levels)
            noise = np.random.normal(0, noise_std, seq.shape)
            augmented_seq = seq + noise
            augmented.append(augmented_seq)
    return augmented


def evaluate_model(model_path, csv_files, original_data, stats, save_compressed=False, output_folder=None):
    """
    Evaluate a trained model on original data
    
    Args:
        model_path: Path to the model checkpoint file
        csv_files: List of CSV file paths
        original_data: List of original sequences
        stats: List of normalization statistics
        save_compressed: Whether to save compressed data (default: False)
        output_folder: Optional output folder name (auto-generated if None)
    
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*70)
    print(f"Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    # ✅ Load checkpoint with map_location for CPU/GPU compatibility
    checkpoint = torch.load(model_path, map_location=device)
    
    # Print model info
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.4f}")
    if 'timestamp' in checkpoint:
        print(f"  Timestamp: {checkpoint['timestamp']}")
    
    # Initialize model
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_frames = checkpoint['num_frames']
    num_features = checkpoint['num_features']
    
    model = LSTMVAE(num_features, num_frames, latent_dim, hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # ✅ Move model to device
    model.eval()
    
    print(f"\nModel architecture:")
    print(f"  Input: {num_frames} frames × {num_features} features")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    
    # Evaluate on each sequence
    print("\n" + "="*70)
    print("Testing compression on each sequence:")
    print("="*70)
    
    total_original_size = 0
    total_compressed_size = 0
    all_errors = []
    compressed_data_list = []
    
    for i, (csv_file, original_seq) in enumerate(zip(csv_files, original_data)):
        seq_mean = stats[i]['mean']
        seq_std = stats[i]['std']
        
        # Compress
        compressed = compress_sequence(model, original_seq, seq_mean, seq_std)
        
        # Decompress
        reconstructed = decompress_sequence(model, compressed, seq_mean, seq_std, original_seq.shape)
        
        # Calculate metrics
        original_size = original_seq.nbytes
        compressed_size = compressed.nbytes + 16  # +16 bytes for mean/std
        ratio = original_size / compressed_size
        mae_error = np.mean(np.abs(reconstructed - original_seq))
        mse_error = np.mean((reconstructed - original_seq) ** 2)
        rmse_error = np.sqrt(mse_error)
        max_error = np.max(np.abs(reconstructed - original_seq))
        
        total_original_size += original_size
        total_compressed_size += compressed_size
        
        error_dict = {
            'file': os.path.basename(csv_file),
            'mae': mae_error,
            'mse': mse_error,
            'rmse': rmse_error,
            'max': max_error,
            'ratio': ratio
        }
        all_errors.append(error_dict)
        
        # Store compressed data if needed
        if save_compressed:
            compressed_data_list.append({
                'filename': os.path.basename(csv_file),
                'latent': compressed,
                'mean': seq_mean,
                'std': seq_std,
                'shape': original_seq.shape
            })
        
        # Print only if error is high
        if mse_error > 0.3:
            print(f"\n⚠️  HIGH ERROR - {os.path.basename(csv_file)}:")
            print(f"  MSE: {mse_error:.6f}, RMSE: {rmse_error:.6f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics:")
    print("="*70)
    
    overall_ratio = total_original_size / total_compressed_size
    avg_mae = np.mean([e['mae'] for e in all_errors])
    avg_mse = np.mean([e['mse'] for e in all_errors])
    avg_rmse = np.mean([e['rmse'] for e in all_errors])
    avg_max = np.mean([e['max'] for e in all_errors])
    
    print(f"Overall compression ratio: {overall_ratio:.1f}:1")
    print(f"Total original size: {total_original_size:,} bytes")
    print(f"Total compressed size: {total_compressed_size:,} bytes")
    print(f"\nAverage errors across all sequences:")
    print(f"  MAE: {avg_mae:.6f}")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  RMSE: {avg_rmse:.6f}")
    print(f"  Max: {avg_max:.6f}")
    
    # Save compressed data if requested
    if save_compressed:
        if output_folder is None:
            eval_timestamp = checkpoint.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            output_folder = f'compressed_data_{eval_timestamp}'
        os.makedirs(output_folder, exist_ok=True)
        
        for comp_data in compressed_data_list:
            output_name = os.path.splitext(comp_data['filename'])[0]
            np.savez(f'{output_folder}/{output_name}.npz',
                     latent=comp_data['latent'],
                     mean=comp_data['mean'],
                     std=comp_data['std'],
                     shape=comp_data['shape'],
                     model_path=model_path)
        print(f"\n✅ Compressed data saved to {output_folder}/")
    else:
        print(f"\n⏭️  Compressed data not saved (set save_compressed=True to save)")
    
    print("="*70)
    
    return {
        'errors': all_errors,
        'avg_mae': avg_mae,
        'avg_mse': avg_mse,
        'avg_rmse': avg_rmse,
        'avg_max': avg_max,
        'compression_ratio': overall_ratio
    }


# Main execution
if __name__ == "__main__":
    # Create run folder with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f'logs/run_{run_timestamp}'
    os.makedirs(run_folder, exist_ok=True)
    print(f"Created run folder: {run_folder}")
    
    # Load and process data
    normalized_data, original_data, csv_files, stats, num_frames, num_features = process_csv_data(normalize=True)
    
    print("\n" + "="*70)
    # augmented_data = augment_data(normalized_data, augmentation_factor=10)
    augmented_data = normalized_data
    print(f"Training data: {len(augmented_data)} sequences")
    
    # Model parameters
    latent_dim = 64
    hidden_dim = 1024
    
    print(f"\nModel architecture:")
    print(f"  Input: {num_frames} frames × {num_features} features")
    print(f"  Hidden LSTM dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    
    compression_ratio = (num_frames * num_features * 8) / (latent_dim * 4)
    print(f"  Compression ratio: ~{compression_ratio:.1f}:1")
    
    # Initialize model
    model = LSTMVAE(num_features, num_frames, latent_dim, hidden_dim)
    model.to(device)  # ✅ Move model to device
    
    # Prepare training data
    train_data = torch.FloatTensor(np.array(augmented_data)).to(device)  # ✅ Move data to device
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Training on: {device}")
    
    # DataLoader
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs_num = 1500
    annealer = KLAnnealer(total_epochs=epochs_num, n_cycles=6)
    
    print(f"\nTraining for {epochs_num} epochs with batch size {batch_size}...")
    print("="*70)
    
    best_loss = float('inf')
    best_model_path = os.path.join(run_folder, 'best_model.pth')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs_num):
        model.train()
        beta = annealer.get_beta(epoch)
        beta = 0.001
        # beta = max(beta * 0.1, 0.005)
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in dataloader:
            # ✅ No need to move batch to device - already on device from DataLoader
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = loss_function(recon, batch, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss
            total_kl += kl_loss
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        if epoch % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, β={beta:.3f}, LR={current_lr:.2e}")
        
        # Save best model (overwrite previous best)
        if avg_recon < best_loss - 0.001:
            best_loss = avg_recon
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_dim': latent_dim,
                'hidden_dim': hidden_dim,
                'num_frames': num_frames,
                'num_features': num_features,
                'stats': stats,
                'epoch': epoch,
                'loss': avg_recon,
                'timestamp': run_timestamp,
                'device': str(device)  # ✅ Save device info
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter > 500:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print("="*70)
    print("Training complete!\n")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best reconstruction loss: {best_loss:.4f}")
    
    # Ask user if they want to save compressed data
    print("\n" + "="*70)
    save_choice = input("Save compressed data? (y/n) [default: n]: ").strip().lower()
    save_compressed = save_choice == 'y'
    
    # Evaluate best model
    if os.path.exists(best_model_path):
        print("\n" + "="*70)
        print("Evaluating BEST model:")
        compressed_folder = os.path.join(run_folder, 'compressed_data') if save_compressed else None
        results = evaluate_model(best_model_path, csv_files, original_data, stats, 
                                save_compressed=save_compressed, output_folder=compressed_folder)
        
        # Save evaluation results
        if results:
            results_file = os.path.join(run_folder, 'evaluation_results.txt')
            with open(results_file, 'w') as f:
                f.write(f"Evaluation Results - {run_timestamp}\n")
                f.write(f"Device: {device}\n")
                f.write("="*70 + "\n")
                f.write(f"Average MAE: {results['avg_mae']:.6f}\n")
                f.write(f"Average MSE: {results['avg_mse']:.6f}\n")
                f.write(f"Average RMSE: {results['avg_rmse']:.6f}\n")
                f.write(f"Average Max Error: {results['avg_max']:.6f}\n")
                f.write(f"Compression Ratio: {results['compression_ratio']:.1f}:1\n")
                f.write("\n" + "="*70 + "\n")
                f.write("Per-file errors:\n")
                for err in results['errors']:
                    f.write(f"\n{err['file']}:\n")
                    f.write(f"  MAE: {err['mae']:.6f}\n")
                    f.write(f"  MSE: {err['mse']:.6f}\n")
                    f.write(f"  RMSE: {err['rmse']:.6f}\n")
                    f.write(f"  Max: {err['max']:.6f}\n")
                    f.write(f"  Ratio: {err['ratio']:.1f}:1\n")
            print(f"\n✅ Evaluation results saved to: {results_file}")
    
    print(f"\n✅ All outputs saved to: {run_folder}/")
