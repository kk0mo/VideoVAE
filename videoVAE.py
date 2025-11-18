import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import glob
import os
import argparse
from collections import Counter
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

def process_csv_data(data_dir='DanceCSV', time_window='5s', robot_model='unitree_g1', normalize=True):
    # Read data and check dimensions
    csv_pattern = os.path.expanduser(f'{data_dir}/{time_window}/{robot_model}/*/*.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    print(f"Data path: {csv_pattern}")
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
        # print(f"\nNormalizing each sequence individually:")
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

def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

def compress_sequence(model, sequence, mean, std):
    """Compress entire sequence to single latent vector"""
    model.eval()
    with torch.no_grad():
        seq_norm = (sequence - mean) / std
        x = torch.FloatTensor(seq_norm).unsqueeze(0).to(device) 
        mu, _ = model.encode(x)
        return mu.cpu().squeeze(0).numpy()

def decompress_sequence(model, latent, mean, std, shape):
    """Decompress latent vector back to sequence"""
    model.eval()
    with torch.no_grad():
        z = torch.FloatTensor(latent).unsqueeze(0).to(device)
        recon = model.decode(z)
        recon = recon.cpu().squeeze(0).numpy()
        return recon * std + mean

def split_data(normalized_data, original_data, csv_files, stats, test_ratio=0.2):
    """
    Split data into training and test sets
    """
    n_samples = len(normalized_data)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test
    
    # Create indices and shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_data = {
        'normalized': [normalized_data[i] for i in train_indices],
        'original': [original_data[i] for i in train_indices],
        'csv_files': [csv_files[i] for i in train_indices],
        'stats': [stats[i] for i in train_indices]
    }
    
    test_data = {
        'normalized': [normalized_data[i] for i in test_indices],
        'original': [original_data[i] for i in test_indices],
        'csv_files': [csv_files[i] for i in test_indices],
        'stats': [stats[i] for i in test_indices]
    }
    
    print(f"\nData split:")
    print(f"  Training samples: {n_train} ({100*(1-test_ratio):.0f}%)")
    print(f"  Test samples: {n_test} ({100*test_ratio:.0f}%)")
    
    return train_data, test_data


def train_model(train_data, num_frames, num_features, latent_dim=64, hidden_dim=1024, 
                epochs_num=1500, batch_size=32, run_folder='logs/run'):
    augmented_data = train_data['normalized']
    
    print(f"\nModel architecture:")
    print(f"  Input: {num_frames} frames × {num_features} features")
    print(f"  Hidden LSTM dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    
    compression_ratio = (num_frames * num_features * 8) / (latent_dim * 4)
    print(f"  Compression ratio: ~{compression_ratio:.1f}:1")
    
    model = LSTMVAE(num_features, num_frames, latent_dim, hidden_dim)
    model.to(device)
    
    train_tensor = torch.FloatTensor(np.array(augmented_data)).to(device)
    print(f"\nTraining data shape: {train_tensor.shape}")
    print(f"Training on: {device}")
    
    dataloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\nTraining for {epochs_num} epochs with batch size {batch_size}...")
    print("="*70)
    
    best_loss = float('inf')
    best_model_path = os.path.join(run_folder, 'best_model.pth')
    patience_counter = 0
    run_timestamp = os.path.basename(run_folder).replace('run_', '')
    
    # Training loop
    for epoch in range(epochs_num):
        model.train()
        beta = 0.001
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in dataloader:
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
        
        # Save model every 200 epochs
        if epoch % 200 == 0 and epoch > 0:
            checkpoint_path = os.path.join(run_folder, f'checkpoint_epoch_{epoch:04d}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_dim': latent_dim,
                'hidden_dim': hidden_dim,
                'num_frames': num_frames,
                'num_features': num_features,
                'stats': train_data['stats'],
                'epoch': epoch,
                'loss': avg_recon,
                'timestamp': run_timestamp,
                'device': str(device)
            }, checkpoint_path)
            print(f"Checkpoint saved: epoch_{epoch:04d}.pth")
        
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
                'stats': train_data['stats'],
                'epoch': epoch,
                'loss': avg_recon,
                'timestamp': run_timestamp,
                'device': str(device)
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
    
    return best_model_path, best_loss


def evaluate_model(model_path, test_data, save_compressed=False, output_folder=None):
    csv_files = test_data['csv_files']
    original_data = test_data['original']

    stats = test_data['stats']
    
    # Print test_data info
    #print("\n" + "="*70)
    print("Test Data Info:")
    print(f"  Number of sequences: {len(original_data)}")
    print(f"Loading model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
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
    model.to(device)
    model.eval()
    
    print(f"\nModel architecture:")
    print(f"  Input: {num_frames} frames × {num_features} features")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    
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
    # print(f"Total original size: {total_original_size:,} bytes")
    # print(f"Total compressed size: {total_compressed_size:,} bytes")
    print(f"\nAverage errors across all sequences:")
    # print(f"  MAE: {avg_mae:.6f}")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  RMSE: {avg_rmse:.6f}")
    # print(f"  Max: {avg_max:.6f}")
    
    # Save compressed data if requested
    if save_compressed:
        if output_folder is None:
            eval_timestamp = checkpoint.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
            output_folder = f'compressed_data_{eval_timestamp}'
        os.makedirs(output_folder, exist_ok=True)
        
        for comp_data in compressed_data_list:
            # Extract dance type from filename or path
            # Assuming format: path/to/dance_type/filename.csv
            filename = comp_data['filename']
            # Try to extract dance type from CSV path if available
            dance_type = 'unknown'
            for csv_file in csv_files:
                if os.path.basename(csv_file) == filename or csv_file.endswith(filename):
                    # Extract the parent directory name as dance type
                    dance_type = os.path.basename(os.path.dirname(csv_file))
                    break
            
            # Create dance type folder if it doesn't exist
            dance_folder = os.path.join(output_folder, dance_type)
            os.makedirs(dance_folder, exist_ok=True)
            
            output_name = os.path.splitext(filename)[0]
            output_path = os.path.join(dance_folder, f'{output_name}.npz')
            np.savez(output_path,
                     latent=comp_data['latent'],
                     mean=comp_data['mean'],
                     std=comp_data['std'],
                     shape=comp_data['shape'],
                     model_path=model_path)
        print(f"\n Compressed data saved to {output_folder}/")
    else:
        print(f"\n Compressed data not saved (set save_compressed=True to save)")
    
    print("="*70)
    
    return {
        'errors': all_errors,
        'avg_mae': avg_mae,
        'avg_mse': avg_mse,
        'avg_rmse': avg_rmse,
        'avg_max': avg_max,
        'compression_ratio': overall_ratio
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate VideoVAE model')
    parser.add_argument('--mode', choices=['train', 'compress'], default='train',
                        help='Mode: train (train model), compress (compress data with existing model)')
    parser.add_argument('--data_dir', type=str, default='DanceCSV',
                        help='Data directory (default: DanceCSV)')
    parser.add_argument('--time_window', type=str, default='5s',
                        help='Time window folder (default: 5s)')
    parser.add_argument('--robot_model', type=str, default='unitree_g1',
                        help='Robot model folder (default: unitree_g1)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint for compression mode')
    parser.add_argument('--save_compressed', action='store_true',
                        help='Save compressed data during training evaluation')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of data to use for testing (default: 0.2)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='Hidden LSTM dimension (default: 1024)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of epochs (default: 1500)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder for compressed data')
    
    args = parser.parse_args()
    
    # Load data
    normalized_data, original_data, csv_files, stats, num_frames, num_features = process_csv_data(
        data_dir=args.data_dir,
        time_window=args.time_window,
        robot_model=args.robot_model,
        normalize=True
    )
    
    if args.mode == 'train':
        # Training mode
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f'logs/run_{run_timestamp}'
        os.makedirs(run_folder, exist_ok=True)
        print(f"Created run folder: {run_folder}")
        
        print("\n" + "="*70)
        print(f"Total data: {len(normalized_data)} sequences")
        
        train_data, test_data = split_data(normalized_data, original_data, csv_files, stats, test_ratio=args.test_ratio)
        
        # Model parameters
        latent_dim = args.latent_dim
        hidden_dim = args.hidden_dim
        epochs_num = args.epochs
        batch_size = args.batch_size
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING PHASE")
        print("="*70)
        best_model_path, best_loss = train_model(
            train_data=train_data,
            num_frames=num_frames,
            num_features=num_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            epochs_num=epochs_num,
            batch_size=batch_size,
            run_folder=run_folder
        )
        
        # Evaluate on test data
        print("\n" + "="*70)
        print("EVALUATION PHASE")
        print("="*70)
        if os.path.exists(best_model_path):
            compressed_folder = os.path.join(run_folder, 'compressed_data') if args.save_compressed else None
            results = evaluate_model(best_model_path, test_data, 
                                    save_compressed=args.save_compressed, output_folder=compressed_folder)
            
            # Save evaluation results
            if results:
                results_file = os.path.join(run_folder, 'evaluation_results.txt')
                with open(results_file, 'w') as f:
                    f.write(f"Evaluation Results - {run_timestamp}\n")
                    f.write(f"Device: {device}\n")
                    f.write(f"Test samples: {len(test_data['csv_files'])}\n")
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
    
    elif args.mode == 'compress':
        # Compression mode - compress data with existing model
        if args.model is None:
            print("ERROR: --model argument required for compress mode")
            print("Usage: python videoVAE.py --mode compress --model <path_to_model>")
            exit(1)
        
        if not os.path.exists(args.model):
            print(f"ERROR: Model file not found: {args.model}")
            exit(1)
        
        print("\n" + "="*70)
        print("COMPRESSION MODE")
        print("="*70)
        
        # Create output folder
        if args.output_folder is None:
            args.output_folder = os.path.join(os.path.dirname(args.model), 'compressed_data')
        
        # Create test_data dict from all data
        test_data = {
            'normalized': normalized_data,
            'original': original_data,
            'csv_files': csv_files,
            'stats': stats
        }
        
        # Evaluate and compress
        results = evaluate_model(args.model, test_data, 
                                save_compressed=True, output_folder=args.output_folder)
        
        if results:
            # Save compression summary
            summary_file = os.path.join(args.output_folder, 'compression_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Compression Summary\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                f.write("="*70 + "\n")
                f.write(f"Total sequences: {len(original_data)}\n")
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
            print(f"✅ Compression summary saved to: {summary_file}")
        
        print(f"✅ Compressed data saved to: {args.output_folder}/")
