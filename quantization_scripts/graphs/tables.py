import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import csv

def read_json_files(base_path):
    # Initialize matrices to store PSNR and SSIM values
    bits = [2, 4, 6, 8, 10, 12, 14, 16]
    psnr_matrix = np.zeros((len(bits), len(bits)))
    ssim_matrix = np.zeros((len(bits), len(bits)))
    
    # Create mapping of bits to matrix indices
    bits_to_idx = {b: i for i, b in enumerate(bits)}
    
    # Find all combination folders
    pattern = os.path.join(base_path, "combination-log-e*b-d*b/info.json")
    json_files = glob(pattern)

    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Get encoder and decoder bits
            encoder_bits = data['encoder_bits']
            decoder_bits = data['decoder_bits']
            
            # Get corresponding matrix indices
            if encoder_bits in bits_to_idx and decoder_bits in bits_to_idx:
                i = bits_to_idx[encoder_bits]
                j = bits_to_idx[decoder_bits]
                
                # Store values in matrices
                psnr_matrix[i, j] = data['psnr']
                ssim_matrix[i, j] = data['ssim']
    
    return psnr_matrix, ssim_matrix, bits

def read_full_quantization_data(base_path):
    # Find all full quantization folders
    pattern = os.path.join(base_path, "full-*b/info.json")
    json_files = glob(pattern)
    
    bits = []
    psnr_values = []
    ssim_values = []
    model_sizes = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract bit value from folder name
            folder_name = os.path.basename(os.path.dirname(json_file))
            bit_value = int(folder_name.split('-')[1].replace('b', ''))
            
            bits.append(bit_value)
            psnr_values.append(data['psnr'])
            ssim_values.append(data['ssim'])
            model_sizes.append(data['quantized_model_size'])
    
    # Sort all lists based on bits
    sorted_indices = np.argsort(bits)
    bits = np.array(bits)[sorted_indices]
    psnr_values = np.array(psnr_values)[sorted_indices]
    ssim_values = np.array(ssim_values)[sorted_indices]
    model_sizes = np.array(model_sizes)[sorted_indices]
    
    return bits, psnr_values, ssim_values, model_sizes

def read_partial_quantization_data(base_path, prefix):
    # Find all encoder/decoder quantization folders
    pattern = os.path.join(base_path, f"{prefix}-*b/info.json")
    json_files = glob(pattern)
    
    bits = []
    psnr_values = []
    ssim_values = []
    model_sizes = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract bit value from folder name
            folder_name = os.path.basename(os.path.dirname(json_file))
            bit_value = int(folder_name.split('-')[1].replace('b', ''))
            
            bits.append(bit_value)
            psnr_values.append(data['psnr'])
            ssim_values.append(data['ssim'])
            
            # Use quantized_model_size for both encoder and decoder
            model_sizes.append(data['quantized_model_size'])
    
    # Sort all lists based on bits
    sorted_indices = np.argsort(bits)
    bits = np.array(bits)[sorted_indices]
    psnr_values = np.array(psnr_values)[sorted_indices]
    ssim_values = np.array(ssim_values)[sorted_indices]
    model_sizes = np.array(model_sizes)[sorted_indices]
    
    return bits, psnr_values, ssim_values, model_sizes

def plot_confusion_matrix(matrix, bits, metric_name, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.3f', xticklabels=bits, yticklabels=bits)
    plt.xlabel('Decoder Bits')
    plt.ylabel('Encoder Bits')
    plt.title(f'{metric_name} Values for Different Quantization Combinations')
    plt.savefig(save_path)
    plt.close()

def plot_metric_vs_bits(bits_full, values_full, bits_enc, values_enc, 
                       bits_dec, values_dec, metric_name, save_path):
    plt.figure(figsize=(12, 7))
    
    plt.plot(bits_full, values_full, 'bo-', linewidth=2, markersize=8, label='Full Model')
    plt.plot(bits_enc, values_enc, 'ro-', linewidth=2, markersize=8, label='Encoder Only')
    plt.plot(bits_dec, values_dec, 'go-', linewidth=2, markersize=8, label='Decoder Only')
    
    plt.grid(True)
    plt.xlabel('Quantization Bits')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Quantization Bits')
    plt.legend()
    
    # Set x-ticks to include all bit values from all three curves
    all_bits = np.unique(np.concatenate([bits_full, bits_enc, bits_dec]))
    plt.xticks(all_bits)
    
    plt.savefig(save_path)
    plt.close()

def plot_model_size(bits_full, sizes_full, bits_enc, sizes_enc, 
                    bits_dec, sizes_dec, save_path):
    plt.figure(figsize=(12, 7))
    
    plt.plot(bits_full, sizes_full, 'bo-', linewidth=2, markersize=8, label='Full Model')
    plt.plot(bits_enc, sizes_enc, 'ro-', linewidth=2, markersize=8, label='Encoder Only')
    plt.plot(bits_dec, sizes_dec, 'go-', linewidth=2, markersize=8, label='Decoder Only')
    
    plt.grid(True)
    plt.xlabel('Quantization Bits')
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size vs Quantization Bits')
    plt.legend()
    
    # Set x-ticks to include all bit values from all three curves
    all_bits = np.unique(np.concatenate([bits_full, bits_enc, bits_dec]))
    plt.xticks(all_bits)
    
    plt.savefig(save_path)
    plt.close()

def collect_metrics():
    base_paths = [
        "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantised-logarithmic",
        "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantised-tensorparallel"
    ]
    
    results = []
    
    for base_path in base_paths:
        # Find all combination folders using glob
        pattern = os.path.join(base_path, "combination*", "info.json")
        json_files = glob(pattern)
        
        for json_file in json_files:
            folder_name = os.path.basename(os.path.dirname(json_file))
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'method': folder_name,
                    'quantization_type': 'logarithmic' if 'log' in folder_name else 'tensorparallel',
                    'psnr': data['psnr'],
                    'ssim': data['ssim'],
                    'encoder_bits': data['encoder_bits'],
                    'decoder_bits': data['decoder_bits'],
                    'original_model_size_MB': data['original_model_size_MB'],
                    'quantized_model_size': data['quantized_model_size'],
                    'compression_ratio': data['original_model_size_MB'] / data['quantized_model_size']
                })
    
    # Write to CSV
    output_path = "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantization_scripts/metrics_comparison.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def main():
    base_path = "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantised"
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Process combination matrices
    psnr_matrix, ssim_matrix, bits = read_json_files(base_path)
    
    # Plot and save confusion matrices
    plot_confusion_matrix(psnr_matrix, bits, 'PSNR', 'plots/psnr_confusion_matrix.png')
    plot_confusion_matrix(ssim_matrix, bits, 'SSIM', 'plots/ssim_confusion_matrix.png')
    
    # Generate metrics CSV
    collect_metrics()

if __name__ == "__main__":
    main()
