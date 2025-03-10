import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

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

def main():
    base_path = "/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/quantised"
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Process combination matrices
    psnr_matrix, ssim_matrix, bits = read_json_files(base_path)
    
    # Plot and save confusion matrices
    plot_confusion_matrix(psnr_matrix, bits, 'PSNR', 'plots/psnr_confusion_matrix.png')
    plot_confusion_matrix(ssim_matrix, bits, 'SSIM', 'plots/ssim_confusion_matrix.png')
    
    # # Process full quantization data
    # bits_full, psnr_full, ssim_full, size_full = read_full_quantization_data(base_path)
    
    # # Process encoder-only data
    # bits_enc, psnr_enc, ssim_enc, size_enc = read_partial_quantization_data(base_path, "encoder")
    
    # # Process decoder-only data
    # bits_dec, psnr_dec, ssim_dec, size_dec = read_partial_quantization_data(base_path, "decoder")
    
    # # Plot metrics vs bits with all three curves
    # plot_metric_vs_bits(bits_full, psnr_full, bits_enc, psnr_enc, bits_dec, psnr_dec,
    #                    'PSNR', 'plots/full_quantization_psnr.png')
    # plot_metric_vs_bits(bits_full, ssim_full, bits_enc, ssim_enc, bits_dec, ssim_dec,
    #                    'SSIM', 'plots/full_quantization_ssim.png')
    # plot_model_size(bits_full, size_full, bits_enc, size_enc, bits_dec, size_dec,
    #                'plots/model_size_vs_bits.png')
    
    # # Print numerical results
    # print("\nFull Quantization Results:")
    # print("Bits\tPSNR\tSSIM\tModel Size (MB)")
    # print("-" * 40)
    # for b, p, s, m in zip(bits_full, psnr_full, ssim_full, size_full):
    #     print(f"{b}\t{p:.3f}\t{s:.3f}\t{m:.2f}")
        
    # print("\nEncoder-only Quantization Results:")
    # print("Bits\tPSNR\tSSIM\tModel Size (MB)")
    # print("-" * 40)
    # for b, p, s, m in zip(bits_enc, psnr_enc, ssim_enc, size_enc):
    #     print(f"{b}\t{p:.3f}\t{s:.3f}\t{m:.2f}")
        
    # print("\nDecoder-only Quantization Results:")
    # print("Bits\tPSNR\tSSIM\tModel Size (MB)")
    # print("-" * 40)
    # for b, p, s, m in zip(bits_dec, psnr_dec, ssim_dec, size_dec):
    #     print(f"{b}\t{p:.3f}\t{s:.3f}\t{m:.2f}")

if __name__ == "__main__":
    main()
