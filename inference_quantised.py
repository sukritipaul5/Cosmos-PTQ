import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import argparse
from tqdm import tqdm
import time
from torchmetrics.image import PeakSignalNoiseRatio
import lpips
import random
from pathlib import Path
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dahuffman import HuffmanCodec
from torchmetrics.image import StructuralSimilarityIndexMeasure

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    return 0, 1, 0


def quantize_model_static(model, bits=8):
    """Perform simple quantization without bit packing"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    
    total_params = 0
    quantized_params = 0
    quantized_state_dict = {}
    
    if bits == 16:
        total_params = sum(p.numel() for p in model_to_quantize.parameters())
        quantized_model = model_to_quantize.to(dtype=torch.bfloat16)
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        quantized_state_dict = quantized_model.state_dict()
        return quantized_model, total_params, quantized_params, quantized_state_dict

    max_val = 2**(bits-1) - 1
    
    for name, param in model_to_quantize.named_parameters():
        if 'weight' in name:
            weight = param.data
            weight_scale = weight.abs().max() / max_val
            
            # Simple quantization
            quantized_weight = (weight / weight_scale).round().clamp(-max_val, max_val).to(torch.int8)
            
            quantized_state_dict[f'{name}_quantized'] = quantized_weight
            quantized_state_dict[f'{name}_scale'] = weight_scale
            
            # For inference
            param.data = (quantized_weight.float() * weight_scale)
            
            total_params += weight.numel() * weight.element_size()
            quantized_params += weight.numel() * (bits / 8)
        else:
            quantized_state_dict[name] = param.data
    
    model_to_quantize = model_to_quantize.to(device=device, dtype=dtype)
    return model_to_quantize, total_params, quantized_params, quantized_state_dict

def load_quantized_model(model, checkpoint_path):
    """Load and decompress quantized weights"""
    checkpoint = torch.load(checkpoint_path)
    compressed_dict = checkpoint['state_dict']
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if f'{name}_quantized' in compressed_dict:
                # Get the shape, codec and compressed data
                shape = compressed_dict[f'{name}_quantized_shape']
                codec = compressed_dict[f'{name}_quantized_codec']
                encoded_data = compressed_dict[f'{name}_quantized']
                
                # Decode the data
                decoded_data = codec.decode(encoded_data)
                
                # Reshape back to tensor
                quantized_weight = torch.tensor(decoded_data).reshape(shape)
                
                # Get scale and dequantize
                weight_scale = compressed_dict[f'{name}_scale']
                param.data = (quantized_weight.float() * weight_scale)
            elif name in compressed_dict:
                param.data = compressed_dict[name]
    
    return model

def save_quantized_model(model_type, model_name, quantized_state_dict, save_path, bits):
    """Save quantized model with Huffman coding compression"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_name = os.path.splitext(save_path)[0]
    quantized_path = f"{base_name}_{bits}bit.pt"
    
    # Create a new state dict for compressed data
    compressed_dict = {}
    
    for name, tensor in quantized_state_dict.items():
        if '_quantized' in name:
            # Store shape for reconstruction
            compressed_dict[f"{name}_shape"] = tensor.shape
            # Flatten and convert to list for Huffman coding
            values = tensor.cpu().numpy().ravel().tolist()
            # Create and train Huffman codec
            codec = HuffmanCodec.from_data(values)
            # Encode the data
            encoded_data = codec.encode(values)
            # Store encoded data and codec
            compressed_dict[name] = encoded_data
            compressed_dict[f"{name}_codec"] = codec
        else:
            # Store other tensors (like scales) as is
            compressed_dict[name] = tensor
    
    # Save with metadata
    save_dict = {
        'model_type': model_type,
        'model_name': model_name,
        'bits': bits,
        'state_dict': compressed_dict
    }
    
    torch.save(save_dict, quantized_path)
    print(f"Saved compressed quantized model to: {quantized_path}")
    
    return quantized_path



class QuantizedImageTokenizer:
    def __init__(self, args, checkpoint_enc=None, checkpoint_dec=None, bits=8, save_dir=None):
        self.bits = bits
        self.encoder = None
        self.decoder = None
        self.save_dir = save_dir
        self.args = args
        
        if checkpoint_enc:
            print("\nProcessing encoder...")
            self.encoder = torch.jit.load(checkpoint_enc)
            print(f"Loaded encoder with {sum(p.numel() for p in self.encoder.parameters())} parameters")
            if not self.args.no_quantize:
                self.encoder, enc_total, enc_quant, enc_state_dict = quantize_model_static(self.encoder, bits)
            else:
                enc_state_dict = self.encoder.state_dict()
                enc_total = 1
                enc_quant = 1
            
            if self.save_dir:
                save_name = f"encoder_quantized_{bits}bit.pt"
                save_path = os.path.join(self.save_dir, save_name)
            else:
                save_path = checkpoint_enc.replace('.jit', f'_quantized_{bits}bit.pt')
            
            quantized_encoder_path = save_quantized_model('encoder', 'Cosmos-Tokenizer', enc_state_dict, save_path, bits)
            
        if checkpoint_dec:
            print("\nProcessing decoder...")
            self.decoder = torch.jit.load(checkpoint_dec)
            print(f"Loaded decoder with {sum(p.numel() for p in self.decoder.parameters())} parameters")
            if not self.args.no_quantize:
                self.decoder, dec_total, dec_quant, dec_state_dict = quantize_model_static(self.decoder, bits)
            else:
                dec_state_dict = self.decoder.state_dict()
                dec_total = 1
                dec_quant = 1
            
            if self.save_dir:
                save_name = f"decoder_quantized_{bits}bit.pt"
                save_path = os.path.join(self.save_dir, save_name)
            else:
                save_path = checkpoint_dec.replace('.jit', f'_quantized_{bits}bit.pt')
            
            quantized_decoder_path = save_quantized_model('decoder', 'Cosmos-Tokenizer', dec_state_dict, save_path, bits)
        
        self.file_size = {
            'encoder': os.path.getsize(quantized_encoder_path) / (1024 * 1024),
            'decoder': os.path.getsize(quantized_decoder_path) / (1024 * 1024)
        }

        self.memory_reduction = {
            'encoder': (enc_total, enc_quant) if checkpoint_enc else None,
            'decoder': (dec_total, dec_quant) if checkpoint_dec else None
        }

    def encode(self, x):
        return self.encoder(x) if self.encoder else None

    def decode(self, x):
        return self.decoder(x) if self.decoder else None

def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        return img.size

def is_valid_image(image_path, min_res):
    try:
        width, height = get_image_resolution(image_path)
        # If min_res is None, accept all valid images
        if min_res is None:
            return True
        return min(width, height) >= min_res
    except:
        return False

def load_and_preprocess_image(image_path, target_size=None):
    image = Image.open(image_path)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_for_saving = image.copy()
    
    if target_size:
        transform = transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()
    
    if target_size:
        transform_uint8 = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
        ])
    else:
        transform_uint8 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
        ])
    
    img_tensor = transform(image)
    img_tensor_uint8 = transform_uint8(image)
    
    return (img_tensor.unsqueeze(0), 
            img_tensor_uint8.unsqueeze(0), 
            original_for_saving)

# def save_comparison(original_image, reconstructed_tensor, output_path, idx):
#     orig_w, orig_h = original_image.size
    
#     original_save_path = os.path.join(output_path, 'original', f'original_{idx:04d}.png')
#     recon_save_path = os.path.join(output_path, 'reconstructed', f'reconstructed_{idx:04d}.png')
    
#     original_image.save(original_save_path, 'PNG')
    
#     reconstructed_image = reconstructed_tensor[0]
    
#     resize_transform = transforms.Resize(
#         (orig_h, orig_w),
#         interpolation=transforms.InterpolationMode.LANCZOS
#     )
#     reconstructed_image = resize_transform(reconstructed_image)
    
#     reconstructed_image = reconstructed_image.cpu().float().numpy()
#     reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
#     reconstructed_image = np.clip(reconstructed_image, 0, 1)
    
#     reconstructed_pil = Image.fromarray((reconstructed_image * 255).astype(np.uint8))
#     reconstructed_pil.save(recon_save_path, 'PNG')


def save_comparison(original_image, reconstructed_tensor, output_path, idx):
    """Save images using PIL's LANCZOS resampling"""
    orig_w, orig_h = original_image.size
    
    original_save_path = os.path.join(output_path, 'original', f'original_{idx:04d}.png')
    recon_save_path = os.path.join(output_path, 'reconstructed', f'reconstructed_{idx:04d}.png')
    
    # Save original
    original_image.save(original_save_path, 'PNG')
    
    # Process reconstructed image
    reconstructed_image = reconstructed_tensor[0]  # Remove batch dimension
    
    # Convert to CPU, float, and clamp values
    reconstructed_image = reconstructed_image.cpu().float().clamp(0, 1)
    # Convert to PIL
    reconstructed_image = transforms.ToPILImage()(reconstructed_image)
    # Use PIL's LANCZOS resampling
    reconstructed_image = reconstructed_image.resize((orig_w, orig_h), Image.LANCZOS)
    reconstructed_image.save(recon_save_path, 'PNG')
    
class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_values = []
        
    def update(self, original_float, original_uint8, reconstructed_float, reconstructed_uint8):
        # Remove batch dimension if batch size is 1
        if original_float.dim() == 4 and original_float.size(0) == 1:
            original_float = original_float.squeeze(0)
            reconstructed_float = reconstructed_float.squeeze(0)
        
        # Add batch dimension back for SSIM
        original_batched = original_float.unsqueeze(0)
        reconstructed_batched = reconstructed_float.unsqueeze(0)
        
        self.psnr.update(reconstructed_float, original_float)        
        self.ssim.update(reconstructed_batched, original_batched)
        
        lpips_value = self.lpips_fn(original_float * 2 - 1, reconstructed_float * 2 - 1)
        self.lpips_values.append(lpips_value.item())
        
    def compute(self):        
        metrics = {
            'psnr': self.psnr.compute().item(),
            'ssim': self.ssim.compute().item(),
            'lpips': np.mean(self.lpips_values),
        }
            
        return metrics

def process_directory(args, rank, world_size):
    device = f'cuda:{rank}'
    
    if rank == 0:
        os.makedirs(os.path.join(args.output_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'reconstructed'), exist_ok=True)
    
    if world_size > 1:
        torch.distributed.barrier()
    
    # Initialize quantized tokenizer
    tokenizer = QuantizedImageTokenizer(
        args=args,
        checkpoint_enc=os.path.join(args.model_dir, args.model_name, "encoder.jit"),
        checkpoint_dec=os.path.join(args.model_dir, args.model_name, "decoder.jit"),
        bits=args.bits
    )
    
    all_images = [f for f in Path(args.input_dir).glob('**/*') 
                 if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    
    # Filter valid images based on resolution requirement (if any)
    valid_images = [img for img in all_images if is_valid_image(img, args.downsample_res)]
    
    # Ensure we have at least some images
    if not valid_images:
        raise ValueError(f"No valid images found in {args.input_dir}. Please check the input directory and resolution requirements.")
    
    # Limit to requested number of samples
    valid_images = valid_images[:1000]
    
    if rank == 0:
        print(f"Found {len(valid_images)} valid images out of {len(all_images)} total images")
    
    rank_images = valid_images[rank::world_size]
    
    metrics_calc = MetricsCalculator(device)
    total_time = 0
    total_samples = 0
    
    if rank == 0:
        print(f"Processing {len(rank_images)} images")

    viz_count = 0
    for idx, image_path in enumerate(tqdm(rank_images, disable=rank != 0)):
        try:
            input_tensor_float, input_tensor_uint8, original_image = load_and_preprocess_image(
                image_path, args.downsample_res)
            
            input_tensor_float = input_tensor_float.to(device).to(torch.bfloat16)
            input_tensor_uint8 = input_tensor_uint8.to(device)
            
            start_time = time.time()
            encoded_output = tokenizer.encode(input_tensor_float)
            indices = encoded_output[0] if isinstance(encoded_output, tuple) else encoded_output
            
            reconstructed_tensor_float = tokenizer.decode(indices)  
            reconstructed_tensor_float = reconstructed_tensor_float.clamp(0, 1)
            
            end_time = time.time()
            total_time += end_time - start_time
            total_samples += 1
            
            reconstructed_tensor_uint8 = (reconstructed_tensor_float.float() * 255).to(torch.uint8)
            
            #make them same shape
            if reconstructed_tensor_float.shape != input_tensor_float.shape:
                # Resize reconstructed tensors to match input shape (NCHW format)
                from torchvision.transforms import Resize
                target_size = (input_tensor_float.shape[2], input_tensor_float.shape[3])
                print(target_size)
                reconstructed_tensor_float = Resize(target_size)(reconstructed_tensor_float)
                reconstructed_tensor_uint8 = Resize(target_size)(reconstructed_tensor_uint8)


            metrics_calc.update(
                input_tensor_float.float(), 
                input_tensor_uint8,
                reconstructed_tensor_float.float(),
                reconstructed_tensor_uint8
            )
            
            if rank == 0 and viz_count < args.num_save_samples:
                save_comparison(
                    original_image, 
                    reconstructed_tensor_float,
                    args.output_dir,
                    viz_count
                )
                viz_count += 1
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    if total_samples == 0:
        raise ValueError("No images were successfully processed. Please check your input images and parameters.")
    
    metrics = metrics_calc.compute()
    throughput = total_samples / total_time if total_time > 0 else 0
    
    if world_size > 1:
        for metric_name, metric_value in metrics.items():
            metric_tensor = torch.tensor([metric_value], device=device)
            dist.all_reduce(metric_tensor)
            metrics[metric_name] = metric_tensor.item() / world_size
        
        throughput_tensor = torch.tensor([throughput], device=device)
        dist.all_reduce(throughput_tensor)
        throughput = throughput_tensor.item() / world_size
    
    # Add memory reduction info to metrics
    if rank == 0:
        memory_metrics = {
            'encoder_reduction': (
                tokenizer.memory_reduction['encoder'][0],
                tokenizer.memory_reduction['encoder'][1]
            ) if tokenizer.memory_reduction['encoder'] else None,
            'decoder_reduction': (
                tokenizer.memory_reduction['decoder'][0],
                tokenizer.memory_reduction['decoder'][1]
            ) if tokenizer.memory_reduction['decoder'] else None
        }
        metrics['memory'] = memory_metrics
        metrics['encoder_file_size_MB'] = tokenizer.file_size['encoder']
        metrics['decoder_file_size_MB'] = tokenizer.file_size['decoder']
        #total file size
        metrics['total_file_size_MB'] = metrics['encoder_file_size_MB'] + metrics['decoder_file_size_MB']


    print(metrics)
    return metrics, throughput

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default="/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/scripts/pretrained_ckpts")
    parser.add_argument('--model_name', type=str, default="Cosmos-Tokenizer-DI8x8")
    parser.add_argument('--downsample_res', type=int, default=None)
    parser.add_argument('--bits', type=int, default=8, help='Number of bits for quantization')
    parser.add_argument('--quantization', type=str, choices=['static', 'dynamic', 'none'], default='none')
    parser.add_argument('--num_save_samples', type=int, default=30, help='Number of sample images to save')
    parser.add_argument('--no_quantize', action='store_true', help='Do not quantize the model')
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    metrics, throughput = process_directory(args, rank, world_size)
    
    if rank == 0:
        metrics_file = os.path.join(args.output_dir, f'metrics_{args.bits}bit.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Resolution: {args.downsample_res}\n")
            f.write(f"Quantization bits: {args.bits}\n")
            f.write(f"Metrics:\n")
            for metric_name, metric_value in metrics.items():
                if metric_name == 'memory':
                    f.write(f"\nMemory metrics:\n")
                    if metrics['memory']['encoder_reduction']:
                        orig_size, quant_size = metrics['memory']['encoder_reduction']
                        reduction = (1 - quant_size/orig_size) * 100
                        f.write(f"Encoder reduction: {reduction:.2f}%\n")
                    if metrics['memory']['decoder_reduction']:
                        orig_size, quant_size = metrics['memory']['decoder_reduction']
                        reduction = (1 - quant_size/orig_size) * 100
                        f.write(f"Decoder reduction: {reduction:.2f}%\n")
                    if metrics['memory']['encoder_reduction'] and metrics['memory']['decoder_reduction']:
                        total_orig = metrics['memory']['encoder_reduction'][0] + metrics['memory']['decoder_reduction'][0]
                        total_quant = metrics['memory']['encoder_reduction'][1] + metrics['memory']['decoder_reduction'][1]
                        total_reduction = (1 - total_quant/total_orig) * 100
                        f.write(f"Total model reduction: {total_reduction:.2f}%\n")
                else:
                    f.write(f"{metric_name}: {metric_value:.4f}\n")
            f.write(f"Throughput: {throughput:.2f} samples/sec\n")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()