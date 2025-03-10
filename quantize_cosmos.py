import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
import numpy as np
import argparse
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import tqdm
import os,glob
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import torchvision
from quantize_models import QuantizeModel
import json


def save_tensor_img(tensor,path='temp.png'):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor * 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculates PSNR between two images efficiently.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        max_val (float): Maximum value of the images (default: 1.0 for normalized images)
    
    Returns:
        float: PSNR value
    """
    # Ensure the images are on the same device
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    # Calculate MSE
    mse = torch.mean((img1 - img2) ** 2)
    # Calculate PSNR
    psnr_val = 20 * torch.log10(torch.tensor(max_val, device=img1.device)) - 10 * torch.log10(mse)
    return psnr_val.item()


def calculate_ssim(img1, img2):
    img1 = img1.float()
    img2 = img2.float()
    # Convert torch tensors to numpy arrays and move to CPU if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Reshape from (B, C, H, W) to (H, W, C) for each image in batch
    # Calculate mean SSIM across batch
    ssim_values = []
    for i in range(img1.shape[0]):  # Loop through batch
        img1_i = np.transpose(img1[i], (1, 2, 0))  # (C,H,W) -> (H,W,C)
        img2_i = np.transpose(img2[i], (1, 2, 0))  # (C,H,W) -> (H,W,C)
        ssim_val = ssim(img1_i, img2_i, data_range=1.0, channel_axis=2, size=11)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


class CosmosModel(nn.Module):
    def __init__(self,model_path):
        super().__init__()
        self.encoder = ImageTokenizer(checkpoint_enc=model_path+'/encoder.jit')
        self.decoder = ImageTokenizer(checkpoint_dec=model_path+'/decoder.jit')

        #record original model size
        self.original_model_size = os.path.getsize(model_path+'/encoder.jit') + os.path.getsize(model_path+'/decoder.jit')
        self.original_model_size = self.original_model_size / (1024 * 1024)
        print(f'Original model size: {self.original_model_size} MB')

        self.encoder_model_size = os.path.getsize(model_path+'/encoder.jit')
        self.decoder_model_size = os.path.getsize(model_path+'/decoder.jit')
        print(f'Encoder model size: {self.encoder_model_size / (1024 * 1024)} MB')
        print(f'Decoder model size: {self.decoder_model_size / (1024 * 1024)} MB')

    def encode(self,x):
        x = x.to(torch.bfloat16)

        out = self.encoder.encode(x)
        
        if len(out) == 2:
            latents,codes = out
            output = {'latents':latents,'codes':codes}
        else:
            latents = out[0] if isinstance(out,tuple) else out
            output = {'latents':latents}

        return output

    def decode(self,x):        
        out = self.decoder.decode(x)
        return out


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted(glob.glob(os.path.join(folder_path,'*.jpg')))
        self.files = self.files[:1000]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        #convert to RGB
        img = img.convert('RGB')
        img = self.transform(img)
        return img

def custom_collate_fn(batch):
    # Process each image separately since they might have different sizes
    return batch

def main(args):
    dataset = ImageDataset(args.image_path)

    #create output path if not exists
    os.makedirs(args.output_path,exist_ok=True)

    # Add collate_fn to handle different sized images
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    

    model = CosmosModel(args.model_path)

    
    if args.quantize_method is not None:
        total_size = 0
        dequant_state_dict = {}
        
        # Quantize encoder if encoder_bits specified
        if args.encoder_bits is not None:
            quant = QuantizeModel(args, model, args.quantize_method, args.encoder_bits, target='encoder')
            encoder_size = quant.quantize()
            print(f'Quantized encoder size: {encoder_size} MB')
            encoder_dequant = quant.dequantize()
            dequant_state_dict.update(encoder_dequant)
            total_size += encoder_size
        
        # Quantize decoder if decoder_bits specified
        if args.decoder_bits is not None:
            quant = QuantizeModel(args, model, args.quantize_method, args.decoder_bits, target='decoder')
            decoder_size = quant.quantize()
            print(f'Quantized decoder size: {decoder_size} MB')
            decoder_dequant = quant.dequantize()
            dequant_state_dict.update(decoder_dequant)
            total_size += decoder_size
        
        print(f'Total quantized model size: {total_size} MB')
        
        # Load quantized weights back into model
        state_dict = model.state_dict()
        for k,v in dequant_state_dict.items():
            state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda()
    model.eval()

    os.makedirs(args.output_path,exist_ok=True)


    psnr_list = []
    ssim_list = []

    # Update the processing loop to handle individual images
    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        # Process each image in the batch separately
        batch_psnr = []
        batch_ssim = []
        batch_outputs = []
        
        for img in batch:
            img = img.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
            encoded = model.encode(img)
            out = model.decode(encoded['latents'])
            out = out.clamp(0, 1)
            
            #sometimes output size might not match. slightly off due to padding.
            if out.shape != img.shape:
                # Handle height (dimension 2)
                min_height = min(out.shape[2], img.shape[2])
                img = img[:,:,:min_height,:]
                out = out[:,:,:min_height,:]
                
                # Handle width (dimension 3)
                min_width = min(out.shape[3], img.shape[3])
                img = img[:,:,:,:min_width]
                out = out[:,:,:,:min_width]

            psnr = calculate_psnr(img, out)
            ssim = calculate_ssim(img, out)
            batch_psnr.append(psnr)
            batch_ssim.append(ssim)
            batch_outputs.append(out)
            
        psnr_list.extend(batch_psnr)
        ssim_list.extend(batch_ssim)

        if i < 2:
            # Save the first n_save images from the batch
            for j in range(min(args.n_save, len(batch))):
                input_img = batch[j].unsqueeze(0).cuda()
                output_img = batch_outputs[j]
                
                # Match dimensions before concatenating
                min_height = min(input_img.shape[2], output_img.shape[2])
                min_width = min(input_img.shape[3], output_img.shape[3])
                
                input_img = input_img[:,:,:min_height,:min_width]
                output_img = output_img[:,:,:min_height,:min_width]
                
                stitched = torch.cat([input_img, output_img], dim=3)
                torchvision.utils.save_image(
                    stitched,
                    os.path.join(args.output_path, f'comparison_batch_{i}_img_{j}.png'),
                    nrow=1,
                    normalize=False
                )

    print(f'PSNR: {np.mean(psnr_list)}, SSIM: {np.mean(ssim_list)}')

    info = {
        'psnr': float(np.mean(psnr_list)),
        'ssim': float(np.mean(ssim_list)),
        'quantize_method': args.quantize_method,
        'encoder_bits': args.encoder_bits,
        'decoder_bits': args.decoder_bits,
        'model_path': args.model_path,
        'image_path': args.image_path,
        'output_path': args.output_path,
        'original_model_size_MB': model.original_model_size,
        'encoder_model_size_MB': model.encoder_model_size / (1024 * 1024),
        'decoder_model_size_MB': model.decoder_model_size / (1024 * 1024)
    }

    if args.quantize_method is not None:
        info['quantized_model_size'] = total_size
        if args.encoder_bits is not None:
            info['quantized_encoder_size'] = encoder_size
        if args.decoder_bits is not None:
            info['quantized_decoder_size'] = decoder_size

    #save as json
    with open(os.path.join(args.output_path,'info.json'),'w') as f:
        json.dump(info, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the image file',default='test_data/')
    parser.add_argument('--batch_size','-bs',type=int,default=5)    
    parser.add_argument('--model_path','-mp',type=str,default='/fs/cml-projects/yet-another-diffusion/Cosmos-Tokenizer/scripts/pretrained_ckpts/Cosmos-Tokenizer-DI8x8')
    parser.add_argument('--output_path','-op',type=str,default='results/')
    parser.add_argument('--n_save','-ns',type=int,default=3)

    parser.add_argument('--quantize_method','-qm',type=str,default=None)   #None, 'per_tensor', 'log_quantization'
    parser.add_argument('--encoder_bits',type=int,help='Bits for encoder quantization',default=8)
    parser.add_argument('--decoder_bits',type=int,help='Bits for decoder quantization',default=8)
    
    args = parser.parse_args()

    # Validate bits arguments
    if (args.encoder_bits is None) != (args.decoder_bits is None):
        parser.error("Either specify both --encoder_bits and --decoder_bits or neither")
    
    if args.encoder_bits is not None:
        args.bits = None  # Clear the old bits field
    
    main(args)



