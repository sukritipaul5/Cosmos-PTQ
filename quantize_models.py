import torch
from typing import Dict
from torch._C import dtype
import numpy as np
from dahuffman import HuffmanCodec
import os
import pickle
DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def save_pickle(obj,path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def quantize_tensor(t, bit=8, axis=-1,dither=False):
    if axis == -1:
        t_valid = t!=0
        if t_valid.sum()==0:
            scale = torch.tensor(0).to(t.device)
            t_min = torch.tensor(0).to(t.device)
        else:
            t_min, t_max =  t[t_valid].min(), t[t_valid].max()
            scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i]!=0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)        
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:,None,None,None]
            t_min = min_max_tf[:,0,None,None,None]
        elif t.dim() == 2:
            scale = scale[:,None]
            t_min = min_max_tf[:,0,None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:,i]!=0
            if t_valid.sum():
                min_max_list.append([t[:,i][t_valid].min(), t[:,i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)             
        scale = (min_max_tf[:,1] - min_max_tf[:,0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None,:,None,None]
            t_min = min_max_tf[None,:,0,None,None]
        elif t.dim() == 2:
            scale = scale[None,:]
            t_min = min_max_tf[None,:,0]            

    if dither:
        print('dithering before quant')
        # Calculate the noise range based on the scale
        noise_range = scale / 2
        # Generate uniform noise in the range [-noise_range, noise_range]
        noise = (torch.rand_like(t) * 2 - 1) * noise_range
        t = t + noise

    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    return quant_t, scale,t_min


def dequantize_tensor(quant_t, scale, t_min):
    return t_min.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t

def per_tensor_quantize(unquantized_state_dict, bits):
    quant_weight_list = []
    scales = []
    t_min_vals = []
    shapes = []

    for key, value in unquantized_state_dict.items():
        quant_t, scale, t_min = quantize_tensor(value, bits)
        
        quant_weight_list.append(quant_t.flatten())
        scales.append(scale.cpu().tolist())
        t_min_vals.append(t_min.cpu().tolist())
        shapes.append(tuple(value.shape))

    cat_param = torch.cat(quant_weight_list)
    input_code_list = cat_param.tolist()
    input_code_list = [int(x) for x in input_code_list]

    print('\nHuffman encoding...')
    codec = HuffmanCodec.from_data(input_code_list)
    encoded = codec.encode(input_code_list)
    
    encoding_info = {
        'codec': codec,
        'encoded': encoded,
        'scales': scales,
        't_min_vals': t_min_vals,
        'shapes': shapes
    }
    
    return encoding_info


def log_quantization(unquantized_state_dict, bits):
    quant_weight_list = []
    params_info = []
    shapes = []

    for key, value in unquantized_state_dict.items():
        # Handle zeros separately
        zero_mask = value == 0
        signs = torch.sign(value)
        abs_tensor = torch.abs(value)
        
        # Log space transformation
        eps = 1e-6
        log_tensor = torch.log(abs_tensor + eps)
        
        # Calculate range in log space for non-zero values
        valid_mask = ~zero_mask
        if valid_mask.sum() > 0:
            min_log = log_tensor[valid_mask].min()
            max_log = log_tensor.max()
        else:
            min_log = torch.tensor(0).to(value.device)
            max_log = torch.tensor(0).to(value.device)
        
        # Quantize in log space
        num_levels = 2**bits - 1  # Reserve one level for zeros
        step_size = (max_log - min_log) / num_levels if num_levels > 0 else torch.tensor(0).to(value.device)
        
        # Quantize
        quantized = torch.round((log_tensor - min_log) / (step_size + 1e-19))
        quantized[zero_mask] = num_levels  # Special value for zeros
        quantized = quantized.to(torch.long)
        
        quant_weight_list.append(quantized.flatten())
        params_info.append({
            'min_log': min_log.cpu().item(),
            'step_size': step_size.cpu().item(),
            'signs': signs,
            'zero_mask': zero_mask
        })
        shapes.append(tuple(value.shape))

    cat_param = torch.cat(quant_weight_list)
    input_code_list = cat_param.tolist()
    input_code_list = [int(x) for x in input_code_list]

    print('\nHuffman encoding...')
    codec = HuffmanCodec.from_data(input_code_list)
    encoded = codec.encode(input_code_list)
    
    encoding_info = {
        'codec': codec,
        'encoded': encoded,
        'params_info': params_info,
        'shapes': shapes
    }
    
    return encoding_info

class QuantizeModel:
    def __init__(self, args, model, quantize_method, bits, target=None):
        self.model = model
        self.quantize_method = quantize_method
        self.bits = bits
        self.args = args
        self.target = target  # 'encoder' or 'decoder'

    def quantize(self):
        state_dict = self.model.state_dict()

        if self.quantize_method == 'per_tensor' or self.quantize_method == 'log_quantization':
            state_dict_to_quantize = {}
            for k,v in state_dict.items():
                if 'weight' in k or 'bias' in k:
                    # Filter based on target (encoder/decoder)
                    if self.target == 'encoder' and 'encoder' in k:
                        state_dict_to_quantize[k] = v
                    elif self.target == 'decoder' and 'decoder' in k:
                        state_dict_to_quantize[k] = v

            self.quantized_keys = list(state_dict_to_quantize.keys())


            if self.quantize_method == 'per_tensor':
                encoding_info = per_tensor_quantize(state_dict_to_quantize, self.bits)
            elif self.quantize_method == 'log_quantization':
                encoding_info = log_quantization(state_dict_to_quantize, self.bits)

            encoding_info['quantized_keys'] = self.quantized_keys

            # Save the encoding info with target-specific filename
            filename = f'encoding_info_{self.target}_{self.bits}bits.pkl'
            self.quant_out_path = os.path.join(self.args.output_path, 'quantized_model', filename)
            os.makedirs(os.path.dirname(self.quant_out_path), exist_ok=True)
            save_pickle(encoding_info, self.quant_out_path)
            
            size_in_bits = len(encoding_info['encoded']) * self.bits / 8
            size_in_MB = size_in_bits / (1024 * 1024)            
            print(f'Size in MB after quantizing {self.target} to {self.bits} bits: {size_in_MB}')

            return size_in_MB

        #elif self.quantize_method == 'log_quantization':
            
            

    def dequantize(self, quant_out_path=None):
        if quant_out_path is None:
            quant_out_path = self.quant_out_path

        quantized_state = load_pickle(quant_out_path)

        if self.quantize_method == 'per_tensor':
            codec = quantized_state['codec']
            encoded = quantized_state['encoded']
            scales = quantized_state['scales']
            t_min_vals = quantized_state['t_min_vals']
            shapes = quantized_state['shapes']

            print('\nHuffman decoding...')
            decoded = codec.decode(encoded)
            print('decoded')

            reconstructed_state = {}
            start = 0
            for k, key in enumerate(quantized_state['quantized_keys']):
                end = start + int(np.prod(shapes[k], dtype=np.int64))
                try:
                    temp = torch.tensor(decoded[start:end], dtype=torch.long).reshape(shapes[k])
                except:                    
                    temp = torch.tensor(decoded[start:end], dtype=torch.float).reshape(shapes[k])
                    
                temp = t_min_vals[k] + scales[k] * temp
                reconstructed_state[key] = temp
                start = end

            return reconstructed_state

        elif self.quantize_method == 'log_quantization':
            codec = quantized_state['codec']
            encoded = quantized_state['encoded']
            params_info = quantized_state['params_info']
            shapes = quantized_state['shapes']

            print('\nHuffman decoding...')
            decoded = codec.decode(encoded)

            reconstructed_state = {}
            start = 0
            for k, key in enumerate(quantized_state['quantized_keys']):
                end = start + int(np.prod(shapes[k], dtype=np.int64))
                # Reshape quantized values
                quantized = torch.tensor(decoded[start:end], dtype=torch.float).reshape(shapes[k])
                
                # Get parameters for this tensor
                info = params_info[k]
                min_log = info['min_log']
                step_size = info['step_size']
                signs = info['signs']
                zero_mask = info['zero_mask']
                
                # Dequantize in log space
                num_levels = 2**self.bits - 1
                is_zero = quantized == num_levels
                
                # Reverse the quantization
                log_tensor = min_log + step_size * quantized
                # Convert back from log space
                abs_tensor = torch.exp(log_tensor)
                # Apply signs and handle zeros
                signs = signs.to(abs_tensor.device)
                reconstructed = signs * abs_tensor
                reconstructed[zero_mask] = 0
                
                reconstructed_state[key] = reconstructed
                start = end

            return reconstructed_state
