# Post-Training Quantization of Image Tokenizers for Autoregressive Image Generation

![Results Overview](assets/ptq.png)

[Report](./PTQ_Report.pdf)

We present **PTQ-Tokenizer**, a framework for efficiently compressing image tokenizers through post-training quantization techniques. This work addresses the computational bottleneck of image tokenization in autoregressive generation models, enabling significantly smaller model sizes while maintaining image quality. We compare logarithmic and per-tensor quantization schemes and demonstrate superior performance with logarithmic methods.

|                      | 8-bit               | 6-bit              | 4-bit             | 2-bit              |
| -------------------- | ------------------- | ------------------ | ----------------- | ------------------ |
| **PSNR (Log)**       | 26.54 dB            | 25.15 dB           | 20.54 dB          | 6.55 dB            |
| **PSNR (Per-Tensor)**| 25.91 dB            | 14.84 dB           | 9.69 dB           | 7.83 dB            |
| **Compression Ratio**| 2.52x               | 4.93x              | 7.85x             | 59.61x             |

Our logarithmic quantization consistently outperforms per-tensor approaches, maintaining PSNR of 25.15 dB versus 14.84 dB at 6-bit quantization while achieving similar compression ratios. Our analysis reveals asymmetric encoder-decoder sensitivity, enabling mixed-precision strategies that achieve up to 7.8x model size reduction while preserving image fidelity.



## Installation

```bash
git clone https://github.com/sukritipaul/Cosmos-Tokenizer.git
cd Cosmos-Tokenizer
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python quantize_cosmos.py \
    --image_path /path/to/images \
    --batch_size 100 \
    --output_path results/ \
    --model_path /path/to/Cosmos-Tokenizer-DI8x8 \
    --n_save 3 \
    --quantize_method per_tensor \
    --encoder_bits 8 \
    --decoder_bits 6
```

### Parameters

- `--image_path`: Directory containing input images
- `--batch_size`: Number of images to process at once (default: 5)
- `--model_path`: Path to pretrained Cosmos tokenizer
- `--output_path`: Directory to save results (default: results/)
- `--n_save`: Number of sample images to save (default: 3)
- `--quantize_method`: Quantization method ('per_tensor' or 'log_quantization')
- `--encoder_bits`: Number of bits for encoder quantization
- `--decoder_bits`: Number of bits for decoder quantization

### Examples

Per-Tensor Quantization (8-bit encoder, 6-bit decoder):
```bash
python quantize_cosmos.py \
    --image_path test_data/ \
    --quantize_method per_tensor \
    --encoder_bits 8 \
    --decoder_bits 6
```

Logarithmic Quantization (6-bit for both):
```bash
python quantize_cosmos.py \
    --image_path test_data/ \
    --quantize_method log_quantization \
    --encoder_bits 6 \
    --decoder_bits 6
```


## Dataset
We use the Boldbrush Artistic Image Dataset (BAID), which contains 5,000 artistic images featuring intricate brush strokes, complex textures, and fine-grained details. These elements are challenging to preserve under aggressive compression schemes, making BAID an ideal test case for evaluating tokenizer quantization.

## Results
Our evaluation reveals that:

- Logarithmic quantization consistently outperforms per-tensor approaches, especially at lower bit widths
- Encoder sensitivity is higher than decoder sensitivity, enabling asymmetric bit allocation
- Mixed-precision strategies (e.g., 6-bit encoder, 4-bit decoder) achieve optimal quality-size tradeoffs

Visual inspection shows logarithmic quantization's superior quality preservation across all bit-width configurations, with particularly strong advantages in preserving artistic details and texture patterns at moderate compression levels.

## Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

Contributions are welcome! Please feel free to submit a Pull Request.


Acknowledgments
This work builds upon the NVIDIA Cosmos Tokenizer and VQVAE. We thank the authors of these projects for making their code and models available.



