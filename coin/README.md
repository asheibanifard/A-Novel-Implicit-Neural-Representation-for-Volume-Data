# COIN: COmpression with Implicit Neural representations

Implementation of the COIN paper ([arXiv:2103.03123](https://arxiv.org/abs/2103.03123)) for MRI image compression.

## Overview

COIN uses a small MLP (Multilayer Perceptron) with SIREN activations to represent images. Instead of storing pixel values, we store the network weights that can reconstruct the image.

**Key idea**: Train an MLP to map (x, y) coordinates → pixel intensity, then store the weights.

## Files

- `model.py` - COIN model with SIREN layers
- `train.py` - Training script for MRI slices
- `demo.ipynb` - Interactive demo notebook

## Usage

```bash
# Train on a single MRI slice
python train.py --modality T1 --slice_idx 38 --epochs 2000

# With custom architecture
python train.py --hidden_features 128 --hidden_layers 4 --epochs 3000
```

## Model Architecture

```
Input: (x, y) coordinates ∈ [-1, 1]²
    ↓
SineLayer (2 → 256, ω₀=30)
    ↓
SineLayer (256 → 256) × 3
    ↓
Linear (256 → 1)
    ↓
Output: Pixel intensity ∈ [0, 1]
```

## References

```bibtex
@article{dupont2021coin,
  title={COIN: COmpression with Implicit Neural representations},
  author={Dupont, Emilien and Goliński, Adam and Alizadeh, Milad and Teh, Yee Whye and Doucet, Arnaud},
  journal={arXiv preprint arXiv:2103.03123},
  year={2021}
}
```
