# Installation

## VP Hough Transform and Gaussian Sphere

Install dependencies:

```bash
sudo apt-get install ninja-build
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install scipy matplotlib scikit-learn scikit-image torch_geometric
```

