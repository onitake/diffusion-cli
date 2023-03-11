# Diffuser CLI

A simple command line tool to generate images with diffuser pipelines.

Uses the ROCm version of PyTorch by default, for AMD GPU support.
See below if you prefer the vanilla version with CUDA.

## Disclaimer

This script is provided for personal entertainment purposes only.
Please respect the rights of others when using it, and honor the licenses
of the models you download.

Please see the LICENSE file for details on permitted use of the script itself.

## Quickstart

For a good initial experience, download a popular model, such as
[stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
or [waifu-diffusion](https://huggingface.co/hakurei/waifu-diffusion), as well as the
[LPW pipeline script](https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py).
Make sure you have git-lfs installed, or Git won't download the big files!

```
wget <url-to-lpw_stable_diffusion.py>
git clone <repository-url-here>
```

PyTorch for AMD GPUs requires the ROCm HIP runtime.
See [AMD ROCm Release Documentation](https://docs.amd.com/category/Release%20Documentation)
for installation instructions. The package is called: `rocm-hip-runtime`

Then start the CLI with:
```
pipenv run ./diffuse.py --batch 1 --model waifu-diffusion --custom lpw_stable_diffusion.py
```

The first run will download all necessary Python modules and precompile
Torch kernels. Subsequent runs will be faster.

On a GPU with sufficient VRAM, you can increase the batch size to generate
multiple images in one go, with increased performance.

`pipenv run ./diffuse.py --help` will show other supported options.

## Image replacement

Image replacement (or image2image mode) is also supported.

Use the `--image` parameter to specify a starting image, and control the influence
with `--strength`. Lower values correspond with a stronger influence.
The valid range is from 0.0 to 1.0.

## Other GPUs or operating systems

Depending on the target platform, you need to use a different PyTorch flavor.
The default is ROCm for AMD GPUs.

Refer to https://pytorch.org/get-started/locally/ for other options.

Once you've modified the `Pipfile`, you need to run `pipenv install` to
download the new PyTorch framework.

### NVIDIA CUDA

In `Pipfile`, Replace the line that says
```
torch = {version = "*", index = "pytorch"}
```
with
```
torch = "*"
```

### CPU only

In `Pipfile`, Replace the line that says
```
url = "https://download.pytorch.org/whl/rocm5.2/"
```
with
```
url = "https://download.pytorch.org/whl/cpu/"
```
