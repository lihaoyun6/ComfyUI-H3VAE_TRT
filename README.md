# ComfyUI-H3VAE_TRT 
Running TensorRT version of the MiniMax-H3 VAE in ComfyUI, which can increase speed by up to 1.7x

## Preview
![](./preview.png)
![](./preview2.png)

## Installation

#### Install the node:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT.git
python -m pip install -r ComfyUI-H3VAE_TRT/requirements.txt
```

## Usage 
### Download Models  
1. Download encoder & decoder `.onnx` files (and `.data` files, if present) -> [here](https://huggingface.co/lihaoyun6/MiniMax-H3-VAE-ONNX)  

	> You can use the `w4a16_awq` decoder if you don't have 12GB+ of VRAM.
	
2. Put models into `ComfyUI/models/vae`

### Nodes
- Compile the TensorRT engine from onnx using the `MiniMax-H3 TRT VAE Compiler` node before first use.  
- After compiling the TRT Engines, you can use the `MiniMax-H3 TRT VAE Loader` node to load them.    

	> If you need to compile outside of ComfyUI, use the `compile.py` script in the project.

## Credits
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) @MiniMax-AI
