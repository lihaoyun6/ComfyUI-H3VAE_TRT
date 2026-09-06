# ComfyUI-H3VAE_TRT

在 ComfyUI 中运行 MiniMax-H3 VAE 的 TensorRT 版本，最高可提升约 1.7 倍的编解码速度

## 预览

![](./preview.png)
![](./preview2.png)

## 安装

#### 安装节点：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT.git
python -m pip install -r ComfyUI-H3VAE_TRT/requirements.txt
```

## 用法

### 下载模型

1. 从 [这里](https://huggingface.co/lihaoyun6/MiniMax-H3-VAE-ONNX) 下载 decoder & encoder 的 .onnx 模型, 以及对应的 .data 文件(如果有的话)

	> 如果显存不足 12GB 的话可以使用 `w4a16_awq` 量化版 decoder.  
	
2. 将模型放入 `ComfyUI/models/vae` 目录

### 节点

- 首次使用前请先通过 `MiniMax-H3 TRT Compiler` 节点将 ONNX 模型编译为 TensorRT 引擎。
- 成功编译 TRT Engine 后，就可以使用 `MiniMax-H3 TRT VAE Loader` 节点加载它们了。  

	> 如果需要在 ComfyUI 之外进行编译, 可以使用项目目录中的`compile.py`脚本

## 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) @MiniMax-AI
