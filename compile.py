import argparse
import os
import sys
import time

# ================= 1. Fast CLI Argument Parser (Instant --help) =================
def parse_args():
  parser = argparse.ArgumentParser(
      description="TensorRT Engine Compiler for MiniMax-H3 VAE",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "-i",
      "--input",
      type=str,
      required=True,
      help="Path to input ONNX model file (.onnx)",
  )
  parser.add_argument(
      "-o",
      "--output",
      type=str,
      default=None,
      help=(
          "Path to output TensorRT engine file (.engine). Defaults to the same"
          " directory and name as the input file."
      ),
  )
  parser.add_argument(
      "-d",
      "--device",
      type=int,
      default=0,
      help="CUDA device index to use for compilation (e.g., 0, 1)",
  )
  parser.add_argument(
      "-w",
      "--workspace",
      type=float,
      default=None,
      help=(
          "Maximum workspace memory limit in GB. Defaults to 4.0GB for Decoder,"
          " 8.0GB for Encoder"
      ),
  )
  return parser.parse_args()


# ================= 2. Heavy Engine Builder (Loaded on demand) =================
def is_onnx_quantized(onnx_path: str) -> bool:
  """Detect if the ONNX model is quantized (W4A16 / INT8 / AWQ)."""
  import onnx

  filename_lower = os.path.basename(onnx_path).lower()
  quant_keywords = ["w4a", "w8a", "int4", "int8", "fp4", "fp8", "awq", "quant"]
  if any(kw in filename_lower for kw in quant_keywords):
    return True

  try:
    model = onnx.load(onnx_path, load_external_data=False)
    for init in model.graph.initializer:
      if init.data_type in (21, 22, 3):  # UINT4, INT4, INT8
        return True
    for node in model.graph.node:
      if node.op_type in ("DequantizeLinear", "QuantizeLinear"):
        return True
  except Exception:
    pass

  return False


def build_engine(
    onnx_path: str,
    output_path: str,
    device_id: int = 0,
    workspace_gb: float = None,
):
  if not os.path.exists(onnx_path):
    print(f"[Error] Input ONNX file does not exist: {onnx_path}")
    sys.exit(1)

  print("[*] Loading dependencies (PyTorch, ONNX, TensorRT)...")
  import onnx
  import torch

  try:
    import tensorrt as trt
  except ImportError:
    print(
        "[Error] TensorRT Python library not found. Please install tensorrt"
        " first!"
    )
    sys.exit(1)

  # 1. Bind and validate target CUDA device
  if not torch.cuda.is_available():
    print("[Error] CUDA is not available on this system.")
    sys.exit(1)

  if device_id < 0 or device_id >= torch.cuda.device_count():
    print(
        f"[Error] Invalid device ID {device_id}. Available devices:"
        f" {torch.cuda.device_count()}"
    )
    sys.exit(1)

  torch.cuda.set_device(device_id)
  gpu_name = torch.cuda.get_device_name(device_id)

  print("=" * 70)
  print(" TensorRT Standalone Engine Compiler")
  print("=" * 70)
  print(f"[*] Target GPU           : cuda:{device_id} ({gpu_name})")
  print(f"[*] Input ONNX Path      : {onnx_path}")
  print(f"[*] Output Engine Path   : {output_path}")

  # 2. Inspect quantization type
  is_quantized = is_onnx_quantized(onnx_path)

  logger = trt.Logger(trt.Logger.INFO)
  builder = trt.Builder(logger)
  config = builder.create_builder_config()

  # 3. Create Network Definition
  if is_quantized:
    print(
        "[*] Precision Mode       : Quantized Model (W4A16 / INT8) ->"
        " Strongly-Typed Mode"
    )
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
      flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
      network = builder.create_network(flags)
    else:
      network = builder.create_network()
  else:
    print(
        "[*] Precision Mode       : Unquantized Model -> FP16 Tensor Cores"
        " Enabled"
    )
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
      flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
      network = builder.create_network(flags)
    else:
      network = builder.create_network()

    if hasattr(trt.BuilderFlag, "FP16"):
      config.set_flag(trt.BuilderFlag.FP16)

  # 4. Parse ONNX Graph from file path (ensuring .onnx.data resolution)
  print("[*] Parsing ONNX computational graph...")
  parser = trt.OnnxParser(network, logger)
  if not parser.parse_from_file(onnx_path):
    print("\n[Error] Failed to parse ONNX model:")
    for error in range(parser.num_errors):
      print(f"  -> {parser.get_error(error)}")
    sys.exit(1)

  # 5. Detect input tensor and configure optimization profile
  input_tensor = network.get_input(0)
  input_name = input_tensor.name
  input_shape = input_tensor.shape

  # Detect whether the model is a Decoder or Encoder
  is_decoder = "latent" in input_name.lower() or input_shape[1] == 24

  if is_decoder:
    model_type = "Decoder"
    if len(input_shape) >= 4 and input_shape[3] == 32:
      shape = (1, 24, 7, 32, 32)
      tile_desc = "512px Spatial Tile (32x32 Latent)"
    else:
      shape = (1, 24, 7, 16, 16)
      tile_desc = "256px Standard Tile (16x16 Latent)"
    default_ws = 4.0
  else:
    model_type = "Encoder"
    shape = (1, 3, 17, 256, 256)
    tile_desc = "17-Frame Causal Tile (256x256 Pixel)"
    default_ws = 8.0

  workspace_gb = workspace_gb if workspace_gb is not None else default_ws
  print(f"[*] Model Architecture   : {model_type} -> {tile_desc}")
  print(f"[*] Input Tensor Profile : '{input_name}' shape={shape}")
  print(f"[*] Workspace Limit      : {workspace_gb:.1f} GB")

  profile = builder.create_optimization_profile()
  profile.set_shape(input_name, shape, shape, shape)
  config.add_optimization_profile(profile)

  # Configure workspace memory pool
  ws_bytes = int(workspace_gb * (1024**3))
  if hasattr(config, "set_memory_pool_limit"):
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
  else:
    config.max_workspace_size = ws_bytes

  # 6. Build and serialize the engine
  print("=" * 70)
  print(
      "[*] Building TensorRT Engine (typically takes 1 ~ 3 minutes, please"
      " wait)..."
  )
  t0 = time.time()

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

  if hasattr(builder, "build_serialized_network"):
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
      print("\n[Error] Failed to build TensorRT engine.")
      sys.exit(1)
    with open(output_path, "wb") as f:
      f.write(serialized_engine)
  else:
    engine = builder.build_engine(network, config)
    if engine is None:
      print("\n[Error] Failed to build TensorRT engine.")
      sys.exit(1)
    with open(output_path, "wb") as f:
      f.write(engine.serialize())

  elapsed = time.time() - t0
  out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
  out_size_gb = out_size_mb / 1024

  print("=" * 70)
  if out_size_gb >= 1.0:
    print(
        f"🎉 Build Succeeded! Engine Size: {out_size_gb:.2f} GB | Elapsed Time:"
        f" {elapsed:.1f}s"
    )
  else:
    print(
        f"🎉 Build Succeeded! Engine Size: {out_size_mb:.2f} MB | Elapsed Time:"
        f" {elapsed:.1f}s"
    )
  print(f"[*] Engine saved to: {output_path}")
  print("=" * 70)


# ================= 3. Entry Point =================
def main():
  # Parse arguments immediately before importing torch/tensorrt
  args = parse_args()

  if args.output is None:
    output_engine_path = os.path.splitext(args.input)[0] + ".engine"
  else:
    output_engine_path = args.output

  build_engine(
      onnx_path=args.input,
      output_path=output_engine_path,
      device_id=args.device,
      workspace_gb=args.workspace,
  )


if __name__ == "__main__":
  main()