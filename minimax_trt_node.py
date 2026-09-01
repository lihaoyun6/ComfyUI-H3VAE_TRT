import gc
import logging
import math
import os
import comfy.cli_args
import comfy.model_management as mm
import comfy.model_patcher
import comfy.sd
import folder_paths
from server import PromptServer
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  import tensorrt as trt

  HAS_TRT = True
except ImportError:
  HAS_TRT = False

logger = logging.getLogger(__name__)


# ================= 1. 将 TensorRT 显存分配权移交给 PyTorch =================


class PyTorchGpuAllocator(trt.IGpuAllocator if HAS_TRT else object):

  def __init__(self):
    super().__init__()
    self.allocated_tensors = {}

  def allocate(self, size, alignment, flags):
    tensor = torch.empty(size, dtype=torch.uint8, device="cuda")
    ptr = tensor.data_ptr()
    self.allocated_tensors[ptr] = tensor
    return ptr

  def deallocate(self, ptr):
    if ptr in self.allocated_tensors:
      del self.allocated_tensors[ptr]


# ================= 2. 内存/显存引擎执行器 =================


class AutoEngineRunner:

  def __init__(self, model_path: str):
    self.model_path = model_path
    self.stream = None
    self.engine_bytes = None
    self.runtime = None
    self.engine = None
    self.context = None
    self.gpu_allocator = None

  def load_to_ram(self):
    if self.engine_bytes is None:
      with open(self.model_path, "rb") as f:
        self.engine_bytes = f.read()

  def load_to_gpu(self):
    if self.context is not None:
      return

    self.load_to_ram()
    if not HAS_TRT:
      raise RuntimeError("TensorRT library not found!")

    if self.runtime is None:
      self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
      if hasattr(trt, "IGpuAllocator"):
        self.gpu_allocator = PyTorchGpuAllocator()
        self.runtime.gpu_allocator = self.gpu_allocator

    if not self.engine_bytes or len(self.engine_bytes) == 0:
      raise RuntimeError(f"Engine file is empty or corrupted: {self.model_path}")

    self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
    if self.engine is None:
      raise RuntimeError(
          f"Failed to deserialize TensorRT engine:"
          f" '{os.path.basename(self.model_path)}'.\n"
          f"Please re-compile the engine on this machine using the 'MiniMax-H3"
          " TRT VAE Compiler' node."
      )

    self.context = self.engine.create_execution_context()
    if self.context is None:
      raise RuntimeError("Failed to create TensorRT execution context (Out of VRAM)!")

  def offload_to_ram(self):
    if self.context is not None:
      self.context = None
      self.engine = None
      self.stream = None
      if self.gpu_allocator is not None:
        self.gpu_allocator.allocated_tensors.clear()
      torch.cuda.empty_cache()

  def infer(self, input_tensor: torch.Tensor, output_shape: tuple, input_name: str) -> torch.Tensor:
    self.load_to_gpu()

    device = input_tensor.device
    dtype = input_tensor.dtype
    input_tensor = input_tensor.contiguous()

    if self.stream is None or self.stream.device != device:
      self.stream = torch.cuda.Stream(device=device)

    self.stream.wait_stream(torch.cuda.current_stream(device))

    output = torch.zeros(output_shape, dtype=dtype, device=device)
    self.context.set_input_shape(input_name, input_tensor.shape)
    self.context.set_tensor_address(input_name, input_tensor.data_ptr())
    out_name = "pixel_tile" if input_name == "latent_tile" else "moments_tile"
    self.context.set_tensor_address(out_name, output.data_ptr())

    self.context.execute_async_v3(self.stream.cuda_stream)
    self.stream.synchronize()
    return output


# ================= 3. MiniMax-H3 加速 VAE 实现 =================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LATENTS_MEAN = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525,
]


class MiniMaxH3TRTVAE(nn.Module):

  def __init__(self, decoder_runner: AutoEngineRunner = None, encoder_runner: AutoEngineRunner = None,):
    super().__init__()
    self.vae_dtype = torch.float16
    self.decoder_runner = decoder_runner
    self.encoder_runner = encoder_runner

    self.vae_ratio = 16
    self.vae_ratio_t = 4
    self.clip_length = 17
    self.token_drop = 3
    self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
    self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
    self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
    self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)

    self.tile_size = 256
    self.tile_overlap_min = 64

    self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1, 1), persistent=False,)
    self.register_buffer("latents_std", torch.tensor(LATENTS_STD).view(1, -1, 1, 1, 1), persistent=False,)
    self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False,)
    self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False,)

  def load_runners_to_gpu(self):
    if self.decoder_runner is not None:
      self.decoder_runner.load_to_gpu()
    if self.encoder_runner is not None:
      self.encoder_runner.load_to_gpu()

  def offload_runners_to_ram(self):
    if self.decoder_runner is not None:
      self.decoder_runner.offload_to_ram()
    if self.encoder_runner is not None:
      self.encoder_runner.offload_to_ram()

  def get_total_engine_size(self):
    total_size = 0
    if self.decoder_runner and os.path.exists(self.decoder_runner.model_path):
      total_size += os.path.getsize(self.decoder_runner.model_path)
    if self.encoder_runner and os.path.exists(self.encoder_runner.model_path):
      total_size += os.path.getsize(self.encoder_runner.model_path)
    return max(total_size, 1024 * 1024 * 500)

  def _decode_temporal_chunks(self, z_len):
    pseudo_total_tokens = z_len + self.token_drop
    pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
    pseudo_total_tokens += pad_tokens
    num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
    if num_chunks < 1:
      pad_tokens += self.tokens_chunk_size
      num_chunks += 1
    return pad_tokens, num_chunks

  def decode_temporal(self, z):
    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    
    # 🌟 关键修复：使用官方原版方法计算，包含不足 1 块时的 pad 保护 (T_lat=2 -> 补齐到 7)
    pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
    
    if pad_tokens > 0:
      z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
      
    dec_chunks, dec_overlap = [], None
    for i in range(num_chunks):
      t_start = i * self.tokens_chunk_size
      clip_z = z[
          :, :,
          t_start : t_start + self.tokens_chunk_size + self.token_overlap,
          :, :,
      ]
      clip_dec = self.tiled_decode(clip_z)
      
      for j in range(split_count):
        f_start = j * chunk_dec
        f_end = min(f_start + chunk_dec, clip_dec.shape[2])
        chunk = clip_dec[:, :, f_start:f_end, :, :]
        chunk = chunk[:, :, self.frame_pre_padding :, :, :]
        
        if j == 0:
          if dec_overlap is not None:
            chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
            dec_overlap = None
          dec_chunks.append(self._finalize_pixels(chunk))
        else:
          dec_overlap = chunk.contiguous()
          
      if i == num_chunks - 1 and dec_overlap is not None:
        dec_chunks.append(self._finalize_pixels(dec_overlap))
        
    return torch.cat(dec_chunks, dim=2)

  def _adaptive_decode(self, z):
    return self.tiled_decode(z)

  def _decode_temporal_pad_frames(self, z_len, pad_tokens):
    if pad_tokens <= 0:
      return 0
    intra_tail = self.clip_length % self.vae_ratio_t
    if intra_tail == 0:
      return pad_tokens * self.vae_ratio_t
    z_len_before_pad = z_len - pad_tokens
    return sum(
        intra_tail
        if (z_len_before_pad + k) % self.tokens_chunk_size == 0
        else self.vae_ratio_t
        for k in range(pad_tokens)
    )

  def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):
    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
    total_frames = 0
    final_overlap_frames = 0

    for i in range(num_chunks):
      t_start_idx = i * self.tokens_chunk_size
      t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
      clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
      clip_frame_len = clip_token_len * self.vae_ratio_t

      for j in range(split_count):
        f_start_idx = j * chunk_dec
        f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
        chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
        if j == 0:
          total_frames += chunk_frames
        else:
          final_overlap_frames = chunk_frames

    total_frames += final_overlap_frames
    return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)

  def decode_output_shape(self, input_shape):
    b, c, t, h, w = input_shape
    if t == 1:
      frames = 1
    else:
      pad_tokens, num_chunks = self._decode_temporal_chunks(t)
      frames = self._decode_temporal_frame_plan(t + pad_tokens, num_chunks, pad_tokens)
      frames = self._decode_temporal_frame_plan(t + pad_tokens, num_chunks, pad_tokens)
    return (b, 3, frames, h * self.vae_ratio, w * self.vae_ratio)

  def _decode_pixels(self, z):
    b, _, t, h, w = z.shape
    out_shape = (b, 3, t * self.vae_ratio_t, h * self.vae_ratio, w * self.vae_ratio,)
    return self.decoder_runner.infer(z, output_shape=out_shape, input_name="latent_tile")

  def _encode_moments(self, x):
    b, c, t, h, w = x.shape
    target_h, target_w = self.tile_size, self.tile_size
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if pad_h > 0 or pad_w > 0:
      x_in = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="constant", value=0.0)
    else:
      x_in = x

    out_shape = (
        b,
        48,
        math.ceil(t / self.vae_ratio_t),
        target_h // self.vae_ratio,
        target_w // self.vae_ratio,
    )
    moments = self.encoder_runner.infer(x_in, output_shape=out_shape, input_name="pixel_tile")

    out_h = math.ceil(h / self.vae_ratio)
    out_w = math.ceil(w / self.vae_ratio)
    return moments[..., :out_h, :out_w]

  def _finalize_pixels(self, part):
    return (part * self.pixel_std.to(part) + self.pixel_mean.to(part)).clamp(0.0, 1.0)

  def _normalize_pixels(self, x):
    return (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)

  def blend(self, a, b, blend_extent, dim):
    blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
    if blend_extent <= 0:
      return b

    weight = (torch.arange(blend_extent, device=b.device, dtype=b.dtype) / blend_extent)
    shape = [1] * a.ndim
    shape[dim] = blend_extent
    weight = weight.view(shape)

    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-blend_extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, blend_extent)

    blended = torch.lerp(a[tuple(slice_a)], b[tuple(slice_b)], weight)
    if blend_extent < b.shape[dim]:
      slice_b_rest = [slice(None)] * b.ndim
      slice_b_rest[dim] = slice(blend_extent, None)
      return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
    return blended

  def split_tiles(self, input_len):
    if self.tile_size >= input_len:
      return [0], [input_len], []
    N = math.ceil(input_len / self.tile_size)
    while True:
      overlaps = [self.tile_overlap_min] * (N - 1)
      remaining = self.tile_size * N - sum(overlaps) - input_len
      if remaining < 0:
        N += 1
      else:
        break
    for i in range(remaining // self.vae_ratio):
      overlaps[i % (N - 1)] += self.vae_ratio
    tile_start_idx = [0]
    for i in range(N - 1):
      tile_start_idx.append(tile_start_idx[-1] + self.tile_size - overlaps[i])
    return tile_start_idx, [self.tile_size] * N, overlaps

  def tiled_decode(self, z):
    height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)

    canvas, row_tails, out_y = None, [], 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
      zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
      new_tails, left_tail, out_x = [], None, 0
      for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
        zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
        tile = self._decode_pixels(z[..., zi : zi + zl, zj : zj + zw])

        if i < len(y_idx) - 1:
          new_tails.append(tile[..., -y_overlap[i] :, :].clone())
        next_left_tail = (tile[..., :, -x_overlap[j] :].clone() if j < len(x_idx) - 1 else None)

        if i > 0:
          tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
        if j > 0:
          tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
        left_tail = next_left_tail

        if i < len(y_idx) - 1:
          tile = tile[..., : -y_overlap[i], :]
        if j < len(x_idx) - 1:
          tile = tile[..., :, : -x_overlap[j]]

        if canvas is None:
          canvas = torch.zeros(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device,)
        canvas[
            ...,
            out_y : out_y + tile.shape[-2],
            out_x : out_x + tile.shape[-1],
        ].copy_(tile)
        out_x += tile.shape[-1]
      row_tails = new_tails
      out_y += tile.shape[-2]
    return canvas

  def tiled_encode(self, x):
    height, width = x.shape[-2], x.shape[-1]
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)

    rows = []
    for i_pos, i_len in zip(y_idx, y_len):
      row = []
      for j_pos, j_len in zip(x_idx, x_len):
        tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
        row.append(self._encode_moments(tile))
      rows.append(row)

    latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
    latent_x_overlap = [o // self.vae_ratio for o in x_overlap]

    result_rows = []
    for i, row in enumerate(rows):
      result_row = []
      for j, tile in enumerate(row):
        if i > 0:
          tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
        if j > 0:
          tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
        if i < len(rows) - 1:
          tile = tile[..., : -latent_y_overlap[i], :]
        if j < len(row) - 1:
          tile = tile[..., :, : -latent_x_overlap[j]]
        result_row.append(tile)
      result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)

  def decode_temporal(self, z):
    chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
    split_count = int(self.token_drop > 0) + 1
  
    # 🌟 核心修复：直接调用 _decode_temporal_chunks，当 T_lat == 2 时会自动把 pad_tokens 设为 5，num_chunks 设为 1
    pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
  
    if pad_tokens > 0:
      z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
      
    dec_chunks, dec_overlap = [], None
    for i in range(num_chunks):
      t_start = i * self.tokens_chunk_size
      t_end = t_start + self.tokens_chunk_size + self.token_overlap
      clip_z = z[:, :, t_start:t_end, :, :]
      clip_dec = self.tiled_decode(clip_z)
      
      for j in range(split_count):
        f_start = j * chunk_dec
        f_end = min(f_start + chunk_dec, clip_dec.shape[2])
        chunk = clip_dec[:, :, f_start:f_end, :, :]
        chunk = chunk[:, :, self.frame_pre_padding :, :, :]
        
        if j == 0:
          if dec_overlap is not None:
            chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
            dec_overlap = None
          dec_chunks.append(self._finalize_pixels(chunk))
        else:
          dec_overlap = chunk.contiguous()
          
      if i == num_chunks - 1 and dec_overlap is not None:
        dec_chunks.append(self._finalize_pixels(dec_overlap))
        
    return torch.cat(dec_chunks, dim=2)

  def encode_temporal(self, x):
    z_list = []
    num_clips = math.ceil(x.shape[2] / self.clip_length)
    for i in range(num_clips):
      clip_x = x[:, :, i * self.clip_length : (i + 1) * self.clip_length, :, :]
      if clip_x.shape[2] < self.clip_length:
        pad_frames = clip_x[:, :, -1:].repeat(1, 1, self.clip_length - clip_x.shape[2], 1, 1)
        clip_x = torch.cat([clip_x, pad_frames], dim=2)
      z_list.append(self.tiled_encode(self._normalize_pixels(clip_x)))

    z = torch.cat(z_list, dim=2)
    if self.token_drop > 0:
      z = z[:, :, : -self.token_drop]
    return z

  def decode(self, z):
    if self.decoder_runner is None:
      raise RuntimeError("Decoder model is not loaded!")
    try:
      z = z * self.latents_std.to(z) + self.latents_mean.to(z)
      if z.shape[2] == 1:
        # 🌟 如果是单张图片 (T=1)，填充到 7 个 token 以满足 TRT 静态切片尺寸
        z_pad = z.repeat(1, 1, 7, 1, 1)
        return self._finalize_pixels(
            self.tiled_decode(z_pad)[:, :, -1:, :, :]
        )
      return self.decode_temporal(z)
    finally:
      pass

  def encode(self, x):
    if self.encoder_runner is None:
        raise RuntimeError("Encoder model is not loaded!")

    try:
        if x.ndim == 4:
            x = x.unsqueeze(2)

        if x.shape[2] == 1:
            # TRT encoder is compiled for exactly 17 input frames.
            # Repeat the single conditioning image to a full H3 VAE clip,
            # then keep only the final latent token.
            x_pad = x.repeat(1, 1, self.clip_length, 1, 1)

            moments = self.tiled_encode(
                self._normalize_pixels(x_pad)
            )[:, :, -1:, :, :]
        else:
            moments = self.encode_temporal(x)

        mean = torch.chunk(moments, 2, dim=1)[0]

        return (
            mean - self.latents_mean.to(mean)
        ) / self.latents_std.to(mean)

    finally:
        pass


# ================= 4. 原生 ModelPatcher 显存追踪器与 VAE 包装 =================


class TRTModelPatcher(comfy.model_patcher.ModelPatcher):
  
  def __init__(self, model, load_device, offload_device=torch.device("cpu"), size=0, weight_inplace_update=False,):
    super().__init__(
        model,
        load_device=load_device,
        offload_device=offload_device,
        size=size,
        weight_inplace_update=weight_inplace_update,
    )
    self._custom_model_size = size
    
  def model_size(self):
    if self._custom_model_size > 0:
      return self._custom_model_size
    return super().model_size()
  
  def patch_model(
      self,
      device_to=None,
      lowvram_model_memory=0,
      load_weights=True,
      force_patch_weights=False,
  ):
    if hasattr(self.model, "load_runners_to_gpu"):
      self.model.load_runners_to_gpu()
    return super().patch_model(
        device_to=device_to,
        lowvram_model_memory=lowvram_model_memory,
        load_weights=False,
        force_patch_weights=False,
    )
  
  def unpatch_model(self, device_to=None, unpatch_weights=True):
    if hasattr(self.model, "offload_runners_to_ram"):
      self.model.offload_runners_to_ram()
    return super().unpatch_model(device_to=device_to, unpatch_weights=False)


class ComfyTRTVAE(comfy.sd.VAE):
  
  def __init__(self, accelerated_vae: MiniMaxH3TRTVAE):
    self.first_stage_model = accelerated_vae
    self.vae_dtype = torch.float16
    self.device = (mm.get_torch_device() if hasattr(mm, "get_torch_device") else torch.device("cuda"))
    self.offload_device = (mm.unet_offload_device() if hasattr(mm, "unet_offload_device") else torch.device("cpu"))
    
    dec_size = 0
    if accelerated_vae.decoder_runner and os.path.exists(accelerated_vae.decoder_runner.model_path):
      dec_size = os.path.getsize(accelerated_vae.decoder_runner.model_path)
    
    enc_size = 0
    if accelerated_vae.encoder_runner and os.path.exists(accelerated_vae.encoder_runner.model_path):
      enc_size = os.path.getsize(accelerated_vae.encoder_runner.model_path)
    
    self.dec_size = max(dec_size, 1024 * 1024 * 1024 * 2)
    self.enc_size = max(enc_size, 1024 * 1024 * 300)
    
    self.patcher = TRTModelPatcher(
        self.first_stage_model,
        load_device=self.device,
        offload_device=self.offload_device,
        size=self.dec_size,
    )
    self.memory_used_decode = lambda shape, dtype: self.dec_size
    self.memory_used_encode = lambda shape, dtype: self.enc_size
    
  def decode(self, samples_in):
    z = samples_in["samples"] if isinstance(samples_in, dict) else samples_in
    if z.ndim == 4:
      z = z.unsqueeze(2)
      
    # 🌟 完全由 ComfyUI 进行显存登记与按需调度（KSampler 需要显存时 ComfyUI 会自动释放它）
    mm.load_models_gpu([self.patcher], memory_required=self.dec_size)
    video = self.first_stage_model.decode(z.half().to(self.device))
    b, c, t, h, w = video.shape
    return video.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).float().cpu()
  
  def encode(self, pixel_in):
    if pixel_in.ndim == 4:
      x = pixel_in.permute(3, 0, 1, 2).unsqueeze(0)
    elif pixel_in.ndim == 5:
      x = pixel_in.permute(0, 4, 1, 2, 3)
    else:
      raise ValueError(f"Unsupported pixel tensor shape: {pixel_in.shape}")
      
    mm.load_models_gpu([self.patcher], memory_required=self.enc_size)
    latents = self.first_stage_model.encode(x.half().to(self.device))
    return latents.float().cpu()


# ================= 5. ComfyUI 节点定义 =================


class MiniMaxH3TRTVAELoader:

  @classmethod
  def INPUT_TYPES(s):
    files = []
    for path in folder_paths.get_folder_paths("vae"):
      for root, _, fs in os.walk(path):
        for f in fs:
          if f.endswith(".engine"):
            files.append(os.path.relpath(os.path.join(root, f), path))
    files = sorted(list(dict.fromkeys(files)))
    options = ["None"] + files

    return {
        "required": {
            "decoder": (
                options,
                {
                    "default": "None",
                    "tooltip": 'Please compile the TensorRT engine using the "MiniMax-H3 TRT VAE Compiler" node before first use.',
                },
            ),
            "encoder": (
                options,
                {
                    "default": "None",
                    "tooltip": 'Please compile the TensorRT engine using the "MiniMax-H3 TRT VAE Compiler" node before first use.',
                },
            ),
        }
    }

  RETURN_TYPES = ("VAE",)
  RETURN_NAMES = ("VAE",)
  FUNCTION = "load_vae"
  CATEGORY = "MiniMax_H3/Acceleration"
  DESCRIPTION = 'Please compile the TensorRT engine using the "MiniMax-H3 TRT VAE Compiler" node before first use.'

  def load_vae(self, decoder, encoder):
    if encoder == "None":
      raise RuntimeError("Encoder cannot be None!")
    if decoder == "None":
      raise RuntimeError("Decoder cannot be None!")

    dec_path = folder_paths.get_full_path("vae", decoder)
    enc_path = folder_paths.get_full_path("vae", encoder)

    dec_runner = AutoEngineRunner(dec_path) if dec_path else None
    enc_runner = AutoEngineRunner(enc_path) if enc_path else None

    vae_instance = MiniMaxH3TRTVAE(decoder_runner=dec_runner, encoder_runner=enc_runner)
    return (ComfyTRTVAE(vae_instance),)


class MiniMaxH3TRTCompilerNode:

  @classmethod
  def INPUT_TYPES(s):
    files = []
    for path in folder_paths.get_folder_paths("vae"):
      for root, _, fs in os.walk(path):
        for f in fs:
          if f.endswith(".onnx"):
            files.append(os.path.relpath(os.path.join(root, f), path))
    files = sorted(list(dict.fromkeys(files)))
    options = ["None"] + files

    return {
        "required": {
            "decoder_onnx": (options,),
            "encoder_onnx": (options,),
            "delete_onnx_after_compile": ("BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Delete ONNX and .onnx.data files from disk after  compilation.",
                },
            ),
        },
        "hidden": {
            "unique_id": "UNIQUE_ID",
        },
    }

  OUTPUT_NODE = True
  RETURN_TYPES = ()
  FUNCTION = "compile_models"
  CATEGORY = "MiniMax_H3/Acceleration"

  @classmethod
  def _build_engine(cls, onnx_path, engine_path, is_decoder=True):
    if not HAS_TRT:
      raise RuntimeError("TensorRT library not found! Please install with pip first.")

    logger.info(f"Initializing TensorRT Builder for: {os.path.basename(onnx_path)}")
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()

    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
      flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
      network = builder.create_network(flags)
    else:
      network = builder.create_network()

    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(onnx_path):
      error_msgs = []
      for error in range(parser.num_errors):
        error_msgs.append(str(parser.get_error(error)))
      raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(error_msgs))

    if hasattr(trt.BuilderFlag, "FP16"):
      config.set_flag(trt.BuilderFlag.FP16)

    workspace_size = (4 if is_decoder else 8) * (1024**3)
    if hasattr(config, "set_memory_pool_limit"):
      config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    else:
      config.max_workspace_size = workspace_size

    profile = builder.create_optimization_profile()
    if is_decoder:
      input_name = "latent_tile"
      shape = (1, 24, 7, 16, 16)
    else:
      input_name = "pixel_tile"
      shape = (1, 3, 17, 256, 256)

    profile.set_shape(input_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    if hasattr(builder, "build_serialized_network"):
      serialized_engine = builder.build_serialized_network(network, config)
      if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine!")
      with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    else:
      engine = builder.build_engine(network, config)
      if engine is None:
        raise RuntimeError("Failed to build the TensorRT engine!")
      with open(engine_path, "wb") as f:
        f.write(engine.serialize())

  def compile_models(self, decoder_onnx, encoder_onnx, delete_onnx_after_compile, unique_id):
    dec_path = folder_paths.get_full_path("vae", decoder_onnx)
    enc_path = folder_paths.get_full_path("vae", encoder_onnx)

    if dec_path is None and enc_path is None:
      raise RuntimeError("No ONNX file selected!")

    mm.unload_all_models()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()

    if dec_path is not None:
      engine_path = os.path.splitext(dec_path)[0] + ".engine"
      logger.info(f"Building Decoder: {os.path.basename(dec_path)}...")
      self._build_engine(dec_path, engine_path, is_decoder=True)
      logger.info(f"Done: {os.path.basename(engine_path)}")
      if delete_onnx_after_compile:
        if os.path.exists(dec_path):
          os.remove(dec_path)
        data_path = dec_path + ".data"
        if os.path.exists(data_path):
          os.remove(data_path)

    if enc_path is not None:
      engine_path = os.path.splitext(enc_path)[0] + ".engine"
      logger.info(f"Building Encoder: {os.path.basename(enc_path)}...")
      self._build_engine(enc_path, engine_path, is_decoder=False)
      logger.info(f"Done: {os.path.basename(engine_path)}")
      if delete_onnx_after_compile:
        if os.path.exists(enc_path):
          os.remove(enc_path)
        data_path = enc_path + ".data"
        if os.path.exists(data_path):
          os.remove(data_path)

    logger.info('All Done! Please press "R" to refresh the model list.')
    PromptServer.instance.send_progress_text('Done! Press "R" to refresh the model list.', unique_id)
    return ()