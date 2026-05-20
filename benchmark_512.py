import time
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

print("Loading model (SD-turbo)...")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

stream = StreamDiffusion(
    pipe,
    t_index_list=[0],
    torch_dtype=torch.float16,
    width=512,
    height=512,
)

stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
    device=pipe.device, dtype=pipe.dtype
)

prompt = "a beautiful landscape, mountains, sunset, photorealistic"
stream.prepare(prompt)

print("Warming up...")
for _ in range(5):
    stream.txt2img()

torch.cuda.synchronize()

print("Benchmarking 100 frames at 512x512...")
start = time.perf_counter()
num_frames = 100
for _ in range(num_frames):
    x_output = stream.txt2img()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

fps = num_frames / elapsed
print(f"\nResults:")
print(f"  Resolution: 512x512")
print(f"  Frames: {num_frames}")
print(f"  Time: {elapsed:.2f}s")
print(f"  FPS: {fps:.1f}")
print(f"  Latency: {1000/fps:.1f}ms per frame")
print(f"  VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

img = postprocess_image(x_output, output_type="pil")[0]
img.save("/home/nvidia/projects/StreamDiffusion/benchmark_sample.png")
print(f"\nSample image saved to benchmark_sample.png")
