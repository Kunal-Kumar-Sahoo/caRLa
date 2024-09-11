import torch
import sys

stdin = sys.stdin
stderr = sys.stderr

# Get list of all available GPUs
gpu_count = torch.cuda.device_count()

print('\n\n\nList of found GPUs:')
for i in range(gpu_count):
    # Get properties of each GPU
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")

if gpu_count == 0:
    print("No GPUs found.")

sys.stdin = stdin
sys.stderr = stderr