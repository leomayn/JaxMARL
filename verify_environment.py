import numpy as np
import jax
from jax.lib import xla_bridge
import os

def main():
    # Set TensorFlow logging level for more detailed output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    # Check NumPy version
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")

    # Check if JAX is using GPU
    try:
        gpu_available = jax.devices("gpu")
        if gpu_available:
            print("JAX is using the following GPU(s):")
            for gpu in gpu_available:
                print(f"  {gpu}")
        else:
            print("No GPU found. JAX is using CPU.")
    except RuntimeError as e:
        print(f"Error checking for GPUs: {e}")

    # Check CUDA version
    try:
        cuda_version = xla_bridge.get_backend().platform_version
        print(f"CUDA version: {cuda_version}")
    except Exception as e:
        print(f"Error checking CUDA version: {e}")

if __name__ == "__main__":
    main()
