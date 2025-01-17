"""Load torch model, compiles it to FHE and exports it"""

import math
import sys
import time
from pathlib import Path
import os

import torch
import torchvision
import torchvision.transforms as transforms
from concrete.fhe import Configuration, Exactness
from concrete.compiler import check_gpu_available
from models import synthetic_cnv_2w2a
import numpy as np
from brevitas.nn import QuantConv2d
from torch.nn import BatchNorm2d

from concrete.ml.deployment import FHEModelDev
from concrete.ml.torch.compile import compile_brevitas_qat_model


def main():
    # Load model
    #    model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    #    loaded = torch.load(Path(__file__).parent / "8_bit_model.pt")
    #    model.load_state_dict(loaded["model_state_dict"])

    # Instantiate the model
    model = synthetic_cnv_2w2a(pre_trained=False)
    
    # Set the model to eval mode
    model.eval()

    checkpoint_path = Path(__file__).parent / "experiments/synthetic_model_checkpoint.pth"

    if not checkpoint_path.exists():
        torch.manual_seed(42)  # For reproducibility
        for layer in model.features:
            if isinstance(layer, QuantConv2d):
                torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                print(f"QuantConv2d weights initialized: {layer.weight}")
            elif isinstance(layer, BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1.0)
                torch.nn.init.constant_(layer.bias, 0.0)
                print(f"BatchNorm2d weights initialized: {layer.weight}")

        # Save the model state to a checkpoint
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        return

    # Load the saved parameters using the available checkpoint
    checkpoint = torch.load(
        #  Path(__file__).parent / "experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar",
        checkpoint_path,
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    dummy_input = torch.randn(1, 3, 32, 32)
    
    with torch.no_grad():
        output = model(dummy_input)
        assert dummy_input.shape == output.shape
        print(output)
    
    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load data
    try:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=False,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    except:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=True,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )

    num_samples = 500
    train_sub_set = torch.stack(
        [train_set[index][0] for index in range(min(num_samples, len(train_set)))]
    )

    compilation_onnx_path = "compilation_synthetic_model.onnx"
    print("Compiling the model ...")
    start_compile = time.time()

    # Configuration constants
    CURRENT_DIR = Path(__file__).resolve().parent
    KEYGEN_CACHE_DIR = CURRENT_DIR.joinpath(".keycache")
    # Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPILATION_DEVICE = "cuda" if check_gpu_available() else "cpu"
    P_ERROR = float(os.environ.get("P_ERROR", 0.01))

    # Configuration setup
    configuration = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location=KEYGEN_CACHE_DIR,
    )

    # Compile the quantized model
    quantized_numpy_module = compile_brevitas_qat_model(
        torch_model=model,
        torch_inputset=train_sub_set,
        p_error=P_ERROR,
        configuration=configuration,
        output_onnx_file=compilation_onnx_path,
        rounding_threshold_bits={"method": Exactness.APPROXIMATE, "n_bits": 6},
        n_bits={"model_inputs": 8, "model_outputs": 8},
    )
    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Key generation
    print("Generating keys ...")
    start_keygen = time.time()
    quantized_numpy_module.fhe_circuit.keygen()

    end_keygen = time.time()
    print(f"Keygen finished in {end_keygen - start_keygen:.2f} seconds")

    print("size_of_inputs", quantized_numpy_module.fhe_circuit.size_of_inputs)
    print("bootstrap_keys", quantized_numpy_module.fhe_circuit.size_of_bootstrap_keys)
    print("keyswitches", quantized_numpy_module.fhe_circuit.size_of_keyswitch_keys)

    dev = FHEModelDev(path_dir="./dev", model=quantized_numpy_module)
    dev.save()


if __name__ == "__main__":
    main()