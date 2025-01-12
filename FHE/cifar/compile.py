"""Load torch model, compiles it to FHE and exports it"""

import sys
import time
from pathlib import Path
import os

import torch
import torchvision
import torchvision.transforms as transforms
from concrete.fhe import Configuration, Exactness
from concrete.compiler import check_gpu_available
from models import cnv_2w2a, split_cnv_model

from concrete.ml.deployment import FHEModelDev
from concrete.ml.torch.compile import compile_brevitas_qat_model

def compile_submodels(submodels, example_input):
    compiled_submodels = []
    for i, submodel in enumerate(submodels):
        compiled_model = compile_brevitas_qat_model(
            submodel,
            example_input
        )
        compiled_submodels.append(compiled_model)
        print(f"Compiled submodel {i + 1}")
    return compiled_submodels

def main():
    # Load model
    #    model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    #    loaded = torch.load(Path(__file__).parent / "8_bit_model.pt")
    #    model.load_state_dict(loaded["model_state_dict"])

    # Instantiate the model
    model = cnv_2w2a(pre_trained=False)

    # Set the model to eval mode
    model.eval()

    # Load the saved parameters using the available checkpoint
    checkpoint = torch.load(
        Path(__file__).parent / "experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Split the model into convolutional and linear submodules
    conv_splits, linear_splits = split_cnv_model(model)

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

    num_samples = 10000
    train_sub_set = torch.stack(
        [train_set[index][0] for index in range(min(num_samples, len(train_set)))]
    )

    #compilation_onnx_path = "compilation_model.onnx"
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
    # Initialize input for the first convolutional submodel
    current_inputset = train_sub_set  # Original training dataset

    compiled_conv_submodels = []
    for i, conv_submodel in enumerate(conv_splits):
        # Set ONNX file name for the submodel
        conv_compilation_onnx_path = f"conv_submodel_{i + 1}.onnx"
        
        # Compile the current submodel
        quantized_conv_module = compile_brevitas_qat_model(
            torch_model=conv_submodel,
            torch_inputset=current_inputset,  # Input for the current submodel
            p_error=P_ERROR,
            configuration=configuration,
            output_onnx_file=conv_compilation_onnx_path,
            rounding_threshold_bits={"method": Exactness.APPROXIMATE, "n_bits": 6},
            n_bits={"model_inputs": 8, "model_outputs": 8},
        )
        compiled_conv_submodels.append(quantized_conv_module)
        print(f"Compiled convolutional submodel {i + 1}")
        
        # Update input for the next submodel
        # Use the compiled module to generate outputs
        current_inputset = [quantized_conv_module(x) for x in current_inputset]
    
    # Flatten the input for the linear submodels
    current_inputset = [torch.flatten(x, start_dim=1) for x in current_inputset]

    compiled_linear_submodels = []
    for i, linear_submodel in enumerate(linear_splits):
        # Set ONNX file name for the submodel
        linear_compilation_onnx_path = f"linear_submodel_{i + 1}.onnx"
        
        # Compile the current submodel
        quantized_linear_module = compile_brevitas_qat_model(
            torch_model=linear_submodel,
            torch_inputset=current_inputset,  # Flattened input for the current submodel
            p_error=P_ERROR,
            configuration=configuration,
            output_onnx_file=linear_compilation_onnx_path,
            rounding_threshold_bits={"method": Exactness.APPROXIMATE, "n_bits": 6},
            n_bits={"model_inputs": 8, "model_outputs": 8},
        )
        compiled_linear_submodels.append(quantized_linear_module)
        print(f"Compiled linear submodel {i + 1}")
        
        # Update input for the next submodel
        current_inputset = [quantized_linear_module(x) for x in current_inputset]

    # Compile the quantized model
    #quantized_numpy_module = compile_brevitas_qat_model(
    #    torch_model=model,
    #    torch_inputset=train_sub_set,
    #    p_error=P_ERROR,
    #    configuration=configuration,
    #    output_onnx_file=compilation_onnx_path,
    #    rounding_threshold_bits={"method": Exactness.APPROXIMATE, "n_bits": 6},
    #    n_bits={"model_inputs": 8, "model_outputs": 8},
    #)
    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Key generation
    print("Generating keys ...")
    start_keygen = time.time()
    #quantized_numpy_module.fhe_circuit.keygen()
    compiled_conv_circuits = []  # Store circuits for later use

    for i, conv_module in enumerate(compiled_conv_submodels):
        print(f"Generating FHE keys for convolutional submodel {i + 1}...")
        conv_module.fhe_circuit.keygen()  # Generate FHE keys
        compiled_conv_circuits.append(conv_module.fhe_circuit)
        print(f"FHE keys generated for convolutional submodel {i + 1}.")

        print("size_of_inputs", conv_module.fhe_circuit.size_of_inputs)
        print("bootstrap_keys", conv_module.fhe_circuit.size_of_bootstrap_keys)
        print("keyswitches",    conv_module.fhe_circuit.size_of_keyswitch_keys)

    compiled_linear_circuits = []  # Store circuits for later use

    for i, linear_module in enumerate(compiled_linear_submodels):
        print(f"Generating FHE keys for linear submodel {i + 1}...")
        linear_module.fhe_circuit.keygen()  # Generate FHE keys
        compiled_linear_circuits.append(linear_module.fhe_circuit)
        print(f"FHE keys generated for linear submodel {i + 1}.")

        print("size_of_inputs", linear_module.fhe_circuit.size_of_inputs)
        print("bootstrap_keys", linear_module.fhe_circuit.size_of_bootstrap_keys)
        print("keyswitches",    linear_module.fhe_circuit.size_of_keyswitch_keys)

    end_keygen = time.time()
    print(f"Keygen finished in {end_keygen - start_keygen:.2f} seconds")

    for i, conv_module in enumerate(compiled_conv_submodels):
        dev = FHEModelDev(path_dir=f"./devc_{i}", model=conv_module)
        dev.save()

    for i, linear_module in enumerate(compiled_linear_submodels):
        dev = FHEModelDev(path_dir=f"./devl_{i}", model=linear_module)
        dev.save()

    #dev = FHEModelDev(path_dir="./dev", model=quantized_numpy_module)
    #dev.save()


if __name__ == "__main__":
    main()