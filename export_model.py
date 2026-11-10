import os
import torch
from model import DummyChordModel


def export_torchscript_model(output_path: str = "baseline_model.pt"):
    # 1. Instantiate the model and set to evaluation mode
    model = DummyChordModel(input_size=12, hidden_size=64, output_size=24)
    model.eval()

    # 2. Dummy input for tracing (1 frame with 12 chromagram bins)
    dummy_input = torch.randn(1, 12)

    # 3. Trace the model using torch.jit.trace
    print(f"Tracing DummyChordModel with dummy input shape: {dummy_input.shape}...")
    traced_model = torch.jit.trace(model, dummy_input)

    # 4. Save the traced model for C++ / LibTorch consumption
    traced_model.save(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"Model successfully exported to: {output_path} ({file_size_kb:.2f} KB)")

    # 5. Verify the exported model by reloading and comparing outputs
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()

    with torch.no_grad():
        original_output = model(dummy_input)
        loaded_output = loaded_model(dummy_input)

    assert torch.allclose(original_output, loaded_output, atol=1e-6), (
        "Output mismatch between original PyTorch model and exported TorchScript model!"
    )
    print("Verification: Exported model outputs match original PyTorch model exactly!")


if __name__ == "__main__":
    export_torchscript_model()
