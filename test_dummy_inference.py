import torch
import torch.nn.functional as F
from model import DummyChordModel

CHORD_CLASSES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
]


def test_dummy_inference():
    # 1. Instantiate the model
    model = DummyChordModel(input_size=12, output_size=24)
    model.eval()

    # 2. Generate random tensor simulating one Chromagram frame (12 pitch classes)
    dummy_frame = torch.randn(1, 12)

    # 3. Perform forward pass
    with torch.no_grad():
        logits = model(dummy_frame)
        probabilities = F.softmax(logits, dim=-1)
        predicted_idx = torch.argmax(probabilities, dim=-1).item()

    # 4. Data flow validations
    assert logits.shape == (1, 24), f"Shape mismatch: {logits.shape} != (1, 24)"
    assert not torch.isnan(logits).any(), "Forward pass failed: logits contain NaN values"
    assert not torch.isinf(logits).any(), "Forward pass failed: logits contain Inf values"
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0)), "Probabilities do not sum to 1"

    # Multi-frame batch validation (e.g., sequence of 10 chromagram frames)
    batch_frames = torch.randn(10, 12)
    batch_logits = model(batch_frames)
    assert batch_logits.shape == (10, 24), f"Batch shape mismatch: {batch_logits.shape} != (10, 24)"

    # 5. Print results
    print("=== Simulated Chromagram Frame Inference ===")
    print(f"Input shape:           {dummy_frame.shape}")
    print(f"Simulated Chroma:      {dummy_frame.squeeze().tolist()}")
    print(f"Output logits shape:   {logits.shape}")
    print(f"Predicted class index: {predicted_idx}")
    print(f"Predicted chord:       {CHORD_CLASSES[predicted_idx]}")
    print(f"Confidence score:      {probabilities[0, predicted_idx].item():.4f}")
    print("\nData flow validation: ALL CHECKS PASSED")


if __name__ == "__main__":
    test_dummy_inference()
