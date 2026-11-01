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

    # 3. Perform inference
    with torch.no_grad():
        logits = model(dummy_frame)
        probabilities = F.softmax(logits, dim=-1)
        predicted_idx = torch.argmax(probabilities, dim=-1).item()

    # 4. Print results
    print("=== Simulated Chromagram Frame Inference ===")
    print(f"Input shape:          {dummy_frame.shape}")
    print(f"Simulated Chroma:     {dummy_frame.squeeze().tolist()}")
    print(f"Output logits shape:  {logits.shape}")
    print(f"Predicted class index: {predicted_idx}")
    print(f"Predicted chord:       {CHORD_CLASSES[predicted_idx]}")
    print(f"Confidence score:     {probabilities[0, predicted_idx].item():.4f}")


if __name__ == "__main__":
    test_dummy_inference()
