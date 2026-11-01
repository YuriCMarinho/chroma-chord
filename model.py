import torch
import torch.nn as nn


class DummyChordModel(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) for chord classification from chromagram representations.

    Input:
        Chroma pitch class vector of dimension `input_size` (default: 12, corresponding to
        the 12 semitones: C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Shape: [batch_size, 12]
        or [12].

    Output:
        Logits across `output_size` chord classes (default: 24, typically representing
        12 major and 12 minor chords). Shape: [batch_size, 24] or [24].
    """

    def __init__(self, input_size: int = 12, hidden_size: int = 64, output_size: int = 24):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = DummyChordModel(input_size=12, output_size=24)
    model.eval()

    # Test with dummy chroma input (batch_size=1, input_size=12)
    dummy_input = torch.randn(1, 12)
    with torch.no_grad():
        output = model(dummy_input)

    print("DummyChordModel created successfully!")
    print(f"Model architecture:\n{model}")
    print(f"Dummy input shape:  {dummy_input.shape}")
    print(f"Output shape:       {output.shape} (Expected: [1, 24])")
