import numpy as np
import soundfile as sf

SR = 22050  # same sample rate used in extract_chroma.py

# Frequencies (Hz) of the notes used in the chords (octave 4, A4 = 440 Hz)
NOTES = {
    "C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23,
    "G": 392.00, "A": 440.00, "B": 493.88,
    "C5": 523.25,
}

CHORDS = {
    "C":  ["C", "E", "G"],
    "G":  ["G", "B", "D"],
    "Am": ["A", "C5", "E"],
    "F":  ["F", "A", "C5"],
}


def synth_chord(notes, duration, sr=SR):
    """Synthesize a chord by summing sine waves with a few harmonics and an envelope."""
    t = np.arange(int(duration * sr)) / sr
    signal = np.zeros_like(t)
    for name in notes:
        f0 = NOTES[name]
        # Fundamental + 2 harmonics with decreasing amplitude (more natural timbre)
        for k, amp in [(1, 1.0), (2, 0.5), (3, 0.25)]:
            signal += amp * np.sin(2 * np.pi * k * f0 * t)

    # Envelope: short attack and exponential decay (avoids clicks)
    attack = int(0.01 * sr)
    env = np.exp(-1.5 * t)
    env[:attack] *= np.linspace(0, 1, attack)
    signal *= env

    return signal / (np.max(np.abs(signal)) + 1e-8) * 0.8


def main():
    # Single C major chord
    sf.write("chord_test.wav", synth_chord(CHORDS["C"], 2.0), SR)
    print("Generated: chord_test.wav (C major)")

    # Single A minor chord
    sf.write("chord_am_test.wav", synth_chord(CHORDS["Am"], 2.0), SR)
    print("Generated: chord_am_test.wav (A minor)")

    # C - G - Am - F progression, 1 second per chord
    progression = np.concatenate(
        [synth_chord(CHORDS[c], 1.0) for c in ["C", "G", "Am", "F"]]
    )
    sf.write("progression_test.wav", progression, SR)
    print("Generated: progression_test.wav (C - G - Am - F)")


if __name__ == "__main__":
    main()
