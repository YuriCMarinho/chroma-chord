import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
  
def extract_and_plot_chroma(audio_path, output_image_path="chromagram.png"):
    # 1. Load the audio as mono at a 22050 Hz sample rate
    print(f"Loading audio: {audio_path}...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
  
    # 2. Compute the Constant-Q Transform (CQT)                                                                      
    # np.abs() gets the magnitude spectrum of the CQT
    cqt_spec = np.abs(librosa.cqt(y=y, sr=sr))
  
    # 3. Extract the chromagram from the computed CQT
    # Since librosa.cqt uses bins_per_octave=12 by default (and chroma_cqt assumes 36),
    # we must pass bins_per_octave=12 so the bins line up with the correct notes.
    chroma = librosa.feature.chroma_cqt(C=cqt_spec, sr=sr, bins_per_octave=12)
  
    # 4. Normalize the chromagram matrix to the 0-1 range (min-max scaling)
    chroma_min = np.min(chroma)
    chroma_max = np.max(chroma)
    chroma_norm = (chroma - chroma_min) / (chroma_max - chroma_min + 1e-8)
  
    # 5. Plot the chromagram and save the image
    plt.figure(figsize=(10, 5))
    img = librosa.display.specshow(
        chroma_norm,
        y_axis="chroma",                                                                                           
        x_axis="time",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar(img, label="Normalized Intensity (0 to 1)")
    plt.title(f"Chromagram (CQT) - {audio_path}")
    plt.tight_layout()
  
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Chromagram saved successfully to: {output_image_path}") 
  
if __name__ == "__main__":
    import os

    test_files = [
        ("chord_test.wav", "chromagram.png"),
        ("chord_am_test.wav", "chromagram_am.png"),
        ("progression_test.wav", "chromagram_progression.png"),
    ]

    for audio_file, output_img in test_files:
        if os.path.exists(audio_file):
            extract_and_plot_chroma(audio_file, output_image_path=output_img)
        else:
            print(f"File not found (run generate_test_audios.py first): {audio_file}")