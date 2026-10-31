import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
  
def extract_and_plot_chroma(audio_path, output_image_path="chromagram.png"):
    # 1. Carrega o áudio em mono e taxa de amostragem de 22050 Hz
    print(f"Carregando áudio: {audio_path}...")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
  
    # 2. Executa a Constant-Q Transform (CQT)                                                                      
    # np.abs() obtém o espectro de magnitude da CQT
    cqt_spec = np.abs(librosa.cqt(y=y, sr=sr))
  
    # 3. Extrai o Cromagrama a partir do CQT calculado
    # Como librosa.cqt usa bins_per_octave=12 por padrão (e chroma_cqt assume 36),
    # precisamos especificar bins_per_octave=12 para alinhar os bins às notas corretas.
    chroma = librosa.feature.chroma_cqt(C=cqt_spec, sr=sr, bins_per_octave=12)
  
    # 4. Normaliza a matriz do Cromagrama entre 0 e 1 (Min-Max Scaling)
    chroma_min = np.min(chroma)
    chroma_max = np.max(chroma)
    chroma_norm = (chroma - chroma_min) / (chroma_max - chroma_min + 1e-8)
  
    # 5. Plota o Cromagrama e salva a imagem
    plt.figure(figsize=(10, 5))
    img = librosa.display.specshow(
        chroma_norm,
        y_axis="chroma",                                                                                           
        x_axis="time",
        sr=sr,
        cmap="coolwarm"
    )
    plt.colorbar(img, label="Intensidade Normalizada (0 a 1)")
    plt.title(f"Chromagram (CQT) - {audio_path}")
    plt.tight_layout()
  
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Cromagrama salvo com sucesso em: {output_image_path}") 
  
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
            print(f"Arquivo não encontrado (rode generate_test_audios.py primeiro): {audio_file}")