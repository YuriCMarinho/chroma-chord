# chroma-chord

Offline audio chord recognition pipeline built in C++20 and PyTorch (LibTorch), designed as the foundation for pure C++ Constant-Q Transform (CQT) chromagram extraction and neural chord classification.

> [!NOTE]
> **Execution Scope — Offline CLI Tool (Non-Real-Time):**
> `chroma-chord` is explicitly designed as a **Command Line Interface (CLI) tool focused on offline audio file processing rather than real-time execution**. It does not capture live microphone streams, interface with real-time audio drivers (e.g., ALSA, JACK, CoreAudio, ASIO), or operate under real-time callback latency constraints. Instead, it loads a complete `.wav` file from disk into memory, segments the signal deterministically, and outputs a full chronological chord transcription to the terminal.

---

## Offline Architecture & Logical Data Flow

The offline C++ inference pipeline processes `.wav` audio files sequentially from disk to terminal predictions:

```mermaid
flowchart LR
    WAV[".wav (Disk)"] --> MEM["Memory (Mono PCM)"]
    MEM --> SEG["Temporal Segmentation"]
    SEG --> CQT["Feature Extraction (Simulated Chroma / Future C++ CQT)"]
    CQT --> TORCH["LibTorch Inference (TorchScript)"]
    TORCH --> TERM["Terminal Output"]
```

```text
.Wav (Disk) -> Memory -> Temporal Segmentation -> LibTorch -> Terminal
```

### Pipeline Stages

1. **`.Wav (Disk)`**
   - Reads uncompressed `.wav` audio files from disk via the header-only `AudioFile<double>` decoder (`include/AudioFile.h`).
2. **`Memory` (Mono Downmixing & Buffering)**
   - Extracts audio metadata (sample rate, channel count, total samples per channel).
   - Downmixes multi-channel audio into a contiguous mono signal stored in memory (`std::vector<double>`) by averaging samples across all channels.
3. **`Temporal Segmentation`**
   - Slices the in-memory mono buffer into fixed-duration analysis windows using a sliding window (`kWindowDurationSec = 0.5s`, `kHopDurationSec = 0.5s`).
   - Converts window and hop durations into sample counts based on the input file's native sample rate (`windowSize = kWindowDurationSec * sampleRate`, `hopLength = kHopDurationSec * sampleRate`).
   - Clamps reads at the end of the buffer and zero-pads partial trailing windows so every frame maintains a uniform length.
   - *Groundwork for C++ CQT*: Currently extracts a 12-element placeholder vector per window (`std::vector<float>(12)`) as the interface boundary where pure C++ CQT chromagram extraction will be plugged in.
4. **`LibTorch` (TorchScript Model Inference)**
   - Loads the traced TorchScript model (`baseline_model.pt`) via `torch::jit::load` in evaluation mode (`model.eval()`).
   - Wraps each 12-dimensional pitch-class frame into a `[1, 12]` `torch::kFloat32` tensor (`torch::from_blob(...).clone()`).
   - Executes the forward pass through the MLP classifier (`12 -> 64 -> 32 -> 24` logits across 12 major and 12 minor chord classes) and selects the highest-scoring chord class via `argmax(1)`.
5. **`Terminal`**
   - Streams the file metadata (sample rate, total samples, window size, hop length) followed by timestamped per-window chord index predictions (`Window at <t> s -> Chord Index: <idx>`) to standard output.

---

## Repository Structure

| File / Directory | Description |
| :--- | :--- |
| `src/main.cpp` | Main C++ offline pipeline (`.wav` loading, mono conversion, temporal segmentation, LibTorch inference, CLI output). |
| `include/AudioFile.h` | Header-only C++ audio file reader/writer used for `.wav` decoding. |
| `model.py` | PyTorch `DummyChordModel` MLP definition (`12` chroma inputs $\rightarrow$ `24` chord class logits). |
| `export_model.py` | Traces and exports `DummyChordModel` to TorchScript (`baseline_model.pt`) for C++ consumption. |
| `test_dummy_inference.py` | Validates model forward pass shapes, probability distributions, and batch inference in Python. |
| `extract_chroma.py` | Reference Python CQT & chromagram extraction pipeline using `librosa`. |
| `generate_test_audios.py` | Synthesizes test `.wav` files (`chord_test.wav`, `chord_am_test.wav`, `progression_test.wav`). |
| `CMakeLists.txt` / `Makefile` | Build configurations for compiling `chroma-chord` with C++20 and linking against LibTorch. |

---

## Functional Requirements

The software operates as an offline command-line chord transcription tool: **it receives an audio file path via command-line argument and outputs its chronological chord transcription to the terminal.**

| ID | Requirement | Description |
| :--- | :--- | :--- |
| **FR-01** | **CLI File Input** | Accepts the target `.wav` audio file path as a positional command-line argument (`chroma-chord <path-to-wav-file>`) and validates argument presence before execution. |
| **FR-02** | **Offline `.wav` Decoding & Mono Normalization** | Loads the uncompressed `.wav` file from disk into memory, extracts stream metadata (sample rate, channel count, total samples), and downmixes multi-channel audio into a single contiguous mono signal. |
| **FR-03** | **Fixed-Interval Temporal Segmentation** | Segments the mono signal chronologically using a sliding analysis window (`0.5s` window duration, `0.5s` hop step) scaled to the file's sample rate, zero-padding any partial trailing window. |
| **FR-04** | **Per-Window Feature & Model Inference** | Extracts a 12-element pitch-class feature vector (`[1, 12]` tensor) for each temporal window and evaluates it through the pre-loaded TorchScript classifier (`baseline_model.pt`) to determine the most probable chord class (`argmax`). |
| **FR-05** | **Chronological Transcription Output** | Prints the audio metadata followed by a time-ordered sequence of chord predictions (`Window at <timestamp> s -> Chord Index: <class_index>`) to standard output (`stdout`). |

---

## System & Build Requirements

### C++ Offline Pipeline
- **Compiler:** C++20 compatible compiler (`g++` or `clang++`)
- **Build System:** `make` or `cmake` ($\ge 3.20$)
- **LibTorch:** PyTorch C++ library (either standalone LibTorch or via a local Python `torch` installation)

### Python Model & Prototyping Scripts
- **Python:** $\ge 3.10$
- **Packages:** `torch`, `numpy`, `librosa`, `soundfile`, `matplotlib`

---

## Usage

### 1. Export the TorchScript Model (Optional if `baseline_model.pt` is present)
```bash
python export_model.py
```

### 2. Build the C++ Engine (CMake & LibTorch Linkage)

The C++ inference engine is configured in `CMakeLists.txt` to compile with **C++20** (`CMAKE_CXX_STANDARD 20`) and link against **LibTorch** (`libtorch`, `libtorch_cpu`, `libc10`).

#### How CMake Resolves and Links LibTorch
1. **Package Discovery (`find_package(Torch REQUIRED)`):** CMake searches `CMAKE_PREFIX_PATH` for `TorchConfig.cmake` to locate the LibTorch headers and shared libraries.
2. **Compiler Flags (`${TORCH_CXX_FLAGS}`):** Appends the required ABI and C++ compilation flags defined by the target LibTorch distribution to `CMAKE_CXX_FLAGS`.
3. **Target Linkage (`target_link_libraries(chroma-chord PRIVATE "${TORCH_LIBRARIES}")`):** Links the `chroma-chord` executable against the LibTorch core libraries (`torch`, `torch_cpu`, `c10`) and configures the runtime library search path (`RPATH`) so the binary can resolve `.so`/`.dylib`/`.dll` dependencies at launch without manual `LD_LIBRARY_PATH` exports.

#### Option A: Linking Against a Python Virtual Environment's PyTorch (Recommended for Dev)
If PyTorch is already installed in your Python environment (e.g., `.venv`), you can query `torch.utils.cmake_prefix_path` directly to pass its embedded LibTorch path to CMake:

```bash
# 1. Configure the CMake build directory using PyTorch's internal CMake prefix
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# 2. Compile the chroma-chord engine
cmake --build build --config Release -j$(nproc)
```

#### Option B: Linking Against a Standalone LibTorch Distribution
If compiling without a Python environment, download the standalone **LibTorch** archive (cxx11 ABI) from [pytorch.org](https://pytorch.org/get-started/locally/), extract it, and pass its root directory to `CMAKE_PREFIX_PATH`:

```bash
# 1. Configure CMake pointing to the extracted libtorch directory
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch

# 2. Compile the chroma-chord engine
cmake --build build --config Release -j$(nproc)
```

#### Alternative: Building with Make
A standalone `Makefile` is also provided for direct `g++` compilation and manual `-ltorch -ltorch_cpu -lc10` linkage:
```bash
make TORCH_DIR=/path/to/libtorch_or_site_packages/torch
```

#### Cross-Compilation Guidelines for Dependencies

When cross-compiling `chroma-chord` for a different target OS or CPU architecture (e.g., building on an `x86_64` host for `aarch64` / ARM64 Linux, macOS Apple Silicon, or Windows MSVC), each project dependency is handled as follows:

| Dependency | Type | Cross-Compilation Requirement |
| :--- | :--- | :--- |
| **`AudioFile.h`** (`include/AudioFile.h`) | Header-only (Standard C++) | **Zero external linkage.** Uses only standard C++ STL containers and streams. Compiles out-of-the-box with any C++20 cross-compiler (`aarch64-linux-gnu-g++`, `clang++ --target=...`, `cl.exe`). |
| **`LibTorch`** (`libtorch`, `libtorch_cpu`, `libc10`) | Precompiled Shared/Static Libraries | **Target-matched binary distribution required.** You must point `CMAKE_PREFIX_PATH` and `CMAKE_FIND_ROOT_PATH` to a LibTorch build compiled for the **target** architecture and matching C++ standard library ABI (`cxx11 ABI`). |
| **`baseline_model.pt`** | Serialized TorchScript Model | **Architecture-independent.** The `.pt` model artifact exported via `python export_model.py` on the host machine is portable across CPU architectures and operating systems without re-exporting on the target device. |

##### Step-by-Step Cross-Compilation with CMake (Example: `x86_64` Host $\rightarrow$ `aarch64` Target)

1. **Obtain Target-Architecture LibTorch:**
   Download or build the target `libtorch` package (e.g., `libtorch-linux-aarch64` with `cxx11` ABI) and extract it on the host machine (e.g., `/opt/libtorch-aarch64`).
2. **Configure CMake with Target Toolchain & Root Path:**
   Pass the cross-compiler toolchain and point both `CMAKE_PREFIX_PATH` and `CMAKE_FIND_ROOT_PATH` to the target `libtorch` directory so `find_package(Torch REQUIRED)` resolves the target headers and `.so`/`.lib` binaries rather than host libraries:
   ```bash
   cmake -S . -B build-aarch64 \
     -DCMAKE_SYSTEM_NAME=Linux \
     -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
     -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
     -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
     -DCMAKE_PREFIX_PATH=/opt/libtorch-aarch64 \
     -DCMAKE_FIND_ROOT_PATH=/opt/libtorch-aarch64 \
     -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
     -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
     -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
     -DCMAKE_BUILD_TYPE=Release
   ```
3. **Build the Target Binary:**
   ```bash
   cmake --build build-aarch64 --config Release -j$(nproc)
   ```
4. **Deploy to Target:**
   Copy the compiled `build-aarch64/chroma-chord` executable, the target LibTorch shared libraries (`libtorch.so`, `libtorch_cpu.so`, `libc10.so`), and `baseline_model.pt` to the target environment.

> [!TIP]
> **Cross-Platform Compiler Flags (`CMakeLists.txt`):** The build script automatically detects the target compiler toolchain—applying `/W4 /permissive-` when targeting Windows `MSVC` and `-Wall -Wextra -Wpedantic` when targeting GCC/Clang on Linux or macOS.

### 3. Run Offline Inference

> [!IMPORTANT]
> Run the binary from the repository root directory so `torch::jit::load("baseline_model.pt")` can locate the exported TorchScript model artifact.

**When built with CMake:**
```bash
./build/chroma-chord chord_test.wav
```

**When built with Make:**
```bash
./bin/chroma-chord chord_test.wav
```

Example terminal output:
```text
Model baseline_model.pt loaded successfully.
Sample Rate:   22050 Hz
Total Samples: 44100
Window Size:   11025 samples (0.5 s)
Hop Length:    11025 samples (0.5 s)
Window at 0.00 s -> Chord Index: 12
Window at 0.50 s -> Chord Index: 12
Window at 1.00 s -> Chord Index: 12
Window at 1.50 s -> Chord Index: 12
```

---

## Technical Debt & Future Roadmap: Porting CQT from Python (Issue #2) to C++

> [!WARNING]
> **Current Technical Debt (`src/main.cpp`):** During the Temporal Segmentation loop, the C++ engine currently populates `chromaFrame` (`std::vector<float>(12)`) by copying the first 12 raw time-domain PCM samples of each window (`window[0..11]`). This acts purely as a structural stub to validate `[1, 12]` tensor creation and LibTorch inference end-to-end, and does not yet represent real frequency-domain pitch classes.

### Reference Algorithm from Issue #2 (`extract_chroma.py`)
In Issue #2, the Constant-Q Transform (CQT) and chromagram extraction pipeline was prototyped and validated in Python using `librosa`:
1. **Constant-Q Transform (`librosa.cqt`):** Computes the magnitude spectrum (`|CQT|`) with `bins_per_octave = 36` (3 bins per semitone) across `7` octaves (`n_bins = 252`), providing narrow logarithmically-spaced filters that prevent spectral leakage into neighboring semitones.
2. **Chroma Folding (`librosa.feature.chroma_cqt`):** Maps and aggregates the 252 CQT frequency bins across octaves into the 12 semitone pitch classes (`C, C#, D, D#, E, F, F#, G, G#, A, A#, B`).
3. **Min-Max Normalization:** Scales the resulting 12-bin chromagram into the `[0, 1]` range via `(chroma - min) / (max - min + 1e-8)`.

### C++ Porting Plan
Because the offline architecture isolates temporal windowing from neural inference, porting the Issue #2 CQT algorithm to the C++ ecosystem is scoped to a drop-in replacement between **Temporal Segmentation** and **LibTorch Inference**:

| Stage | Current State (Technical Debt) | Target State (Pure C++ CQT Port) |
| :--- | :--- | :--- |
| **Input** | `std::vector<double> window` (`0.5s` mono PCM) | `std::vector<double> window` (`0.5s` mono PCM) |
| **Transform** | Raw slice of `window[0..11]` cast to `float` | Native C++ CQT (`36` bins/octave, `252` total bins) + magnitude calculation |
| **Pitch Folding** | None | Octave folding from `252` CQT bins into `12` semitone pitch-class energies |
| **Normalization** | Unnormalized PCM amplitudes | Min-max / $L_p$ normalization to `[0.0f, 1.0f]` |
| **Output to LibTorch** | `torch::from_blob(chromaFrame.data(), {1, 12}, torch::kFloat32)` | `torch::from_blob(chromaFrame.data(), {1, 12}, torch::kFloat32)` |

Once ported, the Python dependency (`librosa`) will no longer be required for feature extraction, completing a self-contained C++20 + LibTorch inference engine.
