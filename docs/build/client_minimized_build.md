# Client Compact Minimized Build Profile

This guide describes how to configure, build, and integrate the **`client-compact-minimized-build`** profile for `speech.cpp`.

This build profile is specifically optimized for client-side and desktop GUI applications that embed speech processing (STT and TTS) without requiring the HTTP server, WebUI frontend assets, or hundreds of megabytes of unused model architectures.

---

## 1. Key Characteristics

| Aspect | Full / Default Build | Client Compact Minimized Build |
|---|---|---|
| **Supported Models** | 45+ model families (TTS, ASR, Diarization, Codecs, Separation, etc.) | **Curated core set**: Supertonic 3 (TTS), Nemotron ASR, Granite Speech 2B (`granite5asr`), Qwen3 ASR (`qwen3_asr` + `qwen3_forced_aligner`), Parakeet TDT (`parakeet_tdt`), plus built-in Silero & MarbleNet VAD. |
| **HTTP Web Server** | Included (`audiocpp_server`) | **Excluded** (`AUDIOCPP_BUILD_SERVER=OFF`) |
| **WebUI Assets** | Multi-megabyte embedded HTML/JS distribution and demo voice audio WAVs | **Excluded** (not compiled or embedded into binaries) |
| **Build Time** | Several minutes across dozens of model translation units | **Sub-minute to ~1 minute** clean compilation |
| **Dead-Code Elimination** | Standard Release optimizations | **Aggressive function-level linking and COMDAT folding** (`/Gy`, `/Gw`, `/GF`, `/OPT:REF`, `/OPT:ICF` on MSVC; `-ffunction-sections`, `-fdata-sections`, `-Wl,--gc-sections` on GCC/Clang) |
| **Shipped Artifacts** | Full runtime, server, CLI, unittests, model manager | `audiocpp.dll` / `libaudiocpp.so` (C ABI, CMake target `speech`), `audiocpp_cli` (`speech_cli`), `audiocpp_gguf` (`speech_gguf`) |

---

## 2. Curated Models Included

The preset comes pre-configured with a high-performance STT + TTS toolkit:

1. **TTS (Text-to-Speech)**:
   - **`supertonic`**: Supertonic 3 multilingual preset-voice TTS. Extremely fast (up to 200x+ RTF on CUDA, 6x+ on CPU).
2. **ASR (Speech-to-Text)**:
   - **`nemotron_asr`**: Fast and accurate streaming/offline ASR.
   - **`granite5asr`**: IBM Granite Speech 4.1 / 2B ASR model (`granite_speech`).
   - **`qwen3_asr`**: Qwen3 ASR 1.7B multilingual ASR with speculative decoding and automatic forced alignment (`qwen3_forced_aligner`).
   - **`parakeet_tdt`**: NeMo Parakeet TDT 0.6B v0.3 Fast ASR.
3. **Voice Activity Detection (VAD)**:
   - **`silero_vad`** & **`marblenet_vad`**: Ultra-low-latency VAD for stream chunking and boundary detection.

All other 40+ optional architectures (e.g. ChatTTS, CosyVoice, Fish Audio, F5-TTS, MeloTTS, Sense ASR, DramaBox, HiggAudio, etc.) are excluded from compilation and linking.

---

## 3. Quick Start: Building with CMake Presets

Ensure your build environment has CMake (>= 3.21) and Ninja installed. On Windows, run from a Visual Studio x64 Developer command prompt (`vcvars64` or `vcvarsall.bat x64`).

### CPU Client Build
```powershell
# Configure using preset
cmake --preset client-compact-minimized-cpu

# Build all client artifacts
cmake --build --preset client-compact-minimized-cpu
```

### CUDA Accelerated Client Build
```powershell
# Configure with CUDA GPU acceleration enabled
cmake --preset client-compact-minimized-cuda

# Build all client artifacts
cmake --build --preset client-compact-minimized-cuda
```

### Build Artifacts Produced
After building, the following binaries are generated in the build directory (`build-client-compact-minimized-cpu/bin/`):
- `audiocpp.dll` (Windows) or `libaudiocpp.so` (Linux): The core C ABI shared library (exports `audiocpp_*`, aliased as `speech`).
- `audiocpp_cli.exe` (aliased as `speech_cli.exe`): The command-line client utility.
- `audiocpp_gguf.exe` (aliased as `speech_gguf.exe`): Lightweight GGUF inspection tool.

---

## 4. Customizing the Selected Models

If your application requires a different subset of models (for example, only TTS or only a single ASR model), you can override `AUDIOCPP_MODELS` during configuration without modifying CMake files:

```powershell
# Example 1: Only Supertonic (TTS) + Qwen3 ASR
cmake --preset client-compact-minimized-cpu -DAUDIOCPP_MODELS="supertonic,qwen3_asr"
cmake --build --preset client-compact-minimized-cpu

# Example 2: Only Nemotron ASR + Granite Speech
cmake --preset client-compact-minimized-cpu -DAUDIOCPP_MODELS="nemotron_asr,granite5asr"
cmake --build --preset client-compact-minimized-cpu
```

Dependencies between models (such as `qwen3_forced_aligner` required by `qwen3_asr`) are resolved and linked automatically.

---

## 5. CMake Configuration Options Reference

| Option | Default in Client Preset | Description |
|---|---|---|
| `AUDIOCPP_MODEL_SET` | `"custom"` | Enables explicit model selection via `AUDIOCPP_MODELS`. |
| `AUDIOCPP_MODELS` | `"supertonic,nemotron_asr,granite5asr,qwen3_asr,parakeet_tdt"` | Comma-separated list of target models to compile. |
| `AUDIOCPP_BUILD_SERVER` | `OFF` | Excludes HTTP server and multi-megabyte embedded WebUI HTML/voice assets. |
| `AUDIOCPP_BUILD_CLI` | `ON` | Builds the standalone `audiocpp_cli` (`speech_cli`) binary. |
| `AUDIOCPP_BUILD_CAPI` | `ON` | Builds the shared C library (`audiocpp.dll` / `libaudiocpp.so`). |
| `AUDIOCPP_BUILD_GGUF_TOOL` | `ON` | Builds the GGUF inspector tool (`audiocpp_gguf`). |
| `AUDIOCPP_STRIP_DEAD_CODE` | `ON` | Enables function-level COMDAT linking and dead code stripping (`/Gy`, `/OPT:REF`). |
| `SPEECHCPP_ENABLE_UNIFIED_ABI` | `OFF` | Set to `ON` only when legacy transcribe.cpp C ABI symbols are required. |
| `ENGINE_BUILD_TESTS` | `OFF` | Skips unit test compilation for maximum build speed. |

---

## 6. Client Application Integration

### Using the C ABI Shared Library
Client desktop applications can dynamically load or link against `audiocpp.dll` / `libaudiocpp.so` using the headers in `capi/include/audiocpp.h`:

```c
#include "audiocpp.h"

// Initialize model loader
audiocpp_model_t* model = audiocpp_load_model("models/supertonic-3", "cpu");

// Run inference...

audiocpp_free_model(model);
```

### Using CMake as a Subproject / FetchContent
In your client application's `CMakeLists.txt`:
```cmake
set(AUDIOCPP_MODEL_SET "custom" CACHE STRING "")
set(AUDIOCPP_MODELS "supertonic,nemotron_asr,granite5asr,qwen3_asr,parakeet_tdt" CACHE STRING "")
set(AUDIOCPP_BUILD_SERVER OFF CACHE BOOL "")
set(AUDIOCPP_BUILD_CLI OFF CACHE BOOL "")
set(ENGINE_BUILD_TESTS OFF CACHE BOOL "")

add_subdirectory(speech.cpp)

target_link_libraries(my_client_app PRIVATE speech)
```
The `speech` alias target points directly to `audiocpp`.
