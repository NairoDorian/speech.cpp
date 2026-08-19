// Parity harness for the MOSS-Audio-Tokenizer-v2 RLFQ dequantizer: runs the
// dequant on a fixed code matrix and dumps the latent for comparison against
// the Python reference (scripts/codec_dequant_ref.py).

#include "engine/models/moss/shared/audio_tokenizer_quantizer.h"

#include "codec_weights_path.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    // The codec weights are a multi-GB external download, so there is no
    // in-tree default. Exit 2 (skipped) without one, matching the convention
    // used by the abi_* bridge tests.
    if (argc < 2) {
        std::fprintf(stderr,
                     "codec_dequant_parity: skipped (usage: codec_dequant_parity "
                     "<codec-weights-file-or-dir> [out.txt])\n");
        return 2;
    }
    const std::filesystem::path codec_dir = argv[1];
    const std::filesystem::path out_path =
        argc > 2 ? std::filesystem::path(argv[2]) : std::filesystem::path("cpp_latent.txt");

    constexpr int64_t kNumQuantizers = 12;
    constexpr int64_t kSteps = 8;

    std::vector<std::vector<int32_t>> codes(kNumQuantizers, std::vector<int32_t>(kSteps));
    for (int64_t i = 0; i < kNumQuantizers; ++i) {
        for (int64_t t = 0; t < kSteps; ++t) {
            codes[static_cast<size_t>(i)][static_cast<size_t>(t)] = static_cast<int32_t>((i * 37 + t * 5) % 1024);
        }
    }

    try {
        const auto weights = moss_parity::open_codec_weights(codec_dir);
        engine::models::moss::MossAudioTokenizerQuantizer dequantizer(*weights, kNumQuantizers);
        const std::vector<float> latent = dequantizer.decode(codes);  // [code_dim, steps]
        const int64_t code_dim = dequantizer.code_dim();

        double mean = 0.0;
        for (const float value : latent) {
            mean += value;
        }
        mean /= static_cast<double>(latent.size());
        double var = 0.0;
        for (const float value : latent) {
            var += (value - mean) * (value - mean);
        }
        var /= static_cast<double>(latent.size());

        std::printf("shape %lld %lld\n", static_cast<long long>(code_dim), static_cast<long long>(kSteps));
        std::printf("first16");
        for (int i = 0; i < 16; ++i) {
            std::printf(" %.6f", latent[static_cast<size_t>(i)]);
        }
        std::printf("\n");
        std::printf("mean %.6f std %.6f\n", mean, std::sqrt(var));

        std::ofstream out(out_path);
        out.precision(8);
        for (const float value : latent) {
            out << value << "\n";
        }
        std::printf("wrote %lld values to %s\n", static_cast<long long>(latent.size()), out_path.string().c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
