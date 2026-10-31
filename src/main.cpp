#include <iostream>
#include <string>
#include <vector>
#include "AudioFile.h"

// Temporal segmentation parameters (in seconds)
constexpr double kWindowDurationSec = 0.5;   // length of each analysis window
constexpr double kHopDurationSec    = 0.25;  // step between window starts (50% overlap)

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-wav-file>\n";
        return 1;
    }

    const std::string filePath = argv[1];

    // Instantiate the audio reader
    AudioFile<double> audioFile;

    // Load the .wav file
    if (!audioFile.load(filePath)) {
        std::cerr << "Error: Failed to load audio file: " << filePath << "\n";
        return 1;
    }

    // Store the mono track in a std::vector
    const int numChannels = audioFile.getNumChannels();
    const int numSamples = audioFile.getNumSamplesPerChannel();
    std::vector<double> monoTrack(numSamples);

    for (int i = 0; i < numSamples; ++i) {
        double sum = 0.0;
        for (int ch = 0; ch < numChannels; ++ch) {
            sum += audioFile.samples[ch][i];
        }
        monoTrack[i] = sum / numChannels;
    }

    // Print file metadata
    std::cout << "Sample Rate:   " << audioFile.getSampleRate() << " Hz\n";
    std::cout << "Total Samples: " << monoTrack.size() << "\n";

    // Convert window/hop durations to sample counts for this file's sample rate
    const int sampleRate = static_cast<int>(audioFile.getSampleRate());
    const int windowSize = static_cast<int>(kWindowDurationSec * sampleRate);
    const int hopLength  = static_cast<int>(kHopDurationSec * sampleRate);

    std::cout << "Window Size:   " << windowSize << " samples (" << kWindowDurationSec << " s)\n";
    std::cout << "Hop Length:    " << hopLength  << " samples (" << kHopDurationSec << " s)\n";

    return 0;
}
