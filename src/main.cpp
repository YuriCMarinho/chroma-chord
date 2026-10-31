#include <iostream>
#include <string>
#include <vector>
#include "AudioFile.h"

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

    audioFile.printSummary();

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

    std::cout << "Successfully extracted mono track (" << monoTrack.size() << " samples).\n";

    return 0;
}
