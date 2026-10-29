#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "chroma-chord v0.1.0 - Audio Chord Recognition Engine\n";
    std::cout << "Usage: chroma-chord <path-to-wav-file>\n";

    if (argc < 2) {
        std::cout << "No audio file provided.\n";
        return 0;
    }

    std::cout << "Audio file specified: " << argv[1] << "\n";
    return 0;
}
