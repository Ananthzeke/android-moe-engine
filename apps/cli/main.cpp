#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
  if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
    std::cout << "Usage: ame [--help | --version]\nHost coding foundation; inference commands are "
                 "not implemented.\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    std::cout << "ame 0.1.0 (coding foundation)\n";
    return 0;
  }
  std::cerr << "Unsupported arguments; use ame --help.\n";
  return 2;
}
