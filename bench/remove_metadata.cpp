#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: remove_metadata INPUT OUTPUT KEY...\n";
    return 2;
  }
  gguf_init_params params{};
  params.no_alloc = true;
  params.ctx = nullptr;
  gguf_context* context = gguf_init_from_file(argv[1], params);
  if (context == nullptr) {
    std::cerr << "cannot read input\n";
    return 1;
  }
  const size_t data_offset = gguf_get_data_offset(context);
  for (int index = 3; index < argc; ++index) {
    const int64_t old = gguf_remove_key(context, argv[index]);
    std::cout << "remove " << argv[index] << " old_id=" << old << "\n";
  }
  if (!gguf_write_to_file(context, argv[2], true)) {
    std::cerr << "cannot write metadata\n";
    gguf_free(context);
    return 1;
  }
  gguf_free(context);

  std::ifstream input(argv[1], std::ios::binary);
  input.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
  std::vector<char> data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  std::ofstream output(argv[2], std::ios::binary | std::ios::app);
  output.write(data.data(), static_cast<std::streamsize>(data.size()));
  return output.good() ? 0 : 1;
}
