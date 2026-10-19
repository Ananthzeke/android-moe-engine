#include "sha256.hpp"

#include <iostream>
#include <string>

namespace {
int failures = 0;
void check(const std::string& value, const std::string& expected, const char* name) {
  const std::string actual = ame::crypto::sha256_hex(value);
  if (actual != expected) {
    std::cerr << "SHA-256 " << name << " mismatch: " << actual << " != " << expected << '\n';
    ++failures;
  }
}
}  // namespace

int main() {
  check("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
  check("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
  check("The quick brown fox jumps over the lazy dog",
        "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592", "fox");
  check(std::string(1000000, 'a'),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "million-a");
  return failures == 0 ? 0 : 1;
}
