#include "benchmarks/DecodeProfileInput.hpp"
#include "TestChecks.hpp"

#include <sstream>
#include <iostream>

int main() {
  using namespace splash::bench;
  using splash::test::require;
  using splash::test::rejects;
  require(profileDecodeTokens(8, 8, true) == 520 && profileDecodeTokens(8, 8, false) == 264,
          "decode capacity does not cover all widths");
  require(profileDecodeTokens(4294967295u, 8, true) > 4294967295u, "decode capacity overflowed");
  std::istringstream valid("0 42\n4294967295\t17");
  require(profilePrompt(valid) == std::vector<uint32_t>({0, 42, 4294967295u, 17}), "token IDs changed");
  for (const char *text : {"-1", "4294967296", "3x", "[1,2]"}) {
    std::istringstream bad(text);
    rejects([&] { profilePrompt(bad); }, "uint32", "invalid token accepted");
  }
  std::istringstream empty(" \n");
  rejects([&] { profilePrompt(empty); }, "empty", "empty prompt accepted");
  require(profileCacheBytes("0") == 0 && profileCacheBytes("2147483648") == 2147483648ull,
          "cache bytes changed");
  for (const char *text : {"-1", "1", "2G", "18446744073709551616"})
    rejects([&] { profileCacheBytes(text); }, "PLE cache", "invalid cache accepted");
  std::cout << "PASS decode profile input (CPU only)\n";
}
