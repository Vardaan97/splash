#include "model/Qwen4Exp.hpp"
#include "model/Qwen4PlePrefetch.hpp"
#include "model/Qwen4Steps.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  splash::model::Qwen4ExpWeights weights;
  weights.pleHash.multipliers = {104729, 130363, 155921};
  for (uint32_t head = 0; head < weights.layout.pleHeads(); ++head) {
    weights.pleHash.headVocabularies.push_back(19999999 + head * 7919);
    weights.pleHash.headOffsets.push_back(head * 20000000);
  }
  uint32_t cases = 0, failures = 0;
  for (const uint64_t position : {0ULL, 1ULL, 2ULL, 32ULL, 16384ULL, 300000ULL, 262142ULL, 1009990ULL}) {
    std::vector<uint32_t> original(position);
    for (uint64_t at = 0; at < position; ++at) original[at] = 1 + (at * 17 + 11) % 10000;
    for (uint32_t pattern = 0; pattern < 5; ++pattern) {
      auto history = original;
      if (position && pattern == 1) history.back() = weights.layout.pleEosToken;
      if (position > 1 && pattern == 2) history[position - 2] = weights.layout.pleEosToken;
      if (position && pattern == 3) history.back() = weights.layout.pleImageToken;
      if (position > 1 && pattern == 4) history[position - 2] = weights.layout.maskToken;
      for (uint32_t count = 1; count <= 7; ++count) {
        std::vector<uint32_t> proposals(count);
        for (uint32_t row = 0; row < count; ++row) proposals[row] = 41 + row;
        if (pattern == 4) proposals[0] = weights.layout.pleImageToken;
        if (pattern == 2 && count > 1) proposals[1] = weights.layout.pleEosToken;
        const auto input = position ? splash::model::qwen4PlePrefetchInput(
            position - 1, std::span(history).first(position - 1), history.back(), weights.layout.pleNgram)
            : splash::model::Qwen4PlePrefetchInput{0, {}};
        std::vector<uint32_t> expected(count * weights.layout.pleHeads()), actual(expected.size());
        splash::model::qwen4PleRows(weights, position, proposals, history, expected.data());
        splash::model::qwen4PleRows(weights, input.position, proposals, input.history, actual.data());
        ++cases;
        if (actual != expected) {
          if (failures < 12) std::cerr << "position=" << position << " pattern=" << pattern
                                      << " proposals=" << count << " suffix=" << input.history.size() << '\n';
          ++failures;
        }
      }
    }
  }
  std::cout << "PLE prefetch history: " << cases << " cases, " << failures << " failures\n";
  return failures ? 1 : 0;
}
