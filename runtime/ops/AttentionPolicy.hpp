#pragma once
#include <string>
#include <cstdint>
#include <string_view>
namespace splash::metal { class MetalBackend; }
namespace splash::ops {
// Scoped comparison control for native gate tools; never used by serving.
void attentionPipeBaseline(bool baseline);
// Only a successful device-local self-test permits selection.
enum class AttentionPipe { Off, Qk2, Prob2, Rows4 };
AttentionPipe attentionPipeSelection();
uint32_t attentionPipeRows(AttentionPipe mode);
std::string_view attentionPipeReduce(std::string_view baseline, AttentionPipe mode);
bool attentionStatsEnabled();
void logAttentionPages(metal::MetalBackend &backend, uint64_t request, const void *mask,
    uint32_t words, uint32_t committed, uint32_t rows, bool sparse);
void attentionKernelSelfTest(metal::MetalBackend &backend);
std::string_view attentionPipePipeline(std::string_view baseline, AttentionPipe mode);
}
