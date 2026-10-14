#pragma once
#include "ops/PagedAttention.hpp"
namespace splash::ops {
// Scoped comparison control for native gate tools; never used by serving.
void prefillCompactBaseline(bool baseline);
void capturePrefillDistribution(const float *row, uint32_t vocabulary);
void prefillCompactSelfTest(metal::MetalBackend &backend);
bool prefillCompactRequested();
bool prefillCompactReady();
AttentionWorkspace prefillCompactWorkspace(uint32_t rows);
void addPrefillCompact(metal::CommandGraph &graph, SplashKvLayer layer,
    metal::MetalBuffer queries, metal::MetalBuffer output, metal::MetalBuffer partials,
    metal::MetalBuffer statistics, metal::MetalBuffer table,
    const kv::ChunkedPrefillParams &chunk, QsaMask mask, kv::Format format);
}
