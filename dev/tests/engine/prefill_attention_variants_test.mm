#define PF7_TEST 1
#include "AttentionFixture.hpp"
#if PF7_TEST
#include "ops/PrefillCompactPolicy.hpp"
#else
#include "ops/AttentionPolicy.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace splash;
using namespace splash::ops;
using namespace splash::test;
// fp64 spot reference at long histories, including the actual quantized KV.
void referenceEnvelope(const AttentionFixture &fixture, QsaMask mask,
                       const std::vector<uint16_t> &baseline, const std::vector<uint16_t> &candidate) {
  const auto &p = fixture.plan();
  const uint32_t limit = p.histories[0] + 1;
  const auto *bits = static_cast<const uint32_t *>(mask.bits.contents()) + uint64_t(mask.row0)*mask.words;
  const auto *queries = static_cast<const uint16_t *>(fixture.buffer(AttentionFixture::Tensor::Queries).contents());
  std::vector<std::pair<uint32_t,double>> scores;
  double maximum = -INFINITY;
  for (uint32_t token=0;token<limit;++token) {
    const uint32_t block=token/4;
    if (!((bits[block/32]>>(block%32))&1)) continue;
    double dot=0;
    for (uint32_t d=0;d<256;++d) {
      const double k=p.shape.format==kv::Format::Int8 ? fixture.keyRow<int8_t>(0,0,token)[d] :
          bf16ToFloat(fixture.keyRow<uint16_t>(0,0,token)[d]);
      dot+=double(bf16ToFloat(queries[d]))*k;
    }
    if (p.shape.format==kv::Format::Int8) dot*=*fixture.scale(0,SPLASH_KV_KEY_SCALES,0,token);
    dot*=.0625;
    scores.emplace_back(token,dot);maximum=std::max(maximum,dot);
  }
  for (uint32_t d : {0u,127u,255u}) {
    double numerator=0,denominator=0;
    for (const auto &[token,score] : scores) {
      double value=p.shape.format==kv::Format::Int8 ? fixture.valueColumn<int8_t>(0,0,token)[d*32] :
          bf16ToFloat(fixture.valueColumn<uint16_t>(0,0,token)[d*32]);
      if (p.shape.format==kv::Format::Int8) value*=*fixture.scale(0,SPLASH_KV_VALUE_SCALES,0,token);
      const double weight=std::exp(score-maximum);
      numerator+=value*weight;denominator+=weight;
    }
    const double expected=denominator ? numerator/denominator : 0;
    const double baseError=std::fabs(bf16ToFloat(baseline[d])-expected);
    const double candidateError=std::fabs(bf16ToFloat(candidate[d])-expected);
    const double envelope=baseError+2*ulpBf16(float(expected))+1e-4;
    if (baseError>.02 || candidateError>envelope)
      throw std::runtime_error("PF7 increased the fp64 reference error envelope");
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: prefill-attention-variants METALLIB [--full]");
    metal::MetalBackend backend(argv[1]);
    if (backend.capabilities().appleGpuFamily < 10) {
      std::cout << "UNAVAILABLE Apple9: candidates restricted to Apple10; M5 memcmp still required\n";
      return 0;
    }
    const bool full = argc > 2 && std::string_view(argv[2]) == "--full";
    const std::vector<uint32_t> histories = full ? std::vector<uint32_t>{0,2048,4096,32768,65536,131072,200704} : std::vector<uint32_t>{0,257};
    const std::vector<uint32_t> rowCounts = full ? std::vector<uint32_t>{1,4,7,8,9,16,17,32,128,129,2048} : std::vector<uint32_t>{1,8,17,32};
    uint64_t cases = 0;
    for (const auto format : {kv::Format::Int8, kv::Format::BFloat16})
      for (const uint32_t geometry : {0u,1u,2u}) {
#if PF7_TEST
        if (geometry != 2) continue;
#endif
        const AttentionShape shape{geometry == 1 ? 16u : 24u, geometry == 0 ? 4u : 2u, 256, format};
        const kv::Layout layout{1,shape.kvHeads,shape.headDimension,shape.format};
        for (const uint32_t history : histories)
          for (const uint32_t rows : rowCounts)
            for (uint32_t pattern = 0; pattern < (geometry == 2 ? 4u : 1u); ++pattern) {
#if PF7_TEST
              prefillCompactBaseline(true);
#else
              attentionPipeBaseline(true);
#endif
              const auto attention = PagedAttention::prefillPlan(rows,shape.queryHeads,layout);
              auto scratch = attention.workspace;
#if PF7_TEST
              const auto compact = prefillCompactWorkspace(rows);
              scratch.partialsBytes = std::max(scratch.partialsBytes, compact.partialsBytes);
              scratch.statisticsBytes = std::max(scratch.statisticsBytes, compact.statisticsBytes);
#endif
              AttentionFixturePlan::Histories past{history,0,0,0};
              const auto plan = AttentionFixturePlan::make(shape,1,rows,past,scratch,{2,1,layout.extentAlignmentPages()*3});
              AttentionFixture fixture(backend,plan,"prefill-variant-test");
              fixture.fill();
              QsaMask mask;
              if (geometry == 2) {
                mask.words = (history + rows + 127)/128;
                mask.row0 = 2;
                mask.bits = backend.allocateBuffer(uint64_t(rows+2)*mask.words*4,metal::BufferStorage::Shared,"mask");
                auto *bits = static_cast<uint32_t *>(mask.bits.contents());
                std::fill_n(bits, uint64_t(rows+2)*mask.words, 0u);
                std::fill_n(bits, uint64_t(2)*mask.words, ~0u); // poisoned prefix
                for (uint32_t row = 0; row < rows; ++row) {
                  const uint32_t blocks = (history+row+4)/4;
                  for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t hash = (block*2654435761u) ^ (pattern == 1 ? row*2246822519u : 0u);
                    const bool selected = pattern == 0 ? block%17 == 0 : pattern == 1 ? hash%std::max(1u,blocks/512) == 0 : pattern == 2 ? block%64 < 8 : false;
                    if (selected) bits[uint64_t(row+2)*mask.words+block/32] |= 1u<<(block%32);
                  }
                  if (pattern != 3) {
                    const uint32_t block = (history+row)/4;
                    bits[uint64_t(row+2)*mask.words+block/32] |= 1u<<(block%32);
                  }
                }
              }
              metal::CommandGraph baseline;
              fixture.addGraph(baseline,attention,mask);
              (void)backend.submitCommandAsync(baseline.command()).wait();
              const auto out = fixture.buffer(AttentionFixture::Tensor::Output);
              std::vector<uint16_t> reference(out.sizeBytes()/2), repeat(reference.size());
              std::memcpy(reference.data(),out.contents(),out.sizeBytes());
#if PF7_TEST
              const std::vector<std::string> modes{"compact"};
#else
              const std::vector<std::string> modes{"qk2","prob2"};
#endif
              for (const auto &mode : modes) {
                (void)mode;
                for (uint32_t arm = 0; arm < 2; ++arm) {
                  std::memset(out.contents(),0,out.sizeBytes());
#if PF7_TEST
                  metal::CommandGraph graph;
                  const uint32_t data = shape.kvHeads*32*256*(format == kv::Format::Int8 ? 1 : 2);
                  const uint32_t scale = format == kv::Format::Int8 ? shape.kvHeads*32*4 : 0;
                  const auto layer = splash_kv_layer(plan.geometry.extentPages,data,scale,plan.geometry.layer);
                  const auto chunk = PagedAttention::prefillParams(history,rows,plan.stride,plan.pages[0]);
                  addPrefillCompact(graph,layer,fixture.buffer(AttentionFixture::Tensor::Queries),out,
                      fixture.buffer(AttentionFixture::Tensor::Partials),fixture.buffer(AttentionFixture::Tensor::Statistics),
                      fixture.buffer(AttentionFixturePlan::table(0)),chunk,mask,format);
                  (void)backend.submitCommandAsync(graph.command()).wait();
#else
                  std::vector<metal::ComputeDispatch> dispatches(baseline.dispatches().begin(),baseline.dispatches().end());
                  for (auto &dispatch : dispatches)
                    if (dispatch.pipelineName.find("_split") != std::string::npos) dispatch.pipelineName += "_"+mode;
                  (void)backend.submitCommandAsync(dispatches).wait();
#endif
                  if (arm == 1 && std::memcmp(repeat.data(),out.contents(),out.sizeBytes())) throw std::runtime_error("candidate repeat changed bits");
                  std::memcpy(repeat.data(),out.contents(),out.sizeBytes());
#if !PF7_TEST
                  if (std::memcmp(reference.data(),out.contents(),out.sizeBytes())) throw std::runtime_error("PF3 output memcmp failed: "+mode);
#else
                  double aa=0,bb=0,ab=0;
                  for (size_t i=0;i<reference.size();++i) {
                    const float a=bf16ToFloat(reference[i]), b=bf16ToFloat(repeat[i]);
                    if (!std::isfinite(a)||!std::isfinite(b)||std::fabs(a-b)>.02f) throw std::runtime_error("PF7 finite/absolute tolerance");
                    aa+=a*a;bb+=b*b;ab+=a*b;
                  }
                  if ((aa||bb) && (!(aa&&bb) || ab/std::sqrt(aa*bb)<.999)) throw std::runtime_error("PF7 cosine tolerance");
                  if (history >= 32768 && rows == 8) referenceEnvelope(fixture,mask,reference,repeat);
#endif
                }
              }
              ++cases;
            }
      }
    std::cout << "PASS prefill-attention-variants cases=" << cases << '\n';
    return 0;
  } catch (const std::exception &error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
