#define PF7_TEST 0
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
              const std::vector<std::string> modes{"qk2","prob2","rows4"};
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
                    if (dispatch.pipelineName.find("_split") != std::string::npos) {
                      dispatch.pipelineName += "_"+mode;
                      if (mode == "rows4") dispatch.threadgroups.y = (rows+3)/4;
                    } else if (mode == "rows4" && dispatch.pipelineName.find("_reduce") != std::string::npos) {
                      dispatch.pipelineName += "_rows4";
                      dispatch.threadgroups.y = 4*(shape.queryHeads/shape.kvHeads);
                      dispatch.threadgroups.z = (rows+3)/4;
                    }
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
#endif
                }
              }
              ++cases;
            }
      }
    // Verify shares the page loop; retain its original lane/split geometry.
    for (auto format : {kv::Format::Int8,kv::Format::BFloat16})
      for (const uint32_t history : histories)
        for (const uint32_t lanes : {1u,4u}) {
          const AttentionShape shape{24,2,256,format};
          const kv::Layout layout{1,2,256,format};
          AttentionFixturePlan::Histories past{};
          std::fill_n(past.begin(),lanes,history);
          attentionPipeBaseline(true);
          const auto attention=PagedAttention::verifyPlan(lanes,24,layout,std::span(past).first(lanes));
          const auto plan=AttentionFixturePlan::make(shape,lanes,kv::kVerifyRows,past,attention.workspace,
              {2,1,layout.extentAlignmentPages()*3});
          AttentionFixture fixture(backend,plan,"verify-pipe-test");fixture.fill();
          QsaMask mask;
          mask.words=(history+kv::kVerifyRows+127)/128;
          mask.bits=backend.allocateBuffer(uint64_t(lanes)*kv::kVerifyRows*mask.words*4,metal::BufferStorage::Shared,"verify-mask");
          auto *bits=static_cast<uint32_t *>(mask.bits.contents());
          for (uint32_t row=0;row<lanes*kv::kVerifyRows;++row)
            for(uint32_t word=0;word<mask.words;++word)
              bits[uint64_t(row)*mask.words+word]=word%3==1 ? 0u : (0x55555555u>>(row%2));
          metal::CommandGraph graph;fixture.addGraph(graph,attention,mask);
          (void)backend.submitCommandAsync(graph.command()).wait();
          auto output=fixture.buffer(AttentionFixture::Tensor::Output);
          std::vector<std::byte> reference(output.sizeBytes());
          std::memcpy(reference.data(),output.contents(),output.sizeBytes());
          for (const std::string mode : {"qk2","prob2"}) {
            auto dispatches=std::vector<metal::ComputeDispatch>(graph.dispatches().begin(),graph.dispatches().end());
            for(auto &dispatch:dispatches)
              if(dispatch.pipelineName.find("_split")!=std::string::npos) dispatch.pipelineName+="_"+mode;
            (void)backend.submitCommandAsync(dispatches).wait();
            if(std::memcmp(reference.data(),output.contents(),output.sizeBytes())) throw std::runtime_error("verify memcmp: "+mode);
          }
          ++cases;
        }
    std::cout << "PASS prefill-attention-variants cases=" << cases << '\n';
    return 0;
  } catch (const std::exception &error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
