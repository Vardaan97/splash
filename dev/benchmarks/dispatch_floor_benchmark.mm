// MTL3 dispatch floor using the same encoder selection and buffer-scope
// barrier as the concurrent decode backend. No model or server required.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 3) return 2;
    const bool check = argc == 3 && std::string(argv[2]) == "--check";
    const unsigned rounds = check ? 1 : argc == 3 ? std::stoul(argv[2]) : 40;
    if (!rounds) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!library) { std::fprintf(stderr, "%s\n", error.description.UTF8String); return 1; }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    constexpr unsigned count = 8 * 4 * 2560;
    id<MTLBuffer> a = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
    std::puts("kind,dispatches,serial_us,concurrent_us,no_barrier_us,boundary_us,barrier_us");
    for (bool chain : {false, true}) {
      id<MTLFunction> fn = [library newFunctionWithName:chain ? @"dispatch_floor_chain" : @"dispatch_floor_empty"];
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
      if (!pipeline) return 1;
      for (unsigned n = 1; n <= 1024; ++n) {
        if (check && (n & (n - 1)) && n != 3 && n != 1023) continue;
        std::vector<double> samples[3];
        for (unsigned r = 0; r < rounds + 2; ++r)
          for (unsigned arm = 0; arm < 3; ++arm) {
            @autoreleasepool {
            const unsigned mode = (arm + r) % 3;
            if (chain && mode == 2) continue; // A dependent chain requires barriers.
            const bool concurrent = mode != 0;
            std::fill_n(static_cast<float *>(a.contents), count, 1.0f);
            std::fill_n(static_cast<float *>(b.contents), count, -1.0f);
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = concurrent
                ? [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                : [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            if (chain) [encoder setBytes:&count length:sizeof(count) atIndex:2];
            for (unsigned d = 0; d < n; ++d) {
              if (mode == 1) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
              if (chain && mode != 2) {
                [encoder setBuffer:(d & 1) ? b : a offset:0 atIndex:0];
                [encoder setBuffer:(d & 1) ? a : b offset:0 atIndex:1];
              }
              [encoder dispatchThreadgroups:MTLSizeMake(chain && mode != 2 ? count / 256 : 1, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(chain && mode != 2 ? 256 : 1, 1, 1)];
            }
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
              std::fprintf(stderr, "%s\n", command.error.description.UTF8String); return 1;
            }
            if (chain && mode != 2) {
              const float *result = static_cast<const float *>(((n & 1) ? b : a).contents);
              for (unsigned i = 0; i < count; ++i)
                if (result[i] != 1.0f + 0.5f * n) return 1;
            }
            if (r >= 2) samples[mode].push_back((command.GPUEndTime - command.GPUStartTime) * 1e6 / n);
            }
          }
        for (auto &s : samples) std::sort(s.begin(), s.end());
        const double serial = samples[0][samples[0].size()/2], concurrent = samples[1][samples[1].size()/2];
        const double noBarrier = chain ? __builtin_nan("") : samples[2][samples[2].size()/2];
        if (!check) std::printf("%s,%u,%.6f,%.6f,%.6f,%.6f,%.6f\n", chain ? "chain" : "empty", n,
                                serial, concurrent, noBarrier, serial-noBarrier, concurrent-noBarrier);
      }
    }
    if (check) std::puts("dispatch-floor: PASS");
    return 0;
  }
}
