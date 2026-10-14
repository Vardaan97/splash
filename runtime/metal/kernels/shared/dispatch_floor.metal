#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void dispatch_floor_empty(uint i [[thread_position_in_grid]]) { (void)i; }

// Same rows, streams and width as the hyper combine. Ping-pong buffers
// make every dispatch depend on the preceding dispatch's writes.
[[max_total_threads_per_threadgroup(256)]]
kernel void dispatch_floor_chain(device const float *input [[buffer(0)]],
                                  device float *output [[buffer(1)]],
                                  constant uint &count [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i < count) output[i] = input[i] + 0.5f;
}
