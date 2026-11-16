//===- bandwidth.m - Qwen app ---------------------------------------------===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The read bandwidth of the GPU: a kernel reads 1 GiB of float4 values with a grid stride, and
// sums them. It prints the best of 10 runs for each number of threads. See README.md.
//
//   clang -O2 -fobjc-arc -framework Foundation -framework Metal decode/bandwidth.m -o bandwidth
//   ./bandwidth
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
static NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
 "kernel void readsum(device const float4 *a [[buffer(0)]], device float *out [[buffer(1)]], constant uint &n [[buffer(2)]],"
 " uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {"
 " float4 s = 0; for (uint i = gid; i < n; i += gs) s += a[i]; out[gid] = s.x + s.y + s.z + s.w; }";
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"readsum"] error:&err];
    size_t bytes = (size_t)1 << 30;  // 1 GiB, larger than any cache
    id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(a.contents, 1, bytes);
    uint n = (uint)(bytes / 16);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (int threads = 10 * 1024; threads <= 10 * 1024 * 64; threads *= 4) {
      id<MTLBuffer> out = [dev newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
      double best = 1e9;
      for (int rep = 0; rep < 12; rep++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:a offset:0 atIndex:0];
        [e setBuffer:out offset:0 atIndex:1];
        [e setBytes:&n length:4 atIndex:2];
        [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        double s = cb.GPUEndTime - cb.GPUStartTime;
        if (rep > 1 && s < best) best = s;
      }
      printf("%7d threads: %.1f GB/s\n", threads, bytes / best / 1e9);
    }
  }
  return 0;
}
