// GPU read speed of 3 kinds of memory: a Metal buffer (newBufferWithLength), a NoCopy buffer over
// anonymous mmap (Vx's vx_metal_alloc), and a NoCopy buffer over a file mmap (llama.cpp).
// Two patterns: a stream over 1 GiB, and 16 KB..1 MB chunks at scattered offsets.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
static NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
 "kernel void stream(device const float4 *a [[buffer(0)]], device float *out [[buffer(1)]], constant uint &n [[buffer(2)]],"
 " uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {"
 " float4 s = 0; for (uint i = gid; i < n; i += gs) s += a[i]; out[gid] = s.x + s.y + s.z + s.w; }\n"
 // a threadgroup of 256 threads reads one chunk of `chunk` float4s at offset starts[tg]
 "kernel void chunks(device const float4 *a [[buffer(0)]], device float *out [[buffer(1)]], constant uint &chunk [[buffer(2)]],"
 " device const uint *starts [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint l [[thread_index_in_threadgroup]]) {"
 " float4 s = 0; device const float4 *p = a + (ulong)starts[tg] * chunk; for (uint i = l; i < chunk; i += 256) s += p[i];"
 " out[tg * 256 + l] = s.x + s.y + s.z + s.w; }";
int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("%s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream"] error:&err];
    id<MTLComputePipelineState> pc = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"chunks"] error:&err];
    const size_t bytes = (size_t)1 << 30;
    const int copies = 4;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("open\n"); return 1; }
    id<MTLBuffer> buf[3][copies];
    const char *names[3] = {"Metal buffer", "anon mmap", "file mmap"};
    for (int c = 0; c < copies; c++) {
      buf[0][c] = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      memset(buf[0][c].contents, 1, bytes);
      void *m = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
      memset(m, 1, bytes);
      buf[1][c] = [dev newBufferWithBytesNoCopy:m length:bytes options:MTLResourceStorageModeShared deallocator:nil];
      void *f = mmap(0, bytes, PROT_READ, MAP_SHARED, fd, (off_t)c * bytes);
      volatile char sink = 0;
      for (size_t i = 0; i < bytes; i += 16384) sink += ((char *)f)[i];
      buf[2][c] = [dev newBufferWithBytesNoCopy:f length:bytes options:MTLResourceStorageModeShared deallocator:nil];
      if (!buf[1][c] || !buf[2][c]) { printf("nocopy failed\n"); return 1; }
    }
    id<MTLBuffer> out = [dev newBufferWithLength:64 << 20 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    uint n = (uint)(bytes / 16);
    for (int round = 0; round < 2; round++)
    for (int k = 0; k < 3; k++) {
      // stream
      double t = 0; int count = 0;
      for (int rep = 0; rep < 8; rep++)
        for (int c = 0; c < copies; c++) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:ps];
          [e setBuffer:buf[k][c] offset:0 atIndex:0];
          [e setBuffer:out offset:0 atIndex:1];
          [e setBytes:&n length:4 atIndex:2];
          [e dispatchThreads:MTLSizeMake(160 * 1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [e endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          if (rep > 0) { t += cb.GPUEndTime - cb.GPUStartTime; count++; }
        }
      printf("%-13s stream    : %.1f GB/s\n", names[k], bytes * (double)count / t / 1e9);
      // scattered chunks: each launch reads 256 MB in chunks
      for (uint chunkBytes = 16384; chunkBytes <= (1 << 20); chunkBytes *= 4) {
        uint chunk = chunkBytes / 16, groups = (256u << 20) / chunkBytes, slots = (uint)(bytes / chunkBytes);
        id<MTLBuffer> st = [dev newBufferWithLength:groups * 4 options:MTLResourceStorageModeShared];
        uint *sp = st.contents;
        uint v = 12345;
        for (uint i = 0; i < groups; i++) { v = v * 1664525u + 1013904223u; sp[i] = v % slots; }
        t = 0; count = 0;
        for (int rep = 0; rep < 8; rep++)
          for (int c = 0; c < copies; c++) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pc];
            [e setBuffer:buf[k][c] offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&chunk length:4 atIndex:2];
            [e setBuffer:st offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (rep > 0) { t += cb.GPUEndTime - cb.GPUStartTime; count++; }
          }
        printf("%-13s chunks %4u KB: %.1f GB/s\n", names[k], chunkBytes / 1024, (256.0 * (1 << 20)) * count / t / 1e9);
      }
    }
  }
  return 0;
}
