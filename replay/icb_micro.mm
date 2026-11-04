// ICB vs direct encode micro benchmark: N dependent GEMV-like dispatches, K buffers + constants each.
// build: clang++ -O2 -std=c++17 -fobjc-arc -framework Metal -framework Foundation icb_micro.mm -o icb_micro
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
static double now_us() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e3; }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double pmin(std::vector<double> v) { return *std::min_element(v.begin(), v.end()); }

static NSString* SRC = @R"MSL(
#include <metal_stdlib>
using namespace metal;
// y[r] = sum_c W[r,c] * x[c]  (+ b[r]); one simdgroup (32 threads) per row; n rows given by constant buffer
kernel void gemv(device const half* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                 device const float* b [[buffer(3)]], device const float* b2 [[buffer(4)]], device float* z [[buffer(5)]],
                 constant uint* cst [[buffer(6)]],
                 uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint cols = cst[0], pos = cst[1];
  float acc = 0.f;
  for (uint c = lane; c < cols; c += 32) acc += float(W[r * cols + c]) * x[c % 256];
  acc = simd_sum(acc);
  if (lane == 0) { y[r] = acc * 1e-3f + b[r % 64] * 0.f + b2[0] * 0.f + float(pos) * 0.f; z[r % 64] = acc; }
}
// tiny kernel for pure gap measurement
kernel void tiny(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint* cst [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  y[i] = x[i] * 1.0001f + float(cst[0]) * 0.f;
}
)MSL";

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 150;
  int rows = argc > 2 ? atoi(argv[2]) : 256;     // threadgroups per dispatch
  int cols = argc > 3 ? atoi(argv[3]) : 2048;    // bytes read per dispatch = rows*cols*2
  int reps = argc > 4 ? atoi(argv[4]) : 60;
  const char* kname = argc > 5 ? argv[5] : "gemv";
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:SRC options:nil error:&err];
    if (!lib) { NSLog(@"lib %@", err); return 1; }
    bool tiny = !strcmp(kname, "tiny");
    id<MTLFunction> fn = [lib newFunctionWithName:@(kname)];
    MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = fn; pd.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&err];
    id<MTLComputePipelineState> pso_plain = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso || !pso_plain) { NSLog(@"pso %@", err); return 1; }
    NSLog(@"device %@ N=%d rows=%d cols=%d bytes/dispatch=%.2f MB kernel=%s", dev.name, N, rows, cols, rows * (double)cols * 2 / 1e6, kname);
    // buffers: per dispatch one weight buffer + one const buffer; chain through two ping-pong activation buffers
    std::vector<id<MTLBuffer>> W(N), C(N);
    size_t wbytes = tiny ? 4096 : (size_t)rows * cols * 2;
    for (int i = 0; i < N; ++i) {
      W[i] = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
      memset(W[i].contents, 0x3c, wbytes);   // half 1.0
      C[i] = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
      uint32_t* c = (uint32_t*)C[i].contents; c[0] = cols; c[1] = i;
    }
    id<MTLBuffer> A = [dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
    id<MTLBuffer> D = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    float* a = (float*)A.contents; for (int i = 0; i < 16384; ++i) a[i] = 0.01f * (i % 7);
    MTLSize grid = tiny ? MTLSizeMake(1024, 1, 1) : MTLSizeMake(32 * rows, 1, 1), tg = tiny ? MTLSizeMake(64, 1, 1) : MTLSizeMake(32, 1, 1);
    if (!tiny) { // one threadgroup per row of 32 threads: use dispatchThreadgroups in direct mode and concurrentDispatchThreadgroups in ICB mode
    }
    MTLSize tgs = MTLSizeMake(rows, 1, 1);
    // ---- ICB
    MTLIndirectCommandBufferDescriptor* d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    d.inheritBuffers = NO; d.inheritPipelineState = NO; d.maxKernelBufferBindCount = 31;
    id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d maxCommandCount:N options:MTLResourceStorageModePrivate];
    if (!icb) { NSLog(@"icb alloc failed"); return 1; }
    for (int barrier = 0; barrier < 2; ++barrier) { (void)barrier; }
    auto encode_icb = [&](id<MTLIndirectCommandBuffer> ib, bool bar) {
      for (int i = 0; i < N; ++i) {
        id<MTLIndirectComputeCommand> c = [ib indirectComputeCommandAtIndex:i];
        [c setComputePipelineState:pso];
        id<MTLBuffer> in = (i % 2) ? B : A, out = (i % 2) ? A : B;
        if (tiny) { [c setKernelBuffer:in offset:0 atIndex:0]; [c setKernelBuffer:out offset:0 atIndex:1]; [c setKernelBuffer:C[i] offset:0 atIndex:2]; }
        else {
          [c setKernelBuffer:W[i] offset:0 atIndex:0]; [c setKernelBuffer:in offset:0 atIndex:1]; [c setKernelBuffer:out offset:0 atIndex:2];
          [c setKernelBuffer:D offset:0 atIndex:3]; [c setKernelBuffer:D offset:256 atIndex:4]; [c setKernelBuffer:D offset:512 atIndex:5]; [c setKernelBuffer:C[i] offset:0 atIndex:6];
        }
        if (bar) [c setBarrier];
        if (tiny) [c concurrentDispatchThreads:grid threadsPerThreadgroup:tg]; else [c concurrentDispatchThreadgroups:tgs threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
    };
    double t0 = now_us(); encode_icb(icb, true); double t_enc_icb = now_us() - t0;
    NSLog(@"ICB one-time encode of %d commands: %.0f us (%.2f us/cmd)", N, t_enc_icb, t_enc_icb / N);
    std::vector<id<MTLResource>> res; for (int i = 0; i < N; ++i) { res.push_back(W[i]); res.push_back(C[i]); } res.push_back(A); res.push_back(B); res.push_back(D);
    // residency set
    id<MTLResidencySet> rs = nil;
    if (@available(macOS 15.0, *)) {
      MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new]; rd.initialCapacity = res.size();
      rs = [dev newResidencySetWithDescriptor:rd error:&err];
      for (auto& r : res) [rs addAllocation:r];
      [rs addAllocation:icb];
      [rs commit];
      [q addResidencySet:rs];
    }
    struct R { std::vector<double> cpu, gpu, wall; };
    R rd_, ri_, rr_;
    auto run = [&](int mode, R& r) {
      double t0 = now_us();
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      if (mode == 0) {
        for (int i = 0; i < N; ++i) {
          [e setComputePipelineState:pso_plain];
          id<MTLBuffer> in = (i % 2) ? B : A, out = (i % 2) ? A : B;
          if (tiny) { [e setBuffer:in offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1]; [e setBuffer:C[i] offset:0 atIndex:2]; }
          else {
            [e setBuffer:W[i] offset:0 atIndex:0]; [e setBuffer:in offset:0 atIndex:1]; [e setBuffer:out offset:0 atIndex:2];
            [e setBuffer:D offset:0 atIndex:3]; [e setBuffer:D offset:256 atIndex:4]; [e setBuffer:D offset:512 atIndex:5]; [e setBuffer:C[i] offset:0 atIndex:6];
          }
          if (i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          if (tiny) [e dispatchThreads:grid threadsPerThreadgroup:tg]; else [e dispatchThreadgroups:tgs threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
      } else {
        // residency set attached to queue (mode 1) or explicit useResources (mode 2)
        if (mode == 2) [e useResources:res.data() count:res.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [e executeCommandsInBuffer:icb withRange:NSMakeRange(0, N)];
      }
      [e endEncoding];
      [cb commit];
      double t1 = now_us();
      [cb waitUntilCompleted];
      double t2 = now_us();
      r.cpu.push_back(t1 - t0); r.wall.push_back(t2 - t0); r.gpu.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e6);
    };
    for (int w = 0; w < 5; ++w) { run(0, rd_); run(1, ri_); }
    rd_ = R(); ri_ = R();
    R r2;
    for (int i = 0; i < reps; ++i) { run(0, rd_); run(1, ri_); if (!tiny || true) run(2, r2); }
    printf("N=%d kernel=%s MB/dispatch=%.2f\n", N, kname, rows * (double)cols * 2 / 1e6);
    printf("direct       : cpu med %8.1f min %8.1f us (%.2f us/launch) | gpu med %9.1f min %9.1f us (%.2f us/launch) | wall med %9.1f\n", med(rd_.cpu), pmin(rd_.cpu), med(rd_.cpu) / N, med(rd_.gpu), pmin(rd_.gpu), med(rd_.gpu) / N, med(rd_.wall));
    printf("icb residency: cpu med %8.1f min %8.1f us (%.2f us/launch) | gpu med %9.1f min %9.1f us (%.2f us/launch) | wall med %9.1f\n", med(ri_.cpu), pmin(ri_.cpu), med(ri_.cpu) / N, med(ri_.gpu), pmin(ri_.gpu), med(ri_.gpu) / N, med(ri_.wall));
    printf("icb useRes   : cpu med %8.1f min %8.1f us (%.2f us/launch) | gpu med %9.1f min %9.1f us (%.2f us/launch) | wall med %9.1f\n", med(r2.cpu), pmin(r2.cpu), med(r2.cpu) / N, med(r2.gpu), pmin(r2.gpu), med(r2.gpu) / N, med(r2.wall));
    // check results equal: run direct then icb and compare A/B
    float* o = (float*)A.contents; float s = 0; for (int i = 0; i < 256; ++i) s += o[i]; printf("checksum A %g\n", s);
  }
  return 0;
}
