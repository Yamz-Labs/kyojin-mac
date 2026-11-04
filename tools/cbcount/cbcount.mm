// Counts the Metal command buffers MLX commits and sums their GPU time (swizzles -[MTLCommandBuffer commit]). Used by tools/engram_pipe.py: a decode
// step must commit none while the Python graph is built and one in async_eval; a host read inside the graph forces an extra commit and serialises the
// CPU encode with the GPU.
// Also records, per commit, the Metal function name of every compute dispatch encoded into it (swizzles the device's pipeline constructors and
// -[MTLComputeCommandEncoder setComputePipelineState:]), so kyojin_mlx.gpuspan can split a round's GPU time per kernel class (probe gpu_kern).
// build: clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation tools/cbcount/cbcount.mm -o tools/cbcount/libcbcount.dylib
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

static long g_commits = 0;
static double g_gpu_us = 0;
static long g_done = 0;
#define NSEQ (1 << 16)
static double g_t0[NSEQ], g_t1[NSEQ];                                    // GPUStartTime / GPUEndTime of commit number seq (seconds, mach uptime clock = time.perf_counter())
static double g_c[NSEQ];                                                 // commit time of seq, same clock (kyojin_mlx.gpuspan: round busy / idle)
static IMP g_orig_commit = nullptr;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
#define MAXD 128
static int16_t* g_disp = nullptr;                 // [NSEQ * MAXD] name ids, in encode order
static uint8_t* g_nd = nullptr;                   // dispatches recorded per commit number
static thread_local int16_t tl_pend[MAXD];
static thread_local int tl_n = 0;
static void take_pending(long seq) {              // the encoding thread's list -> this commit number
  if (!g_disp || !g_nd) return;
  int n = tl_n < MAXD ? tl_n : MAXD;
  int16_t* d = &g_disp[(seq % NSEQ) * MAXD];
  for (int i = 0; i < n; i++) d[i] = tl_pend[i];
  g_nd[seq % NSEQ] = (uint8_t)n;
  tl_n = 0;
}
static void h_commit(id self, SEL c) {
  long seq = __atomic_fetch_add(&g_commits, 1, __ATOMIC_RELAXED);
  take_pending(seq);
  g_c[seq % NSEQ] = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
  [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    pthread_mutex_lock(&g_mu); g_gpu_us += (cb.GPUEndTime - cb.GPUStartTime) * 1e6; g_done++;
    g_t0[seq % NSEQ] = cb.GPUStartTime; g_t1[seq % NSEQ] = cb.GPUEndTime; pthread_mutex_unlock(&g_mu);
  }];
  ((void (*)(id, SEL))g_orig_commit)(self, c);
}

// ---- kernel names per commit -------------------------------------------------------------------------------------------------------------
// One entry per compute dispatch: the Metal function name it ran. The encoder appends on the encoding thread; commit hands the list to the commit
// number the driver stamps, so gpuspan can price a dispatch by its own command buffer (MLX_MAX_OPS_PER_BUFFER=1) or split a multi-op buffer.
static pthread_mutex_t g_kmu = PTHREAD_MUTEX_INITIALIZER;
static NSMutableArray<NSString*>* g_names = nil;  // interned kernel names, 0..n-1

#define PMAP (1 << 14)
static uintptr_t g_pk[PMAP];                      // pipeline state pointer -> name id
static int16_t g_pv[PMAP];

static int name_intern(const char* s) {           // g_kmu held
  if (!s) s = "?";
  NSString* ns = [NSString stringWithUTF8String:s];
  if (!g_names) g_names = [NSMutableArray array];
  NSUInteger i = [g_names indexOfObject:ns];
  if (i == NSNotFound) { [g_names addObject:ns]; i = g_names.count - 1; }
  return (int)i;
}
static int pmap_get(uintptr_t k) {                // g_kmu held
  uintptr_t h = (k >> 4) & (PMAP - 1);
  for (int i = 0; i < PMAP; i++) {
    uintptr_t s = (h + i) & (PMAP - 1);
    if (g_pk[s] == k) return g_pv[s];
    if (g_pk[s] == 0) return -1;
  }
  return -1;
}
static void pmap_put(uintptr_t k, int v) {        // g_kmu held
  uintptr_t h = (k >> 4) & (PMAP - 1);
  for (int i = 0; i < PMAP; i++) {
    uintptr_t s = (h + i) & (PMAP - 1);
    if (g_pk[s] == 0 || g_pk[s] == k) { g_pk[s] = k; g_pv[s] = (int16_t)v; return; }
  }
}
static void note_pipeline(id p, id<MTLFunction> fn) {
  if (!p || !fn) return;
  const char* n = [[fn name] UTF8String];
  pthread_mutex_lock(&g_kmu);
  int idx = name_intern(n);
  pmap_put((uintptr_t)p, idx);
  pthread_mutex_unlock(&g_kmu);
}
static void pend_add(id p) {
  if (tl_n >= MAXD) return;
  uintptr_t k = (uintptr_t)p;
  pthread_mutex_lock(&g_kmu);
  int idx = pmap_get(k);
  if (idx < 0) {                                   // a pipeline built before the hooks were installed: its own label, if MLX set one
    const char* lab = [[(id<MTLComputePipelineState>)p label] UTF8String];
    idx = name_intern(lab && lab[0] ? lab : "?");
    pmap_put(k, idx);
  }
  pthread_mutex_unlock(&g_kmu);
  tl_pend[tl_n++] = (int16_t)idx;
}

static int g_rec = 0;                           // dispatch recording on (probe gpu_kern): pipeline names are always catalogued, dispatches only then
static IMP g_orig_pf = nullptr, g_orig_pd = nullptr, g_orig_pd2 = nullptr, g_orig_pdf = nullptr;
static IMP g_orig_setp = nullptr, g_orig_dtg = nullptr, g_orig_dt = nullptr;

static id h_pf(id self, SEL c, id<MTLFunction> fn, NSError** e) {
  id r = ((id (*)(id, SEL, id, NSError**))g_orig_pf)(self, c, fn, e);
  note_pipeline(r, fn);
  return r;
}
static id h_pd(id self, SEL c, MTLComputePipelineDescriptor* d, MTLPipelineOption o, id* refl, NSError** e) {
  id r = ((id (*)(id, SEL, id, MTLPipelineOption, id*, NSError**))g_orig_pd)(self, c, d, o, refl, e);
  note_pipeline(r, [d computeFunction]);
  return r;
}
static id h_pd2(id self, SEL c, MTLComputePipelineDescriptor* d, NSError** e) {
  id r = ((id (*)(id, SEL, id, NSError**))g_orig_pd2)(self, c, d, e);
  note_pipeline(r, [d computeFunction]);
  return r;
}
static id h_pdf(id self, SEL c, id<MTLComputePipelineState> base, id extra, NSError** e) {
  id r = ((id (*)(id, SEL, id, id, NSError**))g_orig_pdf)(self, c, base, extra, e);
  if (r) { pthread_mutex_lock(&g_kmu); int idx = name_intern([[(id<MTLComputePipelineState>)base label] UTF8String]); pmap_put((uintptr_t)r, idx); pthread_mutex_unlock(&g_kmu); }
  return r;
}
static void h_setp(id self, SEL c, id<MTLComputePipelineState> p) {
  if (g_rec) pend_add(p);
  ((void (*)(id, SEL, id))g_orig_setp)(self, c, p);
}
static void h_dtg(id self, SEL c, MTLSize g, MTLSize t) { ((void (*)(id, SEL, MTLSize, MTLSize))g_orig_dtg)(self, c, g, t); }
static void h_dt(id self, SEL c, MTLSize g, MTLSize t) { ((void (*)(id, SEL, MTLSize, MTLSize))g_orig_dt)(self, c, g, t); }

static void hook(Class cls, SEL sel, IMP fn, IMP* orig) {
  if (!cls || !sel) return;
  Method m = class_getInstanceMethod(cls, sel);
  if (!m) return;
  *orig = method_getImplementation(m);
  method_setImplementation(m, fn);
}

extern "C" {
void cb_install(void) {
  if (g_orig_commit) return;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  Method m = class_getInstanceMethod(object_getClass(cb), @selector(commit));
  g_orig_commit = method_getImplementation(m);
  method_setImplementation(m, (IMP)h_commit);
  g_disp = (int16_t*)calloc((size_t)NSEQ * MAXD, sizeof(int16_t));
  g_nd = (uint8_t*)calloc(NSEQ, 1);
  Class dc = object_getClass(dev);
  hook(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)h_pf, &g_orig_pf);
  hook(dc, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)h_pd, &g_orig_pd);
  hook(dc, @selector(newComputePipelineStateWithDescriptor:error:), (IMP)h_pd2, &g_orig_pd2);
  hook(dc, @selector(newComputePipelineStateWithAdditionalBinaryFunctions:error:), (IMP)h_pdf, &g_orig_pdf);
  id<MTLCommandBuffer> cb2 = [q commandBuffer];           // a second buffer: an encoder cannot be opened once the first is committed
  id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
  Class ec = object_getClass(enc);
  hook(ec, @selector(setComputePipelineState:), (IMP)h_setp, &g_orig_setp);
  hook(ec, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)h_dtg, &g_orig_dtg);
  hook(ec, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)h_dt, &g_orig_dt);
  [enc endEncoding];
  g_rec = 0;
  [cb commit];
  [cb2 commit];
  tl_n = 0;
}
long cb_commits(void) { return __atomic_load_n(&g_commits, __ATOMIC_RELAXED); }
// GPU microseconds of the command buffers that completed since the last reset (call after mx.synchronize / eval)
double cb_gpu_us(void) { pthread_mutex_lock(&g_mu); double v = g_gpu_us; pthread_mutex_unlock(&g_mu); return v; }
long cb_completed(void) { pthread_mutex_lock(&g_mu); long v = g_done; pthread_mutex_unlock(&g_mu); return v; }
// GPU start / end time of commit number seq (valid for the last 65536 commits, after the buffer completed)
void cb_commit_time(long seq, double* t) { *t = g_c[seq % NSEQ]; }
void cb_times(long seq, double* t0, double* t1) { pthread_mutex_lock(&g_mu); *t0 = g_t0[seq % NSEQ]; *t1 = g_t1[seq % NSEQ]; pthread_mutex_unlock(&g_mu); }
void cb_reset(void) { pthread_mutex_lock(&g_mu); g_gpu_us = 0; g_done = 0; pthread_mutex_unlock(&g_mu); }
void cb_rec(int on) { g_rec = on ? 1 : 0; if (!g_rec) tl_n = 0; }
// compute dispatches recorded for commit number seq, and the int'ed name of dispatch i (cb_kname copies the UTF8 name)
long cb_ndisp(long seq) { return g_nd ? g_nd[seq % NSEQ] : 0; }
int cb_disp(long seq, int i) { return (g_disp && i >= 0 && i < MAXD) ? g_disp[(seq % NSEQ) * MAXD + i] : -1; }
long cb_nnames(void) { return g_names ? (long)g_names.count : 0; }
int cb_kname(int id, char* buf, int n) {
  if (!g_names || id < 0 || id >= (int)g_names.count || n <= 0) return 0;
  const char* s = [[g_names objectAtIndex:(NSUInteger)id] UTF8String];
  int k = 0; while (s[k] && k < n - 1) { buf[k] = s[k]; k++; }
  buf[k] = 0; return k;
}
}
