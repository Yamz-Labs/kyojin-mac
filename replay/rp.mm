// Kyojin step replay: record the Metal calls MLX makes for one decode step, rebuild them as an indirect command buffer (ICB),
// execute the ICB from a one-launch trigger kernel with no per-launch CPU work. Swizzles the AGX encoder / device classes.
// build: clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation rp.mm -o librp.dylib
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <deque>

static std::unordered_map<std::string, IMP> g_orig;
static void swz(Class c, SEL s, IMP imp) {
  Method m = class_getInstanceMethod(c, s);
  if (!m) { fprintf(stderr, "rp: %s has no %s\n", class_getName(c), sel_getName(s)); return; }
  g_orig[std::string(class_getName(c)) + "/" + sel_getName(s)] = method_getImplementation(m);
  method_setImplementation(m, imp);
}
static Class g_enc, g_cb, g_dev, g_heap;
// The original IMPs of the `new...` family return an owned (+1) object. Called through a plain function-pointer cast ARC takes the result for +0 and retains it once more: every
// buffer and pipeline created through a hook would leak (and the hooks, returning to MLX which owns what a `new...` method returns, must hand over +1 as well).
#define RETAINED __attribute__((ns_returns_retained))
typedef RETAINED id (*NewFn_fn_err)(id, SEL, id, NSError**);
typedef RETAINED id (*NewFn_desc_opt)(id, SEL, id, NSUInteger, id*, NSError**);
typedef RETAINED id (*NewFn_len_opt)(id, SEL, NSUInteger, NSUInteger);
#define ORIG(c, sel) g_orig[std::string(class_getName(c)) + "/" + sel]

// ------------------------------------------------------------------ recording model
struct Bind { int idx; bool bytes; id buf; uint64_t off; uint32_t arena; uint32_t len; };
struct Cmd {
  id pso; bool threadgroups; MTLSize grid, tg; bool barrier;
  std::vector<Bind> binds; std::vector<std::pair<uint32_t, uint32_t>> tgmem;   // (index, length)
};
struct Rec {
  std::vector<Cmd> cmds; std::vector<uint8_t> arena; std::vector<id> keep; std::unordered_set<void*> born; std::unordered_set<void*> seen; bool snap_born = false; std::unordered_map<void*, std::vector<uint8_t>> snap, endsnap; int encoders = 0, fences = 0, others = 0;
};
static bool g_swallow = false;                                 // record only: the dispatches of a recording are not executed (a probe pass costs the CPU encode, no GPU work, no state written)
static std::deque<Rec> g_rec(64);                                // grows on demand (rp_rec_begin)
static int g_cur = -1;
static std::unordered_map<void*, std::string> g_psoname;      // pso -> function name
static bool g_flag_icb = true;                                 // create every pipeline with supportIndirectCommandBuffers
static std::vector<id> g_born_keep;
static id g_trig_pso = nil;
static bool g_trig_on = false;
static std::unordered_map<void*, int> g_region_pso;           // trigger pipeline -> region handle
static int g_trig_region = -1;                               // region of the trigger pipeline now bound (-1 none)
static int region_of_name(const char* n) { const char* p = strstr(n, "kyojin_region_trigger_"); return p ? atoi(p + 22) : -1; }
static std::map<uintptr_t, uintptr_t> g_pins;
// Session-wide first-bind state of the external (not born, not pinned) buffers: contents of the small ones (they become private clones), a hash of the big ones
// (big and unchanged by the end of the recorded session = read-only weights: bound in place, kept alive by the chain; big and changed = state we cannot clone: refused).
static const size_t kSmallMax = 1 << 20;
static std::unordered_map<void*, uint64_t> g_hash0;
static uint64_t hash_buf(const void* p, size_t n) {
  uint64_t h = 1469598103934665603ull; const uint64_t* w = (const uint64_t*)p; size_t m = n / 8;
  for (size_t i = 0; i < m; ++i) { h ^= w[i]; h *= 1099511628211ull; }
  for (size_t i = m * 8; i < n; ++i) { h ^= ((const uint8_t*)p)[i]; h *= 1099511628211ull; }
  return h;
}
static bool is_pinned(id<MTLBuffer> b) {
  uintptr_t c = (uintptr_t)[b contents], e = c + [b length];
  auto it = g_pins.upper_bound(c); if (it != g_pins.begin()) { auto p = std::prev(it); if (p->second > c && p->first < e) return true; }
  return it != g_pins.end() && it->first < e;
}

struct EncState { id pso = nil; Bind slot[31]; bool has[31] = {}; std::map<uint32_t, uint32_t> tgmem; bool barrier = false; } g_es;
static inline Rec& R() { return g_rec[g_cur]; }
static inline bool recording() { return g_cur >= 0; }

// ------------------------------------------------------------------ device hooks
RETAINED static id h_pso_fn(id self, SEL _cmd, id fn, NSError** err) {
  id p = ((NewFn_fn_err)ORIG(g_dev, "newComputePipelineStateWithFunction:error:"))(self, _cmd, fn, err);
  if (p) { g_psoname[(__bridge void*)p] = [[fn name] UTF8String]; if ([[fn name] containsString:@"kyojin_replay_trigger"]) g_trig_pso = p; int rh = region_of_name([[fn name] UTF8String]); if (rh >= 0) g_region_pso[(__bridge void*)p] = rh; }
  return p;
}
struct Access;
static void note_access(id p, id refl);
RETAINED static id h_pso_desc2(id self, SEL _cmd, MTLComputePipelineDescriptor* d, NSError** err) {
  if (g_flag_icb) d.supportIndirectCommandBuffers = YES;
  id __autoreleasing refl = nil;                                                    // the 4-argument variant gives the buffer access of the pipeline (hazard barriers of a region)
  id p = ((NewFn_desc_opt)ORIG(g_dev, "newComputePipelineStateWithDescriptor:options:reflection:error:"))(self, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), d, MTLPipelineOptionArgumentInfo | MTLPipelineOptionBindingInfo, &refl, err);
  if (p) note_access(p, refl);
  return p;
}
// per pipeline: access of each buffer index from the pipeline reflection (1 read-only, 2 may be written, 0 not an argument); absent = unknown, treated as may-write
struct Access { uint8_t a[32]; };
static std::unordered_map<void*, Access> g_access;
static bool rec_writes(Rec& r, void* key) {   // does any kernel of the recording have write access to this buffer (pipeline reflection; independent of execution)
  for (auto& c : r.cmds) { auto ac = g_access.find((__bridge void*)c.pso); for (auto& b : c.binds) if (!b.bytes && (__bridge void*)b.buf == key) { uint8_t a = ac == g_access.end() ? 2 : ac->second.a[b.idx]; if (a == 2) return true; } }
  return false;
}
static void note_access(id p, id refl) {
  if (!p || !refl) return;
  Access ac; memset(ac.a, 0, sizeof ac.a);
  for (id<MTLBinding> b in ((MTLComputePipelineReflection*)refl).bindings) if (b.type == MTLBindingTypeBuffer && b.index < 31) ac.a[b.index] = (b.access == MTLBindingAccessReadOnly) ? 1 : 2;
  g_access[(__bridge void*)p] = ac;
}
RETAINED static id h_pso_desc(id self, SEL _cmd, MTLComputePipelineDescriptor* d, NSUInteger opt, id* refl, NSError** err) {
  if (g_flag_icb) d.supportIndirectCommandBuffers = YES;
  id __autoreleasing mine = nil; id __autoreleasing* rp = refl ? refl : &mine;
  id p = ((NewFn_desc_opt)ORIG(g_dev, "newComputePipelineStateWithDescriptor:options:reflection:error:"))(self, _cmd, d, opt | MTLPipelineOptionArgumentInfo | MTLPipelineOptionBindingInfo, rp, err);
  if (p) note_access(p, *rp);
  if (p && d.computeFunction) { g_psoname[(__bridge void*)p] = [[d.computeFunction name] UTF8String]; if ([[d.computeFunction name] containsString:@"kyojin_replay_trigger"]) g_trig_pso = p; int rh = region_of_name([[d.computeFunction name] UTF8String]); if (rh >= 0) g_region_pso[(__bridge void*)p] = rh; }
  return p;
}
RETAINED static id h_heapbuf(id self, SEL _cmd, NSUInteger len, NSUInteger opt) {            // MLX allocates small arrays from a heap: they are born buffers too
  id b = ((NewFn_len_opt)ORIG(g_heap, "newBufferWithLength:options:"))(self, _cmd, len, opt);
  if (recording() && b) { R().born.insert((__bridge void*)b); R().keep.push_back(b); }
  return b;
}
RETAINED static id h_newbuf(id self, SEL _cmd, NSUInteger len, NSUInteger opt) {
  id b = ((NewFn_len_opt)ORIG(g_dev, "newBufferWithLength:options:"))(self, _cmd, len, opt);
  if (recording() && b) { R().born.insert((__bridge void*)b); R().keep.push_back(b); }
  return b;
}

// ------------------------------------------------------------------ replay objects and permanent (cheap) trigger state
struct Replay {
  id<MTLIndirectCommandBuffer> icb = nil; id<MTLBuffer> arena = nil; std::vector<id> own; id<MTLResidencySet> rs = nil;
  std::vector<std::pair<id, const std::vector<uint8_t>*>> refresh; int ncmds = 0; id<MTLBuffer> out_buf = nil; NSUInteger out_off = 0, out_bytes = 0; int out_slot = 1; long clone_bytes = 0, nclones = 0;
};
struct Chain { std::vector<Replay*> steps; size_t pos = 0; std::vector<id> own, pinned_bufs; std::vector<std::pair<id, const std::vector<uint8_t>*>> clones; id<MTLResidencySet> rs = nil; int out_slot = 1; long clone_bytes = 0, nclones = 0, bound_bytes = 0, nbound = 0; };
static std::vector<Chain*> g_chains;
static std::vector<id<MTLResidencySet>> g_all_rs;               // every residency set we own: attached to every MLX queue and command buffer
static Chain* g_active = nil;
static __unsafe_unretained id g_tb[32]; static NSUInteger g_to[32];
static id g_mlx_queue = nil;
static id<MTLComputePipelineState> g_copy_pso = nil;
static int g_use_res = getenv("RP_USE_RES") ? 1 : 0;
static long g_replays = 0; static long g_dbg[8] = {0};
static bool g_time_on = false; static std::vector<double> g_gpu_us; static std::vector<double> g_gpu_wall_us; static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

// ------------------------------------------------------------------ encoder hooks (record only while a recording is open)
#define E(sel) ORIG(g_enc, sel)
#define EC(sel) ([]{ static IMP i = ORIG(g_enc, sel); return i; }())     /* original IMP of an encoder method, looked up once per call site */
static void h_setPSO(id self, SEL c, id p) { if (recording()) g_es.pso = p; g_trig_on = (p == g_trig_pso) && g_trig_pso; if (!g_region_pso.empty()) { auto it = g_region_pso.find((__bridge void*)p); g_trig_region = it == g_region_pso.end() ? -1 : it->second; if (g_trig_region >= 0) memset((void*)g_tb, 0, sizeof g_tb); } ((void(*)(id, SEL, id))EC("setComputePipelineState:"))(self, c, p); }
static void h_setBuf(id self, SEL c, id b, NSUInteger o, NSUInteger i) {
  if ((g_trig_on || g_trig_region >= 0) && i < 32) { g_tb[i] = b; g_to[i] = o; }
  if (recording() && i < 31) {
    Rec& r = R(); void* key = (__bridge void*)b;
    if (r.seen.insert(key).second) {
      r.keep.push_back(b);
      if (!r.born.count(key) && !is_pinned((id<MTLBuffer>)b)) {
        id<MTLBuffer> mb = (id<MTLBuffer>)b; size_t n = [mb length];
        if (n <= kSmallMax) { std::vector<uint8_t>& v = r.snap[key]; v.resize(n); memcpy(v.data(), [mb contents], n); }
        else if (!g_hash0.count(key)) g_hash0[key] = hash_buf([mb contents], n);
      } else if (r.snap_born && r.born.count(key)) {
        id<MTLBuffer> mb = (id<MTLBuffer>)b; size_t n = [mb length]; if (n <= kSmallMax) { std::vector<uint8_t>& v = r.snap[key]; v.resize(n); memcpy(v.data(), [mb contents], n); }
      }
    }
    Bind& s = g_es.slot[i]; s.idx = i; s.bytes = false; s.buf = b; s.off = o; g_es.has[i] = true;
  }
  ((void(*)(id, SEL, id, NSUInteger, NSUInteger))EC("setBuffer:offset:atIndex:"))(self, c, b, o, i);
}
static void h_setBufs(id self, SEL c, const id* b, const NSUInteger* o, NSRange r) {
  if (recording()) { R().others++; fprintf(stderr, "rp: setBuffers (range) not recorded\n"); }
  ((void(*)(id, SEL, const id*, const NSUInteger*, NSRange))EC("setBuffers:offsets:withRange:"))(self, c, b, o, r);
}
static void h_setOff(id self, SEL c, NSUInteger o, NSUInteger i) {
  if (recording() && i < 31 && g_es.has[i]) g_es.slot[i].off = o;
  ((void(*)(id, SEL, NSUInteger, NSUInteger))EC("setBufferOffset:atIndex:"))(self, c, o, i);
}
static void h_setBytes(id self, SEL c, const void* p, NSUInteger l, NSUInteger i) {
  if (recording() && i < 31) {
    Rec& r = R(); size_t o = (r.arena.size() + 255) & ~(size_t)255; r.arena.resize(o + ((l + 15) & ~(size_t)15)); memcpy(r.arena.data() + o, p, l);
    Bind& s = g_es.slot[i]; s.idx = i; s.bytes = true; s.buf = nil; s.arena = (uint32_t)o; s.len = (uint32_t)l; g_es.has[i] = true;
  }
  ((void(*)(id, SEL, const void*, NSUInteger, NSUInteger))EC("setBytes:length:atIndex:"))(self, c, p, l, i);
}
static void h_setTG(id self, SEL c, NSUInteger l, NSUInteger i) { if (recording()) g_es.tgmem[(uint32_t)i] = (uint32_t)l; ((void(*)(id, SEL, NSUInteger, NSUInteger))EC("setThreadgroupMemoryLength:atIndex:"))(self, c, l, i); }
static void record_dispatch(bool tgs, MTLSize g, MTLSize t) {
  if (R().cmds.size() > 400000) { fprintf(stderr, "rp: a recording passed 400000 dispatches: closed (a recording left open would grow without bound)\n"); g_cur = -1; return; }
  Cmd cmd; cmd.pso = g_es.pso; cmd.threadgroups = tgs; cmd.grid = g; cmd.tg = t; cmd.barrier = g_es.barrier; g_es.barrier = false;
  for (int i = 0; i < 31; ++i) if (g_es.has[i]) cmd.binds.push_back(g_es.slot[i]);
  for (auto& kv : g_es.tgmem) cmd.tgmem.push_back(kv);
  R().cmds.push_back(std::move(cmd));
}
static void run_replay(id enc);
static void run_region(id enc, int h);
static bool region_ready(int h);
static void cb_commit_slots(id cb);
static void h_dT(id self, SEL c, MTLSize g, MTLSize t) { if (g_trig_region >= 0 && region_ready(g_trig_region)) { int h = g_trig_region; g_trig_region = -1; run_region(self, h); return; } if (g_trig_on && g_active) { g_trig_on = false; run_replay(self); return; } if (recording()) record_dispatch(false, g, t); if (g_swallow && recording()) return; ((void(*)(id, SEL, MTLSize, MTLSize))EC("dispatchThreads:threadsPerThreadgroup:"))(self, c, g, t); }
static void h_dTG(id self, SEL c, MTLSize g, MTLSize t) { if (g_trig_region >= 0 && region_ready(g_trig_region)) { int h = g_trig_region; g_trig_region = -1; run_region(self, h); return; } if (g_trig_on && g_active) { g_trig_on = false; run_replay(self); return; } if (recording()) record_dispatch(true, g, t); if (g_swallow && recording()) return; ((void(*)(id, SEL, MTLSize, MTLSize))EC("dispatchThreadgroups:threadsPerThreadgroup:"))(self, c, g, t); }
static void h_barS(id self, SEL c, NSUInteger s) { if (recording()) g_es.barrier = true; ((void(*)(id, SEL, NSUInteger))EC("memoryBarrierWithScope:"))(self, c, s); }
static void h_barR(id self, SEL c, const id* r, NSUInteger n) { if (recording()) g_es.barrier = true; ((void(*)(id, SEL, const id*, NSUInteger))EC("memoryBarrierWithResources:count:"))(self, c, r, n); }
static void h_upF(id self, SEL c, id f) { if (recording()) R().fences++; ((void(*)(id, SEL, id))EC("updateFence:"))(self, c, f); }
static void h_waF(id self, SEL c, id f) { if (recording()) R().fences++; ((void(*)(id, SEL, id))EC("waitForFence:"))(self, c, f); }
#define CB(sel) ORIG(g_cb, sel)
#define CBC(sel) ([]{ static IMP i = ORIG(g_cb, sel); return i; }())
// a new encoder starts after the work of the previous one: the first command of it carries a barrier
static id h_cenc(id self, SEL c, NSUInteger t) { if (recording()) { R().encoders++; g_es = EncState(); g_es.barrier = true; } return ((id(*)(id, SEL, NSUInteger))CBC("computeCommandEncoderWithDispatchType:"))(self, c, t); }
static long g_commits = 0;
static void h_commit(id self, SEL c) {
  g_commits++;
  cb_commit_slots(self);
  if (g_time_on) {
    double t0 = CFAbsoluteTimeGetCurrent() * 1e6;
    [(id<MTLCommandBuffer>)self addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      pthread_mutex_lock(&g_mu); g_gpu_us.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e6); g_gpu_wall_us.push_back(CFAbsoluteTimeGetCurrent() * 1e6 - t0); pthread_mutex_unlock(&g_mu);
    }];
  }
  ((void(*)(id, SEL))CBC("commit"))(self, c);
}
static std::vector<id> g_queues;
static id h_qcbu(id self, SEL c) {
  g_mlx_queue = self; bool known = false; for (id q : g_queues) if (q == self) known = true;
  if (!known) { g_queues.push_back(self); if (getenv("RP_DEBUG")) fprintf(stderr, "rp: new MLX queue %p (now %zu)\n", (__bridge void*)self, g_queues.size()); for (id<MTLResidencySet> rs : g_all_rs) [(id<MTLCommandQueue>)self addResidencySet:rs]; }
  id cb = ((id(*)(id, SEL))ORIG(object_getClass(self), "commandBufferWithUnretainedReferences"))(self, c);
  for (id<MTLResidencySet> rs : g_all_rs) [(id<MTLCommandBuffer>)cb useResidencySet:rs];      // the set is resident for every command buffer, whatever queue it comes from
  return cb;
}
static id h_benc(id self, SEL c) { if (recording()) R().others++; return ((id(*)(id, SEL))CB("blitCommandEncoder"))(self, c); }


// ------------------------------------------------------------------ replay execution (called from the trigger kernel's dispatch hook, on the live MLX encoder)
static void run_replay(id enc) {
  Replay* r = g_active->steps[g_active->pos]; g_active->pos = (g_active->pos + 1) % g_active->steps.size(); g_replays++; const int out_slot = g_active->out_slot;
  id<MTLComputeCommandEncoder> e = (id<MTLComputeCommandEncoder>)enc;
  auto barrier = [&]() { ((void(*)(id, SEL, NSUInteger))EC("memoryBarrierWithScope:"))(enc, @selector(memoryBarrierWithScope:), MTLBarrierScopeBuffers); };
  for (auto& rf : r->refresh) { id<MTLBuffer> b = (id<MTLBuffer>)rf.first; memcpy([b contents], rf.second->data(), [b length]); }   // inputs the CPU wrote between the recorded steps (the previous replay has completed: synchronous use)
  if (g_use_res) { static std::vector<id<MTLResource>> rv; if (rv.empty()) { for (id b : g_active->own) rv.push_back((id<MTLResource>)b); for (Replay* q : g_active->steps) { rv.push_back(q->arena); rv.push_back(q->icb); } } [e useResources:(const id<MTLResource> __unsafe_unretained*)rv.data() count:rv.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite]; }
  barrier();
  [e executeCommandsInBuffer:r->icb withRange:NSMakeRange(0, r->ncmds)];
  barrier();
  if (getenv("RP_DEBUG")) fprintf(stderr, "run_replay: out_buf %p tb[slot] %p to %lu in tb[0] %p\n", (__bridge void*)r->out_buf, (__bridge void*)g_tb[out_slot], (unsigned long)g_to[out_slot], (__bridge void*)g_tb[0]);
  g_dbg[0]++; if (!g_tb[out_slot]) g_dbg[1]++; if (!r->out_buf) g_dbg[2]++; g_dbg[3] = (long)(uintptr_t)(__bridge void*)g_tb[out_slot]; g_dbg[4] = (long)g_to[out_slot];
  if (r->out_buf && g_tb[out_slot]) {                    // persistent output -> the trigger op's own output buffer (one tiny copy launch)
    uint32_t nwords = (uint32_t)(r->out_bytes / 4);
    ((void(*)(id, SEL, id))EC("setComputePipelineState:"))(enc, @selector(setComputePipelineState:), g_copy_pso);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))EC("setBuffer:offset:atIndex:"))(enc, @selector(setBuffer:offset:atIndex:), r->out_buf, r->out_off, 0);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))EC("setBuffer:offset:atIndex:"))(enc, @selector(setBuffer:offset:atIndex:), g_tb[out_slot], g_to[out_slot], 1);
    ((void(*)(id, SEL, const void*, NSUInteger, NSUInteger))EC("setBytes:length:atIndex:"))(enc, @selector(setBytes:length:atIndex:), &nwords, 4, 2);
    ((void(*)(id, SEL, MTLSize, MTLSize))EC("dispatchThreads:threadsPerThreadgroup:"))(enc, @selector(dispatchThreads:threadsPerThreadgroup:), MTLSizeMake(nwords, 1, 1), MTLSizeMake(256, 1, 1));
  }
}

// ------------------------------------------------------------------ static regions (live replay of a pure function with fixed shapes)
// A region is a function whose command stream does not depend on its inputs' values (fixed shapes, array inputs only). Recorded once from an eager run, rebuilt as an ICB;
// each call is a one-launch trigger op (inputs = the function's arrays, outputs = its results): the hook rebinds the ICB commands that read the inputs to the buffers MLX gave
// the trigger, runs the ICB on private scratch, and copies the results into the trigger's output buffers. No state is kept between calls, so any data flow is just MLX arrays.
struct Patch { int cmd, slot, in; uint64_t delta; };
struct OutCopy { bool from_in; id buf; uint64_t off; int in; uint64_t nbytes; };
struct RSlot { id<MTLIndirectCommandBuffer> icb = nil; id<MTLBuffer> arena = nil; std::atomic<int> busy{0}; std::vector<id<MTLIndirectComputeCommand>> pic, allic; std::vector<id> hb; };
// Parametrised region: fields of the recorded stream (grid / threadgroup, arena words, host-constant buffer words) that are periodic-affine in a host integer p (the position of the call),
// v(p) = T[(p - p0) mod L] + slope * ((p - p0) div L), written into the slot's own arena / host buffers / ICB commands before every execute.
struct Fld { int kind; int cmd; int axis; uint64_t off; int hbi; std::vector<int64_t> T; int64_t slope; };   // kind 0 grid, 1 threadgroup, 2 arena word, 3 host buffer word
struct HostBuf { int cmd, idx; size_t size; id priv; std::vector<uint8_t> init; };
struct Region {
  bool built = false, freed = false; int n_in = 0, n_out = 0, ncmds = 0; Rec rec; std::unordered_map<void*, id> mp; std::vector<Patch> patches; std::vector<OutCopy> outs;
  std::vector<RSlot*> slots; std::vector<id> own; std::vector<id> priv;               // own: every buffer the region binds; priv: the private scratch (ours alone)
  id<MTLResidencySet> rs = nil; long replays = 0, rebinds = 0, extra_slots = 0, clone_bytes = 0, nclones = 0, nbound = 0;
  std::unordered_map<void*, uint64_t> ext_hash;                          // debug: hash of the first 4 MB of every buffer bound in place at build time
  std::vector<char> bar;                                               // per command: barrier before it (MLX's own + the hazards the eager order hid)
  long n_mlx_bar = 0, n_extra_bar = 0;
  std::unordered_map<uint64_t, int> patch_of;                          // (cmd << 8 | slot) -> index in patches
  std::vector<Fld> flds; std::vector<HostBuf> hbs; int fit_L = 0, param_in = -1; long p0 = 0, n_written = 0; std::vector<id> priv_scratch; std::vector<std::string> priv_info; std::vector<std::pair<id, std::vector<uint8_t>>> priv_init;
};
static std::vector<Region*> g_regions;
static id<MTLResidencySet> g_region_rs = nil;                          // ONE residency set for all regions (a command buffer takes a bounded number of sets)
static std::mutex g_slot_mu; static std::vector<RSlot*> g_cb_slots;      // slots used by the command buffer being encoded: busy until it completes
static id<MTLComputePipelineState> g_copyb_pso = nil;
static bool region_ready(int h) {
  if (h < 0 || h >= (int)g_regions.size() || !g_regions[h]) return false;
  if (g_regions[h]->freed) { fprintf(stderr, "rp: the trigger of a released region %d was evaluated (a lazy result of a dropped drafter): refusing to run garbage\n", h); abort(); }
  return g_regions[h]->built;
}
static void cb_commit_slots(id cb) {
  std::vector<RSlot*> v; { std::lock_guard<std::mutex> l(g_slot_mu); if (g_cb_slots.empty()) return; v.swap(g_cb_slots); }
  [(id<MTLCommandBuffer>)cb addCompletedHandler:^(id<MTLCommandBuffer>) { for (RSlot* s : v) s->busy.store(0); }];
}
static uintptr_t buf_ptr(id b, uint64_t off) { return (uintptr_t)[(id<MTLBuffer>)b contents] + off; }
static RSlot* make_slot(Region* R) {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  auto newbuf = [&](NSUInteger len) { return ((NewFn_len_opt)ORIG(g_dev, "newBufferWithLength:options:"))(dev, @selector(newBufferWithLength:options:), len, MTLResourceStorageModeShared); };
  RSlot* s = new RSlot(); Rec& r = R->rec;
  MTLIndirectCommandBufferDescriptor* d = [MTLIndirectCommandBufferDescriptor new];
  d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
  d.inheritBuffers = NO; d.inheritPipelineState = NO; d.maxKernelBufferBindCount = 31;
  s->arena = newbuf(r.arena.size() + 256); if (!r.arena.empty()) memcpy([s->arena contents], r.arena.data(), r.arena.size());
  s->icb = [dev newIndirectCommandBufferWithDescriptor:d maxCommandCount:r.cmds.size() options:MTLResourceStorageModeShared];
  if (!s->icb) { fprintf(stderr, "rp region: ICB allocation failed (%zu commands)\n", r.cmds.size()); delete s; return nullptr; }
  s->pic.assign(R->patches.size(), nil);
  for (size_t i = 0; i < r.cmds.size(); ++i) {
    Cmd& c = r.cmds[i]; id<MTLIndirectComputeCommand> ic = [s->icb indirectComputeCommandAtIndex:i];
    [ic setComputePipelineState:(id<MTLComputePipelineState>)c.pso];
    s->allic.push_back(ic);
    for (auto& b : c.binds) {
      if (b.bytes) { [ic setKernelBuffer:s->arena offset:b.arena atIndex:b.idx]; continue; }
      { int hi = -1; for (size_t q = 0; q < R->hbs.size(); ++q) if (R->hbs[q].priv && R->hbs[q].priv == b.buf) { hi = (int)q; break; }
        if (hi >= 0) { if (s->hb.size() < R->hbs.size()) s->hb.resize(R->hbs.size(), nil);
          if (!s->hb[hi]) { s->hb[hi] = newbuf([(id<MTLBuffer>)R->hbs[hi].priv length]); memcpy([s->hb[hi] contents], [(id<MTLBuffer>)R->hbs[hi].priv contents], [(id<MTLBuffer>)R->hbs[hi].priv length]); }
          [ic setKernelBuffer:(id<MTLBuffer>)s->hb[hi] offset:b.off atIndex:b.idx]; continue; } }
      auto pi = R->patch_of.find(((uint64_t)i << 8) | (uint64_t)b.idx);
      if (pi != R->patch_of.end()) { s->pic[pi->second] = ic; [ic setKernelBuffer:(id<MTLBuffer>)b.buf offset:b.off atIndex:b.idx]; }   // rebound at every call
      else [ic setKernelBuffer:(id<MTLBuffer>)b.buf offset:b.off atIndex:b.idx];                     // resolved at the build: private scratch or the external buffer itself
    }
    for (auto& t : c.tgmem) [ic setThreadgroupMemoryLength:t.second atIndex:t.first];
    if (R->bar[i] || getenv("RP_SERIAL")) [ic setBarrier];
    if (c.threadgroups) [ic concurrentDispatchThreadgroups:c.grid threadsPerThreadgroup:c.tg]; else [ic concurrentDispatchThreads:c.grid threadsPerThreadgroup:c.tg];
  }
  if (g_region_rs && R->built) { for (id h_ : s->hb) if (h_) [g_region_rs addAllocation:(id<MTLAllocation>)h_]; [g_region_rs addAllocation:s->arena]; [g_region_rs addAllocation:s->icb]; [g_region_rs commit]; [g_region_rs requestResidency]; }     // slots made after the build
  return s;
}
static RSlot* acquire_slot(Region* R) {
  std::lock_guard<std::mutex> l(g_slot_mu);
  for (RSlot* s : R->slots) { int z = 0; if (s->busy.compare_exchange_strong(z, 1)) { g_cb_slots.push_back(s); return s; } }
  RSlot* s = make_slot(R); if (!s) return nullptr;
  R->slots.push_back(s); R->extra_slots++; s->busy.store(1); g_cb_slots.push_back(s); return s;
}
static void apply_params(Region* R, RSlot* s, int h, int32_t pv) {
  {
    long k = (long)pv - R->p0;
    if (k < 0) { fprintf(stderr, "rp region %d: parameter %d below the first sample %ld\n", h, pv, R->p0); abort(); }
    long L = R->fit_L; std::unordered_map<int, std::array<int64_t, 6>> gt; std::unordered_set<int> dirty;
    for (size_t q = 0; q < R->hbs.size(); ++q) if (q < s->hb.size() && s->hb[q]) memcpy([(id<MTLBuffer>)s->hb[q] contents], R->hbs[q].init.data(), R->hbs[q].init.size());   // donated host constants are rewritten by the kernels: restore the sample-0 image first
    for (Fld& f : R->flds) {
      int64_t v = f.T[k % L] + f.slope * (k / L);
      { static int skip = getenv("RP_SKIP") ? atoi(getenv("RP_SKIP")) : 0; if (skip & (1 << f.kind)) continue; static int noslope = getenv("RP_NOSLOPE") != nullptr; if (noslope) v = f.T[k % L]; }
      if (f.kind == 2) { int32_t w = (int32_t)v; memcpy((uint8_t*)[s->arena contents] + f.off, &w, 4); }
      else if (f.kind == 3) { if (f.hbi >= (int)s->hb.size() || !s->hb[f.hbi]) continue; int32_t w = (int32_t)v; memcpy((uint8_t*)[(id<MTLBuffer>)s->hb[f.hbi] contents] + f.off, &w, 4); }
      else { auto it = gt.find(f.cmd); if (it == gt.end()) { Cmd& c = R->rec.cmds[f.cmd]; it = gt.emplace(f.cmd, std::array<int64_t, 6>{(int64_t)c.grid.width, (int64_t)c.grid.height, (int64_t)c.grid.depth, (int64_t)c.tg.width, (int64_t)c.tg.height, (int64_t)c.tg.depth}).first; }
        it->second[(f.kind == 1 ? 3 : 0) + f.axis] = v; dirty.insert(f.cmd); }
    }
    for (int ci : dirty) { auto& a = gt[ci]; Cmd& c = R->rec.cmds[ci]; id<MTLIndirectComputeCommand> ic = s->allic[ci]; MTLSize g = MTLSizeMake(a[0], a[1], a[2]), t = MTLSizeMake(a[3], a[4], a[5]);
      [ic reset];                                                                                   // re-encode the whole command: pipeline, bindings, barrier, new size
      [ic setComputePipelineState:(id<MTLComputePipelineState>)c.pso];
      for (auto& b : c.binds) {
        if (b.bytes) { [ic setKernelBuffer:s->arena offset:b.arena atIndex:b.idx]; continue; }
        int hi = -1; for (size_t q = 0; q < R->hbs.size(); ++q) if (R->hbs[q].priv && R->hbs[q].priv == b.buf) { hi = (int)q; break; }
        if (hi >= 0) { [ic setKernelBuffer:(id<MTLBuffer>)s->hb[hi] offset:b.off atIndex:b.idx]; continue; }
        auto pi = R->patch_of.find(((uint64_t)ci << 8) | (uint64_t)b.idx);
        if (pi != R->patch_of.end()) { [ic setKernelBuffer:(id<MTLBuffer>)g_tb[R->patches[pi->second].in] offset:g_to[R->patches[pi->second].in] + R->patches[pi->second].delta atIndex:b.idx]; }
        else [ic setKernelBuffer:(id<MTLBuffer>)b.buf offset:b.off atIndex:b.idx];
      }
      for (auto& tm : c.tgmem) [ic setThreadgroupMemoryLength:tm.second atIndex:tm.first];
      if (R->bar[ci] || getenv("RP_SERIAL")) [ic setBarrier];
      if (c.threadgroups) [ic concurrentDispatchThreadgroups:g threadsPerThreadgroup:t]; else [ic concurrentDispatchThreads:g threadsPerThreadgroup:t]; }
  }
}
static int g_prefix_n = -1, g_poison_dyn = -1, g_pz_lo = 0, g_pz_hi = 0, g_hz_lo = 0, g_hz_hi = 0;
static void run_region(id enc, int h) {
  if (recording()) fprintf(stderr, "rp: a region replay was evaluated inside a recording (its stream is not part of the recorded region)\n");
  Region* R = g_regions[h];
  for (int i = 0; i < R->n_in + R->n_out; ++i) if (!g_tb[i]) { fprintf(stderr, "rp region %d: trigger buffer %d not bound (inputs and outputs must all be arguments of the stub)\n", h, i); abort(); }
  RSlot* s = acquire_slot(R);
  if (!s) { fprintf(stderr, "rp region %d: no ICB slot\n", h); abort(); }
  id<MTLComputeCommandEncoder> e = (id<MTLComputeCommandEncoder>)enc;
  auto barrier = [&]() { ((void(*)(id, SEL, NSUInteger))EC("memoryBarrierWithScope:"))(enc, @selector(memoryBarrierWithScope:), MTLBarrierScopeBuffers); };
  if (!R->priv_init.empty()) { for (auto& pi : R->priv_init) { id<MTLBuffer> pb = (id<MTLBuffer>)pi.first; if (pi.second.empty()) memset([pb contents], 0, [pb length]); else { memcpy([pb contents], pi.second.data(), pi.second.size()); } } }
  if (g_pz_hi > g_pz_lo) { int ix = 0; for (id pb : R->priv) { bool hbx = false; for (auto& hb : R->hbs) if (hb.priv && hb.priv == pb) hbx = true; if (hbx) continue; if (ix >= g_pz_lo && ix < g_pz_hi) memset([(id<MTLBuffer>)pb contents], 0, [(id<MTLBuffer>)pb length]); ix++; } }
  if (g_poison_dyn == 2) { for (id pb : R->priv) { bool hbx = false; for (auto& hb : R->hbs) if (hb.priv && hb.priv == pb) hbx = true; if (!hbx) memset([(id<MTLBuffer>)pb contents], 0, [(id<MTLBuffer>)pb length]); } }
  else if (g_poison_dyn >= 0) { for (id pb : R->priv_scratch) memset([(id<MTLBuffer>)pb contents], g_poison_dyn ? 0xFF : 0x00, [(id<MTLBuffer>)pb length]); }
  if (getenv("RP_POISON")) { static int val = atoi(getenv("RP_POISON")); static long lo = getenv("RP_POISON_LO") ? atol(getenv("RP_POISON_LO")) : 0, hi = getenv("RP_POISON_HI") ? atol(getenv("RP_POISON_HI")) : 1 << 30; long ix = 0; for (id pb : R->priv_scratch) { if (ix >= lo && ix < hi) memset([(id<MTLBuffer>)pb contents], val ? 0xFF : 0x00, [(id<MTLBuffer>)pb length]); ix++; } if (getenv("RP_POISON_N") && R->replays == 0) for (size_t q = lo; q < std::min<size_t>(hi, R->priv_info.size()) && q < lo + 3; ++q) fprintf(stderr, "rp: poisoned scratch %zu: %s\n", q, R->priv_info[q].c_str());
    if (getenv("RP_POISON_N") && R->replays == 0) fprintf(stderr, "rp: %zu scratch buffers poisonable\n", R->priv_scratch.size()); }
  if (!R->flds.empty()) apply_params(R, s, h, *(int32_t*)((uint8_t*)[(id<MTLBuffer>)g_tb[R->param_in] contents] + g_to[R->param_in]));
  if (g_hz_hi > g_hz_lo) for (int q = g_hz_lo; q < g_hz_hi && q < (int)s->hb.size(); ++q) if (s->hb[q]) memset([(id<MTLBuffer>)s->hb[q] contents], 0, [(id<MTLBuffer>)s->hb[q] length]);
  for (size_t p = 0; p < R->patches.size(); ++p) { Patch& pt = R->patches[p]; [s->pic[p] setKernelBuffer:(id<MTLBuffer>)g_tb[pt.in] offset:g_to[pt.in] + pt.delta atIndex:pt.slot]; }
  R->rebinds += R->patches.size(); R->replays++;
  if (getenv("RP_DEBUG") && R->replays <= 2) { for (int i = 0; i < R->n_in + R->n_out; ++i) fprintf(stderr, "rp region %d replay %ld: trigger buf %d = %p + %lu -> ptr %lx\n", h, R->replays, i, (__bridge void*)g_tb[i], (unsigned long)g_to[i], (unsigned long)buf_ptr(g_tb[i], g_to[i])); }
  for (int i = 0; i < R->n_in + R->n_out; ++i) [e useResource:(id<MTLBuffer>)g_tb[i] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  barrier();
  [e executeCommandsInBuffer:s->icb withRange:NSMakeRange(0, g_prefix_n >= 0 ? std::min<int>(g_prefix_n, R->ncmds) : R->ncmds)];
  barrier();
  for (int j = 0; j < R->n_out; ++j) {
    OutCopy& oc = R->outs[j]; id src = oc.from_in ? g_tb[oc.in] : oc.buf; uint64_t soff = oc.from_in ? g_to[oc.in] + oc.off : oc.off;
    if (!oc.from_in) for (size_t q = 0; q < R->hbs.size(); ++q) if (R->hbs[q].priv && R->hbs[q].priv == oc.buf && q < s->hb.size() && s->hb[q]) { src = s->hb[q]; break; }   // a host buffer: the slot copy is the one the ICB read
    id dst = g_tb[R->n_in + j]; uint64_t doff = g_to[R->n_in + j];
    bool w4 = (oc.nbytes % 4) == 0; uint32_t n = (uint32_t)(w4 ? oc.nbytes / 4 : oc.nbytes);
    ((void(*)(id, SEL, id))EC("setComputePipelineState:"))(enc, @selector(setComputePipelineState:), w4 ? g_copy_pso : g_copyb_pso);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))EC("setBuffer:offset:atIndex:"))(enc, @selector(setBuffer:offset:atIndex:), src, soff, 0);
    ((void(*)(id, SEL, id, NSUInteger, NSUInteger))EC("setBuffer:offset:atIndex:"))(enc, @selector(setBuffer:offset:atIndex:), dst, doff, 1);
    ((void(*)(id, SEL, const void*, NSUInteger, NSUInteger))EC("setBytes:length:atIndex:"))(enc, @selector(setBytes:length:atIndex:), &n, 4, 2);
    ((void(*)(id, SEL, MTLSize, MTLSize))EC("dispatchThreads:threadsPerThreadgroup:"))(enc, @selector(dispatchThreads:threadsPerThreadgroup:), MTLSizeMake(n, 1, 1), MTLSizeMake(std::min<uint32_t>(n, 256), 1, 1));
  }
}

// ------------------------------------------------------------------ API
extern "C" {
static void rp_install_impl(int flag_icb) {
  g_flag_icb = flag_icb;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  g_dev = object_getClass(dev); g_cb = object_getClass(cb); g_enc = object_getClass(e);
  [e endEncoding]; [cb commit];
  swz(g_dev, @selector(newComputePipelineStateWithFunction:error:), (IMP)h_pso_fn);
  swz(g_dev, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)h_pso_desc);
  swz(g_dev, @selector(newComputePipelineStateWithDescriptor:error:), (IMP)h_pso_desc2);
  swz(g_dev, @selector(newBufferWithLength:options:), (IMP)h_newbuf);
  { MTLHeapDescriptor* hd = [MTLHeapDescriptor new]; hd.size = 65536; hd.storageMode = MTLStorageModeShared; id<MTLHeap> hp = [dev newHeapWithDescriptor:hd]; g_heap = object_getClass(hp);
    swz(g_heap, @selector(newBufferWithLength:options:), (IMP)h_heapbuf); }
  swz(g_enc, @selector(setComputePipelineState:), (IMP)h_setPSO);
  swz(g_enc, @selector(setBuffer:offset:atIndex:), (IMP)h_setBuf);
  swz(g_enc, @selector(setBuffers:offsets:withRange:), (IMP)h_setBufs);
  swz(g_enc, @selector(setBufferOffset:atIndex:), (IMP)h_setOff);
  swz(g_enc, @selector(setBytes:length:atIndex:), (IMP)h_setBytes);
  swz(g_enc, @selector(setThreadgroupMemoryLength:atIndex:), (IMP)h_setTG);
  swz(g_enc, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)h_dT);
  swz(g_enc, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)h_dTG);
  swz(g_enc, @selector(memoryBarrierWithScope:), (IMP)h_barS);
  swz(g_enc, @selector(memoryBarrierWithResources:count:), (IMP)h_barR);
  swz(g_enc, @selector(updateFence:), (IMP)h_upF);
  swz(g_enc, @selector(waitForFence:), (IMP)h_waF);
  swz(g_cb, @selector(computeCommandEncoderWithDispatchType:), (IMP)h_cenc);
  swz(g_cb, @selector(blitCommandEncoder), (IMP)h_benc);
  swz(g_cb, @selector(commit), (IMP)h_commit);
  swz(object_getClass(q), @selector(commandBufferWithUnretainedReferences), (IMP)h_qcbu);
  NSError* er = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void rp_copy(device const uint* s [[buffer(0)]], device uint* d [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) d[i] = s[i]; }\n" options:nil error:&er];
  g_copy_pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rp_copy"] error:&er];
  id<MTLLibrary> lib2 = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void rp_copyb(device const uchar* s [[buffer(0)]], device uchar* d [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) d[i] = s[i]; }\n" options:nil error:&er];
  g_copyb_pso = [dev newComputePipelineStateWithFunction:[lib2 newFunctionWithName:@"rp_copyb"] error:&er];
}

static id<MTLBuffer> find_buf(const std::unordered_map<void*, id>& mp, uintptr_t ptr, NSUInteger* off) {
  for (auto& kv : mp) { id<MTLBuffer> ob = (__bridge id<MTLBuffer>)kv.first; uintptr_t c = (uintptr_t)[ob contents]; if (ptr >= c && ptr < c + [ob length]) { *off = ptr - c; return (id<MTLBuffer>)kv.second; } }
  return nil;
}
// Build one ICB per recording in `slots` (a chain of consecutive decode steps) from recordings that share one buffer identity table: the state a step writes is the
// state the next one reads. Buffers whose contents lie in a pinned range (model weights, held arrays) are bound as they are; every other buffer (born in a step or a
// per-step external) is replaced by a private copy owned by the chain: initialised from the snapshot taken when the earliest recording bound it as an external (the
// pre-step contents), zero when the earliest recording allocated it. out_ptrs[i] / out_bytes: the output array of step i (copied back by the trigger).
static int rp_build_chain_impl(int n, const int* slots, const uintptr_t* out_ptrs, uintptr_t out_bytes, int out_slot) {
  NSError* err = nil;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  Chain* ch = new Chain(); ch->out_slot = out_slot;
  std::unordered_map<void*, id> mp;                                  // recorded buffer -> bound buffer
  auto newbuf = [&](NSUInteger len) { return ((NewFn_len_opt)ORIG(g_dev, "newBufferWithLength:options:"))(dev, @selector(newBufferWithLength:options:), len, MTLResourceStorageModeShared); };
  std::unordered_map<void*, const std::vector<uint8_t>*> last_end;     // buffer -> its contents at the end of the latest recorded step that bound it
  std::vector<std::vector<std::pair<id, const std::vector<uint8_t>*>>> refreshes(n);
  for (int si = 0; si < n; ++si) {
    Rec& r = g_rec[slots[si]]; std::unordered_set<void*> bound_here;
    for (auto& c : r.cmds) for (auto& b : c.binds) if (!b.bytes) {
      void* key = (__bridge void*)b.buf; id<MTLBuffer> ob = (id<MTLBuffer>)b.buf;
      if (!bound_here.insert(key).second) continue;
      auto m = mp.find(key);
      if (m != mp.end()) {
        // already mapped by an earlier step: state flows on the GPU when the contents at the start of this step equal the end of the earlier one; otherwise the CPU (or the allocator) rewrote it in between: refresh at replay
        auto sn = r.snap.find(key); auto le = last_end.find(key);
        if (sn != r.snap.end() && le != last_end.end() && le->second->size() == sn->second.size() && memcmp(le->second->data(), sn->second.data(), sn->second.size()) != 0) refreshes[si].push_back({m->second, &sn->second});
        continue;
      }
      if (is_pinned(ob)) { mp[key] = ob; ch->own.push_back(ob); ch->pinned_bufs.push_back(ob); continue; }
      auto h0 = g_hash0.find(key);
      if (h0 != g_hash0.end()) {
        if (hash_buf([ob contents], [ob length]) != h0->second) { fprintf(stderr, "rp_build_chain: big buffer (%lu B) changed during the recorded steps and has no snapshot\n", (unsigned long)[ob length]); return -5; }
        mp[key] = ob; ch->own.push_back(ob); ch->bound_bytes += [ob length]; ch->nbound++; continue;   // read-only weights: bound in place
      }
      id<MTLBuffer> nb = newbuf([ob length]);
      auto sn = r.snap.find(key);
      const std::vector<uint8_t>* init = sn != r.snap.end() ? &sn->second : nullptr;
      if (init) memcpy([nb contents], init->data(), [ob length]); else memset([nb contents], 0, [ob length]);
      mp[key] = nb; ch->own.push_back(nb); ch->clones.push_back({nb, init}); ch->clone_bytes += [ob length]; ch->nclones++;
    }
    for (void* key : bound_here) { auto e = r.endsnap.find(key); if (e != r.endsnap.end()) last_end[key] = &e->second; }
  }
  MTLIndirectCommandBufferDescriptor* d = [MTLIndirectCommandBufferDescriptor new];
  d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
  d.inheritBuffers = NO; d.inheritPipelineState = NO; d.maxKernelBufferBindCount = 31;
  int bad = 0;
  for (int si = 0; si < n; ++si) {
    Rec& r = g_rec[slots[si]]; Replay* rp = new Replay(); rp->ncmds = (int)r.cmds.size(); rp->refresh = refreshes[si];
    if (out_ptrs[si]) { NSUInteger off = 0; rp->out_buf = find_buf(mp, out_ptrs[si], &off); rp->out_off = off; rp->out_bytes = out_bytes; if (!rp->out_buf) { fprintf(stderr, "rp_build_chain: output of step %d not found among recorded buffers\n", si); return -2; } }
    rp->arena = newbuf(r.arena.size() + 256);
    if (!r.arena.empty()) memcpy([rp->arena contents], r.arena.data(), r.arena.size());
    rp->icb = [dev newIndirectCommandBufferWithDescriptor:d maxCommandCount:r.cmds.size() options:MTLResourceStorageModeShared];
    if (!rp->icb) { fprintf(stderr, "rp_build_chain: ICB allocation failed (%zu commands)\n", r.cmds.size()); return -1; }
    for (size_t i = 0; i < r.cmds.size(); ++i) {
      Cmd& c = r.cmds[i]; id<MTLIndirectComputeCommand> ic = [rp->icb indirectComputeCommandAtIndex:i];
      id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>)c.pso;
      if (!pso.supportIndirectCommandBuffers) bad++;
      [ic setComputePipelineState:pso];
      for (auto& b : c.binds) {
        if (b.bytes) [ic setKernelBuffer:rp->arena offset:b.arena atIndex:b.idx];
        else [ic setKernelBuffer:(id<MTLBuffer>)mp[(__bridge void*)b.buf] offset:b.off atIndex:b.idx];
      }
      for (auto& t : c.tgmem) [ic setThreadgroupMemoryLength:t.second atIndex:t.first];
      if (c.barrier) [ic setBarrier];
      if (c.threadgroups) [ic concurrentDispatchThreadgroups:c.grid threadsPerThreadgroup:c.tg]; else [ic concurrentDispatchThreads:c.grid threadsPerThreadgroup:c.tg];
    }
    ch->steps.push_back(rp);
  }
  if (bad) fprintf(stderr, "rp_build_chain: %d commands use pipelines without supportIndirectCommandBuffers\n", bad);
  // residency: one set for everything the chain owns, attached to the MLX queue
  MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new]; rd.initialCapacity = ch->own.size() + 2 * n + 4;
  ch->rs = [dev newResidencySetWithDescriptor:rd error:&err];
  if (!ch->rs) { fprintf(stderr, "rp_build_chain: residency set failed %s\n", err.description.UTF8String); return -3; }
  for (id b : ch->own) [ch->rs addAllocation:(id<MTLAllocation>)b];
  for (Replay* rp : ch->steps) { [ch->rs addAllocation:rp->arena]; [ch->rs addAllocation:rp->icb]; }
  [ch->rs commit]; [ch->rs requestResidency];
  if (g_queues.empty()) { fprintf(stderr, "rp_build_chain: MLX queue not seen yet\n"); return -4; }
  for (id q : g_queues) [(id<MTLCommandQueue>)q addResidencySet:ch->rs];
  g_chains.push_back(ch); g_all_rs.push_back(ch->rs);
  return (int)g_chains.size() - 1;
}

// ---- region API
// Fit the parametrised fields from M samples (recordings in `slots`, taken at consecutive integer parameters ps[k] = ps[0] + k). Call BEFORE rp_region_build (which consumes slot 0's recording).
// Returns the period L (> 0), or -1: the samples do not share one layout, or no period <= (M - 1) / 2 makes every field periodic-affine.
int rp_region_fit(int h, int M, const int* slots, const int* ps, int param_in, int maxL) {
  Region* R = g_regions[h]; if (M < 3) return -1;
  for (int k = 0; k < M; ++k) if (ps[k] != ps[0] + k) return -1;
  struct Lay { int cmd, idx; size_t size; };
  std::vector<std::vector<int64_t>> V(M); std::vector<Fld> locs; std::vector<HostBuf> hbs;
  std::vector<char> neverw;                                               // per host buffer: snapshot == end snapshot in every sample (a real constant, never written)
  if (getenv("RP_FITCMD")) { int fc = atoi(getenv("RP_FITCMD")); Rec& r = g_rec[slots[0]]; if (fc < (int)r.cmds.size()) for (auto& b : r.cmds[fc].binds) if (!b.bytes) { void* key = (__bridge void*)b.buf; id<MTLBuffer> mb = (id<MTLBuffer>)b.buf;
      fprintf(stderr, "  fitdbg sample0 cmd %d slot %d: len %lu born %d snap %d endsnap %d pinned %d\n", fc, b.idx, (unsigned long)[mb length], (int)r.born.count(key), (int)r.snap.count(key), (int)r.endsnap.count(key), (int)is_pinned(mb)); } }
  std::set<int> lbad;                                                    // host buffers whose size or presence differs between samples: scratch
  if (getenv("RP_FITDBG")) {                                                    // aliasing pattern of the bindings: born buffers by first-appearance ordinal, external ones by address
    auto sig = [&](Rec& r) { std::vector<std::vector<int64_t>> out; std::unordered_map<void*, int> ord; for (auto& c : r.cmds) { std::vector<int64_t> v; for (auto& b : c.binds) if (!b.bytes) { void* key = (__bridge void*)b.buf;
        if (r.born.count(key)) { auto it = ord.find(key); if (it == ord.end()) it = ord.emplace(key, (int)ord.size()).first; v.push_back(-1 - (int64_t)it->second); } else v.push_back((int64_t)(uintptr_t)key); v.push_back((int64_t)b.off); } out.push_back(v); } return out; };
    auto s0 = sig(g_rec[slots[0]]);
    for (int k = 1; k < M; ++k) { auto sk = sig(g_rec[slots[k]]); int shown = 0, diff = 0;
      for (size_t ci = 0; ci < sk.size(); ++ci) if (sk[ci] != s0[ci]) { diff++; if (shown < 4) { auto nm = g_psoname.find((__bridge void*)g_rec[slots[0]].cmds[ci].pso); fprintf(stderr, "  alias sample %d cmd %zu %s:", k, ci, nm == g_psoname.end() ? "?" : nm->second.substr(0, 40).c_str());
          for (size_t q = 0; q < sk[ci].size() && q < 14; ++q) fprintf(stderr, " %lld/%lld", (long long)s0[ci][q] % 100000, (long long)sk[ci][q] % 100000); fprintf(stderr, "\n"); shown++; } }
      { std::set<std::pair<void*, void*>> ext; for (size_t ci = 0; ci < sk.size(); ++ci) { Cmd& ca = g_rec[slots[0]].cmds[ci]; Cmd& cb = g_rec[slots[k]].cmds[ci]; for (size_t q = 0; q < ca.binds.size() && q < cb.binds.size(); ++q) if (!ca.binds[q].bytes && !cb.binds[q].bytes) {
          void* ka = (__bridge void*)ca.binds[q].buf; void* kb = (__bridge void*)cb.binds[q].buf; if (!g_rec[slots[0]].born.count(ka) && !g_rec[slots[k]].born.count(kb) && ka != kb) ext.insert({ka, kb}); } }
        fprintf(stderr, "  sample %d: %zu distinct external buffer swaps:", k, ext.size()); int n = 0; for (auto& e : ext) { if (n++ < 8) fprintf(stderr, " [len %lu -> %lu]", (unsigned long)[(id<MTLBuffer>)(__bridge id)e.first length], (unsigned long)[(id<MTLBuffer>)(__bridge id)e.second length]); } fprintf(stderr, "\n"); }
      fprintf(stderr, "  alias sample %d: %d of %zu commands bind a different pattern\n", k, diff, sk.size()); }
  }
  std::set<std::pair<int, int>> cand;                                    // (cmd, slot) of a born buffer never written by a kernel in AT LEAST one sample
  for (int k = 0; k < M; ++k) { Rec& r = g_rec[slots[k]]; std::unordered_set<void*> dn;
    for (size_t ci = 0; ci < r.cmds.size(); ++ci) for (auto& b : r.cmds[ci].binds) if (!b.bytes) { void* key = (__bridge void*)b.buf; if (!dn.insert(key).second) continue;
      auto sn = r.snap.find(key); if (sn != r.snap.end() && sn->second.size() <= (r.born.count(key) ? 4096u : 256u)) cand.insert({(int)ci, b.idx}); } }
  for (int k = 0; k < M; ++k) {
    Rec& r = g_rec[slots[k]]; std::vector<Fld> L0; std::vector<HostBuf> H0; std::vector<int64_t>& v = V[k]; std::unordered_set<void*> done;
    for (size_t ci = 0; ci < r.cmds.size(); ++ci) { Cmd& c = r.cmds[ci];
      int64_t gv[6] = {(int64_t)c.grid.width, (int64_t)c.grid.height, (int64_t)c.grid.depth, (int64_t)c.tg.width, (int64_t)c.tg.height, (int64_t)c.tg.depth};
      for (int a = 0; a < 6; ++a) { v.push_back(gv[a]); L0.push_back({a < 3 ? 0 : 1, (int)ci, a % 3, 0, -1, {}, 0}); }
      for (auto& b : c.binds) if (b.bytes) for (uint32_t w = 0; w + 4 <= b.len; w += 4) { int32_t x; memcpy(&x, &r.arena[b.arena + w], 4); v.push_back(x); L0.push_back({2, (int)ci, 0, (uint64_t)b.arena + w, -1, {}, 0}); }
      if (k == 0) { for (auto& b : c.binds) if (!b.bytes) { void* key = (__bridge void*)b.buf; if (!done.insert(key).second) continue;
          auto sn = r.snap.find(key); if (sn == r.snap.end() || !cand.count({(int)ci, b.idx})) continue;
          int hi = (int)H0.size(); H0.push_back({(int)ci, b.idx, sn->second.size(), nil, sn->second});
          for (size_t w = 0; w + 4 <= sn->second.size(); w += 4) { int32_t x; memcpy(&x, &sn->second[w], 4); v.push_back(x); L0.push_back({3, (int)ci, 0, (uint64_t)w, hi, {}, 0}); } } }
      else { for (size_t q = 0; q < hbs.size(); ++q) if (hbs[q].cmd == (int)ci) { id bb = nil; for (auto& b : c.binds) if (!b.bytes && b.idx == hbs[q].idx) bb = b.buf;
          void* key = (__bridge void*)bb; auto sn = bb ? r.snap.find(key) : r.snap.end();
          if (!bb || sn == r.snap.end() || sn->second.size() < hbs[q].size) { lbad.insert((int)q); if (getenv("RP_FITDBG2") && hbs[q].cmd == (getenv("RP_FITCMD") ? atoi(getenv("RP_FITCMD")) : 17)) fprintf(stderr, "  lbad: sample %d hb %zu cmd %d slot %d: bb %d in-snap %d size-now %zu size-sample0 %zu\n", k, q, hbs[q].cmd, hbs[q].idx, (int)(bb != nil), (int)(bb && sn != r.snap.end()), sn != r.snap.end() ? sn->second.size() : (size_t)0, hbs[q].size); for (size_t w = 0; w + 4 <= hbs[q].size; w += 4) { v.push_back(0); L0.push_back({3, (int)ci, 0, (uint64_t)w, (int)q, {}, 0}); } continue; }
          for (size_t w = 0; w + 4 <= hbs[q].size; w += 4) { int32_t x; memcpy(&x, &sn->second[w], 4); v.push_back(x); L0.push_back({3, (int)ci, 0, (uint64_t)w, (int)q, {}, 0}); } } }
    }
    if (k > 0) { Rec& r0 = g_rec[slots[0]]; if (r0.cmds.size() != r.cmds.size()) { fprintf(stderr, "rp_region_fit: sample %d has %zu commands vs %zu\n", k, r.cmds.size(), r0.cmds.size());
        size_t d = 0; while (d < std::min(r0.cmds.size(), r.cmds.size()) && r0.cmds[d].pso == r.cmds[d].pso) d++;
        auto nn = [&](Rec& q, size_t i) { if (i >= q.cmds.size()) return std::string("(end)"); auto it = g_psoname.find((__bridge void*)q.cmds[i].pso); return it == g_psoname.end() ? std::string("?") : it->second.substr(0, 50); };
        fprintf(stderr, "  first differing command %zu: sample0 %s | sample%d %s\n", d, nn(r0, d).c_str(), k, nn(r, d).c_str());
        for (size_t t = 1; t <= 3; ++t) fprintf(stderr, "    +%zu: sample0 %s | sample%d %s\n", t, nn(r0, d + t).c_str(), k, nn(r, d + t).c_str()); return -1; }
      for (size_t ci = 0; ci < r.cmds.size(); ++ci) if (r.cmds[ci].pso != r0.cmds[ci].pso || r.cmds[ci].threadgroups != r0.cmds[ci].threadgroups || r.cmds[ci].binds.size() != r0.cmds[ci].binds.size()) {
        auto n1 = g_psoname.find((__bridge void*)r.cmds[ci].pso), n0 = g_psoname.find((__bridge void*)r0.cmds[ci].pso);
        fprintf(stderr, "rp_region_fit: sample %d cmd %zu differs: %s vs %s (binds %zu vs %zu)\n", k, ci, n1 == g_psoname.end() ? "?" : n1->second.substr(0, 50).c_str(), n0 == g_psoname.end() ? "?" : n0->second.substr(0, 50).c_str(), r.cmds[ci].binds.size(), r0.cmds[ci].binds.size()); return -1; } }
    if (k == 0) { locs = L0; hbs = H0; neverw.assign(hbs.size(), 1); }
    else { if (L0.size() != locs.size()) { fprintf(stderr, "rp_region_fit: sample %d has another layout (%zu fields vs %zu)\n", k, L0.size(), locs.size()); return -1; }
    }
    for (size_t q = 0; q < hbs.size(); ++q) { Rec& rr = g_rec[slots[k]]; id bb = nil; for (auto& b : rr.cmds[hbs[q].cmd].binds) if (!b.bytes && b.idx == hbs[q].idx) bb = b.buf; if (!bb) { neverw[q] = 0; continue; }
      if (rec_writes(rr, (__bridge void*)bb)) neverw[q] = 0; }
  }
  if (getenv("RP_FITDBG")) {                                                   // born small buffers not in cand whose first-bind snapshot varies across samples
    std::map<std::pair<int,int>, std::vector<std::vector<uint8_t>>> seen; std::map<std::pair<int,int>, int> pr;
    for (int k = 0; k < M; ++k) { Rec& r = g_rec[slots[k]]; std::unordered_set<void*> dn;
      for (size_t ci = 0; ci < r.cmds.size(); ++ci) for (auto& b : r.cmds[ci].binds) if (!b.bytes) { void* key = (__bridge void*)b.buf; if (!r.born.count(key) || !dn.insert(key).second) continue;
        auto sn = r.snap.find(key); if (sn == r.snap.end() || cand.count({(int)ci, b.idx})) continue; seen[{(int)ci, b.idx}].push_back(sn->second); } }
    int shown = 0, tot = 0, varying = 0;
    for (auto& kv : seen) { tot++; if ((int)kv.second.size() != M) continue; bool var = false; for (int k = 1; k < M; ++k) if (kv.second[k] != kv.second[0]) var = true; if (!var) continue; varying++;
      if (shown < 10 && kv.second[0].size() <= 64) { auto nm = g_psoname.find((__bridge void*)g_rec[slots[0]].cmds[kv.first.first].pso); fprintf(stderr, "  scratch-like varying: cmd %d slot %d %s size %zu:", kv.first.first, kv.first.second, nm == g_psoname.end() ? "?" : nm->second.substr(0, 36).c_str(), kv.second[0].size());
        for (int k = 0; k < M && k < 6; ++k) { int32_t x; memcpy(&x, kv.second[k].data(), 4); fprintf(stderr, " %d", x); } fprintf(stderr, "\n"); shown++; } }
    fprintf(stderr, "  non-cand born buffers %d, varying across samples %d\n", tot, varying); }
  for (int L = 1; L <= std::min(maxL, (M - 1) / 2); ++L) {
    bool ok = true; size_t nf = locs.size(); std::vector<int64_t> slope(nf, 0); std::vector<char> var(nf, 0);
    std::set<int> badhb = lbad;
    for (size_t f = 0; f < nf && ok; ++f) {
      int64_t sl = V[L][f] - V[0][f]; bool fit = true;
      for (int k = 0; k + L < M; ++k) if (V[k + L][f] - V[k][f] != sl) { fit = false; break; }
      if (locs[f].kind == 3 && badhb.count(locs[f].hbi)) continue;
      if (!fit && locs[f].kind == 3) { badhb.insert(locs[f].hbi); continue; }
      if (!fit) { ok = false; break; }
      slope[f] = sl; if (ok) { bool c = sl == 0; for (int k = 1; k < L && c; ++k) if (V[k][f] != V[0][f]) c = false; var[f] = !c; }
    }
    if (!ok) { if (getenv("RP_FITDBG") && L == 2) { int shown = 0; for (size_t f = 0; f < nf && shown < 12; ++f) { int64_t sl = V[L][f] - V[0][f]; bool g = true; for (int k = 0; k + L < M; ++k) if (V[k + L][f] - V[k][f] != sl) g = false;
          if (!g) { auto nm = g_psoname.find((__bridge void*)g_rec[slots[0]].cmds[locs[f].cmd].pso); fprintf(stderr, "  nofit kind %d cmd %d %s off %llu hb %d:", locs[f].kind, locs[f].cmd, nm == g_psoname.end() ? "?" : nm->second.substr(0, 40).c_str(), (unsigned long long)locs[f].off, locs[f].hbi); for (int k = 0; k < M; ++k) fprintf(stderr, " %lld", (long long)V[k][f]); fprintf(stderr, "\n"); shown++; } } } continue; }
    bool strict_fail = false;
    static bool strict_off = getenv("RP_STRICT") && getenv("RP_STRICT")[0] == '0';   // default: any host buffer that does not follow the position table refuses the table (a silent drop = a stale constant)
    for (int q : badhb) if (!lbad.count(q) && !strict_off) { if (getenv("RP_FITDBG")) fprintf(stderr, "rp_region_fit: host constant (cmd %d slot %d size %zu) never written by a kernel but not periodic-affine in the position: refusing the table\n", hbs[q].cmd, hbs[q].idx, hbs[q].size);
      if (getenv("RP_FITDBG")) for (size_t f = 0; f < nf; ++f) if (locs[f].kind == 3 && locs[f].hbi == q) { bool g = true; int64_t sl = V[L][f] - V[0][f]; for (int k = 0; k + L < M; ++k) if (V[k + L][f] - V[k][f] != sl) g = false; if (!g) { fprintf(stderr, "    word %llu over the samples:", (unsigned long long)(locs[f].off / 4)); for (int k = 0; k < M; ++k) fprintf(stderr, " %lld", (long long)V[k][f]); fprintf(stderr, "\n"); } }
      strict_fail = true; break; }
    if (strict_fail) continue;
    R->flds.clear(); std::vector<int> remap(hbs.size(), -1); std::vector<HostBuf> keep;
    for (size_t q = 0; q < hbs.size(); ++q) if (!badhb.count((int)q)) { remap[q] = (int)keep.size(); keep.push_back(hbs[q]); }
    for (size_t f = 0; f < nf; ++f) if (var[f] && !(locs[f].kind == 3 && badhb.count(locs[f].hbi))) { Fld x = locs[f]; x.slope = slope[f]; if (x.kind == 3) x.hbi = remap[x.hbi]; for (int k = 0; k < L; ++k) x.T.push_back(V[k][f]); R->flds.push_back(x); }
    if (getenv("RP_FITDBG")) fprintf(stderr, "rp_region_fit: L %d, %zu host buffers kept, %zu dropped as scratch\n", L, keep.size(), badhb.size());
    if (getenv("RP_FITDBG2")) { for (size_t q = 0; q < hbs.size(); ++q) if (hbs[q].cmd == (getenv("RP_FITCMD") ? atoi(getenv("RP_FITCMD")) : 17)) fprintf(stderr, "  fit hb cmd %d slot %d size %zu: %s\n", hbs[q].cmd, hbs[q].idx, hbs[q].size, badhb.count((int)q) ? "DROPPED" : (lbad.count((int)q) ? "lbad" : "kept")); }
    R->fit_L = L; R->p0 = ps[0]; R->param_in = param_in; R->hbs = keep; return L;
  }
  return -1;
}
int rp_region_selfcheck(int h, int slot_k, int p) {   // apply the table for p to slot 0 and compare it with the recording of slot_k: number of differing fields (first ones printed)
  Region* R = g_regions[h]; RSlot* s = R->slots[0]; apply_params(R, s, h, p); Rec& r = g_rec[slot_k]; int bad = 0;
  for (size_t ci = 0; ci < r.cmds.size(); ++ci) { Cmd& c = r.cmds[ci]; Cmd& c0 = R->rec.cmds[ci];
    for (auto& b : c.binds) if (b.bytes) for (uint32_t w = 0; w + 4 <= b.len; w += 4) { int32_t a, e; memcpy(&e, &r.arena[b.arena + w], 4); memcpy(&a, (uint8_t*)[s->arena contents] + b.arena + w, 4);
        if (a != e) { if (bad++ < 10) fprintf(stderr, "  selfcheck arena cmd %zu off %u: table %d recorded %d\n", ci, b.arena + w, a, e); } }
    (void)c0; }
  for (size_t q = 0; q < R->hbs.size(); ++q) { HostBuf& hb = R->hbs[q]; Rec& rr = g_rec[slot_k]; auto& c = rr.cmds[hb.cmd]; id bb = nil; for (auto& b : c.binds) if (!b.bytes && b.idx == hb.idx) bb = b.buf;
    auto sn = bb ? rr.snap.find((__bridge void*)bb) : rr.snap.end(); if (sn == rr.snap.end() || sn->second.size() != hb.size) continue;
    for (size_t w = 0; w + 4 <= hb.size; w += 4) { int32_t a, e; memcpy(&e, &sn->second[w], 4); memcpy(&a, (uint8_t*)[(id<MTLBuffer>)s->hb[q] contents] + w, 4); if (a != e) { if (bad++ < 10) fprintf(stderr, "  selfcheck hostbuf %zu (cmd %d slot %d) word %zu: table %d recorded %d\n", q, hb.cmd, hb.idx, w / 4, a, e); } } }
  return bad;
}
long rp_region_peek(int h, int idx, void* out, long n) { Region* R = g_regions[h]; if (idx >= (int)R->priv_scratch.size()) return -1; id<MTLBuffer> b = (id<MTLBuffer>)R->priv_scratch[idx]; long m = std::min<long>(n, [b length]); memcpy(out, [b contents], m); return m; }
int rp_region_cmp(int h, int slot_e) {   // compare the end-of-call contents of the buffers the recording `slot_e` (eager run at the current state) bound with the region's private buffers, cmd by cmd
  Region* R = g_regions[h]; Rec& e = g_rec[slot_e];
  if (e.cmds.size() != R->rec.cmds.size()) { fprintf(stderr, "  cmp: command counts differ %zu vs %zu\n", e.cmds.size(), R->rec.cmds.size()); return -1; }
  for (size_t ci = 0; ci < e.cmds.size(); ++ci) if (e.cmds[ci].pso != R->rec.cmds[ci].pso || e.cmds[ci].binds.size() != R->rec.cmds[ci].binds.size()) { fprintf(stderr, "  cmp: cmd %zu differs in pipeline / binds\n", ci); return -1; }
  int bad = 0, shown = 0, checked = 0; std::unordered_set<void*> done;
  for (size_t ci = 0; ci < e.cmds.size(); ++ci) for (size_t q = 0; q < e.cmds[ci].binds.size(); ++q) { Bind& be = e.cmds[ci].binds[q]; Bind& br = R->rec.cmds[ci].binds[q]; if (be.bytes || br.bytes) continue;
    void* key = (__bridge void*)be.buf; if (!e.born.count(key) || !done.insert(key).second) continue;
    auto es = e.endsnap.find(key); if (es == e.endsnap.end()) continue;
    id<MTLBuffer> rb = (id<MTLBuffer>)br.buf; bool priv = false; for (id pb : R->priv) if (pb == br.buf) { priv = true; break; } if (!priv) continue;
    size_t n = std::min<size_t>(es->second.size(), [rb length]); checked++;
    if (memcmp(es->second.data(), [rb contents], n) != 0) { bad++; if (shown++ < 12) { auto nm = g_psoname.find((__bridge void*)e.cmds[ci].pso); size_t first = 0; while (first < n && es->second[first] == ((uint8_t*)[rb contents])[first]) first++;
        fprintf(stderr, "  cmp: first mismatch cmd %zu slot %d %s len %zu first diff byte %zu\n", ci, be.idx, nm == g_psoname.end() ? "?" : nm->second.substr(0, 46).c_str(), n, first);
        if (shown <= 4) { fprintf(stderr, "       eager f32:"); for (size_t w = 0; w + 4 <= std::min<size_t>(n, 32); w += 4) { float f; memcpy(&f, &es->second[w], 4); fprintf(stderr, " %g", f); } fprintf(stderr, " | replay f32:"); for (size_t w = 0; w + 4 <= std::min<size_t>(n, 32); w += 4) { float f; memcpy(&f, (uint8_t*)[rb contents] + w, 4); fprintf(stderr, " %g", f); } fprintf(stderr, "\n"); } } } }
  fprintf(stderr, "  cmp: %d buffers compared, %d differ\n", checked, bad); return bad;
}
void rp_set_swallow(int v) { g_swallow = v != 0; }
void rp_set_prefix(int n) { g_prefix_n = n; }
void rp_set_poison(int v) { g_poison_dyn = v; }
void rp_set_zero_range(int lo, int hi) { g_pz_lo = lo; g_pz_hi = hi; }
void rp_set_hb_zero(int lo, int hi) { g_hz_lo = lo; g_hz_hi = hi; }
int rp_region_nhb(int h) { return (int)g_regions[h]->hbs.size(); }
int rp_region_hb_info(int h, int q, char* out, int cap) { Region* R = g_regions[h]; if (q >= (int)R->hbs.size()) return -1; HostBuf& hb = R->hbs[q]; auto nm = g_psoname.find((__bridge void*)R->rec.cmds[hb.cmd].pso); int nf = 0; for (auto& f : R->flds) if (f.kind == 3 && f.hbi == q) nf++; snprintf(out, cap, "hb %d cmd %d slot %d %s size %zu fields %d first words:%s", q, hb.cmd, hb.idx, nm == g_psoname.end() ? "?" : nm->second.substr(0, 50).c_str(), hb.size, nf, ""); std::string w; for (size_t k = 0; k + 4 <= std::min<size_t>(hb.size, 24); k += 4) { int32_t v; memcpy(&v, &hb.init[k], 4); w += " " + std::to_string(v); } strncat(out, w.c_str(), cap - strlen(out) - 1); return 0; }
int rp_region_npriv(int h) { Region* R = g_regions[h]; int n = 0; for (id pb : R->priv) { bool hbx = false; for (auto& hb : R->hbs) if (hb.priv && hb.priv == pb) hbx = true; if (!hbx) n++; } return n; }
int rp_region_priv_info(int h, int k, char* out, int cap) { Region* R = g_regions[h]; int ix = 0; for (size_t q = 0; q < R->priv.size(); ++q) { id pb = R->priv[q]; bool hbx = false; for (auto& hb : R->hbs) if (hb.priv && hb.priv == pb) hbx = true; if (hbx) continue; if (ix++ == k) { std::string w; for (size_t ci = 0; ci < R->rec.cmds.size() && w.empty(); ++ci) for (auto& b : R->rec.cmds[ci].binds) if (!b.bytes && b.buf == pb) { auto nm = g_psoname.find((__bridge void*)R->rec.cmds[ci].pso); w = "first bound cmd " + std::to_string(ci) + " slot " + std::to_string(b.idx) + " " + (nm == g_psoname.end() ? "?" : nm->second.substr(0, 60)) + " len " + std::to_string([(id<MTLBuffer>)pb length]); break; } snprintf(out, cap, "%s", w.c_str()); return 0; } } return -1; }
int rp_region_cmp2(int h1, int h2, int quiet) {   // two regions with the same command structure, both just executed: first differing private buffers in command order
  Region* A = g_regions[h1]; Region* B = g_regions[h2];
  if (A->rec.cmds.size() != B->rec.cmds.size()) { fprintf(stderr, "  cmp2: command counts differ %zu vs %zu\n", A->rec.cmds.size(), B->rec.cmds.size()); return -1; }
  int bad = 0, shown = 0, checked = 0; std::unordered_set<void*> done;
  for (size_t ci = 0; ci < A->rec.cmds.size(); ++ci) {
    if (A->rec.cmds[ci].pso != B->rec.cmds[ci].pso || A->rec.cmds[ci].binds.size() != B->rec.cmds[ci].binds.size()) { fprintf(stderr, "  cmp2: cmd %zu structure differs\n", ci); return -2; }
    for (size_t q = 0; q < A->rec.cmds[ci].binds.size(); ++q) { Bind& ba = A->rec.cmds[ci].binds[q]; Bind& bb = B->rec.cmds[ci].binds[q]; if (ba.bytes || bb.bytes) continue;
      void* key = (__bridge void*)ba.buf; if (!done.insert(key).second) continue;
      { bool hbx = false; for (auto& hb : A->hbs) if (hb.priv && hb.priv == ba.buf) hbx = true; if (hbx) continue; }
      bool pa = false, pb = false; for (id x : A->priv) if (x == ba.buf) { pa = true; break; } for (id x : B->priv) if (x == bb.buf) { pb = true; break; } if (!pa || !pb) continue;
      id<MTLBuffer> ra = (id<MTLBuffer>)ba.buf; id<MTLBuffer> rb = (id<MTLBuffer>)bb.buf; size_t n = std::min<size_t>([ra length], [rb length]); checked++;
      if (memcmp([ra contents], [rb contents], n) != 0) { bad++; if (!quiet && shown++ < 10) { auto nm = g_psoname.find((__bridge void*)A->rec.cmds[ci].pso); size_t first = 0; while (first < n && ((uint8_t*)[ra contents])[first] == ((uint8_t*)[rb contents])[first]) first++;
          fprintf(stderr, "  cmp2: first mismatches: cmd %zu slot %d %s len %zu first diff byte %zu\n", ci, ba.idx, nm == g_psoname.end() ? "?" : nm->second.substr(0, 46).c_str(), n, first);
          { fprintf(stderr, "      MISMATCH BUFFER ints key:"); for (size_t w = 0; w + 4 <= std::min<size_t>(n, 48); w += 4) { int32_t v; memcpy(&v, (uint8_t*)[ra contents] + w, 4); fprintf(stderr, " %d", v); } fprintf(stderr, " | eagerR2:"); for (size_t w = 0; w + 4 <= std::min<size_t>(n, 48); w += 4) { int32_t v; memcpy(&v, (uint8_t*)[rb contents] + w, 4); fprintf(stderr, " %d", v); } fprintf(stderr, "\n"); }
          if (false) for (size_t q2 = 0; q2 < A->rec.cmds[ci].binds.size(); ++q2) { Bind& xa = A->rec.cmds[ci].binds[q2]; Bind& xb = B->rec.cmds[ci].binds[q2];
              if (xa.bytes) { fprintf(stderr, "      slot %d (bytes) key:", xa.idx); for (uint32_t w = 0; w + 4 <= std::min<uint32_t>(xa.len, 16); w += 4) { int32_t v; memcpy(&v, &A->rec.arena[xa.arena + w], 4); fprintf(stderr, " %d", v); } fprintf(stderr, " | eagerR2:"); for (uint32_t w = 0; w + 4 <= std::min<uint32_t>(xb.len, 16); w += 4) { int32_t v; memcpy(&v, &B->rec.arena[xb.arena + w], 4); fprintf(stderr, " %d", v); } fprintf(stderr, "\n"); continue; }
              id<MTLBuffer> ka = (id<MTLBuffer>)xa.buf; id<MTLBuffer> kb = (id<MTLBuffer>)xb.buf; fprintf(stderr, "      slot %d len %lu key f32:", xa.idx, (unsigned long)[ka length]); for (int w = 0; w < 4; ++w) { float f; memcpy(&f, (uint8_t*)[ka contents] + xa.off + 4 * w, 4); fprintf(stderr, " %g", f); } fprintf(stderr, " | eagerR2 f32:"); for (int w = 0; w < 4; ++w) { float f; memcpy(&f, (uint8_t*)[kb contents] + xb.off + 4 * w, 4); fprintf(stderr, " %g", f); } fprintf(stderr, "\n"); } } } }
  }
  if (!quiet) { for (size_t j = 0; j < std::min(A->outs.size(), B->outs.size()) && j < 1; ++j) { OutCopy& oa = A->outs[j]; OutCopy& ob = B->outs[j];
      fprintf(stderr, "  cmp2: output 0 source: key from_in %d off %llu nbytes %llu | R2 from_in %d off %llu nbytes %llu", (int)oa.from_in, (unsigned long long)oa.off, (unsigned long long)oa.nbytes, (int)ob.from_in, (unsigned long long)ob.off, (unsigned long long)ob.nbytes);
      if (!oa.from_in && !ob.from_in) { size_t nb = std::min<size_t>(oa.nbytes, ob.nbytes); bool eq = memcmp((uint8_t*)[(id<MTLBuffer>)oa.buf contents] + oa.off, (uint8_t*)[(id<MTLBuffer>)ob.buf contents] + ob.off, nb) == 0; fprintf(stderr, " | source contents equal: %d", (int)eq); } fprintf(stderr, "\n"); } }
  if (!quiet) fprintf(stderr, "  cmp2: %d private buffers compared, %d differ\n", checked, bad); return bad;
}
long rp_region_out_src(int h, int j, void* out, long n) { Region* R = g_regions[h]; if (j >= (int)R->outs.size()) return -1; OutCopy& oc = R->outs[j]; if (oc.from_in) return -2; long m = std::min<long>(n, oc.nbytes); memcpy(out, (uint8_t*)[(id<MTLBuffer>)oc.buf contents] + oc.off, m); return m; }
int rp_region_tablecmp(int h1, int h2, int p) {   // the key region's table applied for p against the exact recording of region h2 (same structure): arena words, grids, host buffer words
  Region* A = g_regions[h1]; Region* B = g_regions[h2]; RSlot* s = A->slots[0]; apply_params(A, s, h1, p); int bad = 0;
  if (A->rec.cmds.size() != B->rec.cmds.size()) return -1;
  for (size_t ci = 0; ci < A->rec.cmds.size(); ++ci) { Cmd& ca = A->rec.cmds[ci]; Cmd& cb = B->rec.cmds[ci];
    if (ca.binds.size() != cb.binds.size()) { fprintf(stderr, "  tablecmp: cmd %zu bind counts differ\n", ci); bad++; continue; }
    for (size_t q = 0; q < ca.binds.size(); ++q) { Bind& ba = ca.binds[q]; Bind& bb = cb.binds[q]; if (ba.bytes != bb.bytes) { fprintf(stderr, "  tablecmp: cmd %zu slot %d bytes-ness differs\n", ci, ba.idx); bad++; continue; }
      if (ba.bytes) { for (uint32_t w = 0; w + 4 <= std::min(ba.len, bb.len); w += 4) { int32_t x, y; memcpy(&x, (uint8_t*)[s->arena contents] + ba.arena + w, 4); memcpy(&y, &B->rec.arena[bb.arena + w], 4); if (x != y) { if (bad++ < 12) fprintf(stderr, "  tablecmp: arena cmd %zu slot %d word %u: table %d exact %d\n", ci, ba.idx, w / 4, x, y); } } }
      else { for (size_t hq = 0; hq < A->hbs.size(); ++hq) if (A->hbs[hq].priv && A->hbs[hq].priv == ba.buf && hq < s->hb.size() && s->hb[hq] && A->hbs[hq].size <= 64 && [&]{ for (auto& f : A->flds) if (f.kind == 3 && f.hbi == (int)hq) return true; return false; }()) { id<MTLBuffer> bbuf = (id<MTLBuffer>)bb.buf; size_t n = std::min<size_t>(A->hbs[hq].size, [bbuf length]);
          for (size_t w = 0; w + 4 <= n; w += 4) { int32_t x, y; memcpy(&x, (uint8_t*)[(id<MTLBuffer>)s->hb[hq] contents] + w, 4); memcpy(&y, (uint8_t*)[bbuf contents] + bb.off + w, 4); if (x != y) { if (bad++ < 12) fprintf(stderr, "  tablecmp: hostbuf cmd %zu slot %d word %zu: table %d exact %d\n", ci, ba.idx, w / 4, x, y); } } break; } } }
    // grids
    int64_t gk[6] = {(int64_t)ca.grid.width, (int64_t)ca.grid.height, (int64_t)ca.grid.depth, (int64_t)ca.tg.width, (int64_t)ca.tg.height, (int64_t)ca.tg.depth}; int64_t ge[6] = {(int64_t)cb.grid.width, (int64_t)cb.grid.height, (int64_t)cb.grid.depth, (int64_t)cb.tg.width, (int64_t)cb.tg.height, (int64_t)cb.tg.depth};
    for (const Fld& f : A->flds) if ((f.kind == 0 || f.kind == 1) && f.cmd == (int)ci) { int ix = (f.kind == 1 ? 3 : 0) + f.axis; int64_t v = f.T[(p - A->p0) % A->fit_L] + f.slope * ((p - A->p0) / A->fit_L); ge[ix] = ge[ix]; gk[ix] = v; }
    for (int ix = 0; ix < 6; ++ix) if (gk[ix] != ge[ix]) { if (bad++ < 12) fprintf(stderr, "  tablecmp: grid/tg cmd %zu axis %d: table %lld exact %lld\n", ci, ix, (long long)gk[ix], (long long)ge[ix]); }
  }
  fprintf(stderr, "  tablecmp: %d differing fields (table at p=%d vs exact recording)\n", bad, p); return bad;
}
int rp_region_hb_vs(int h1, int q, int h2, char* out, int cap) {   // 1 = host buffer q of region h1 (table init for this call's slot 0) equals the buffer of region h2 at the same (cmd, slot)
  Region* A = g_regions[h1]; Region* B = g_regions[h2]; HostBuf& hb = A->hbs[q]; RSlot* s = A->slots[0]; id bb = nil;
  for (auto& b : B->rec.cmds[hb.cmd].binds) if (!b.bytes && b.idx == hb.idx) bb = b.buf; if (!bb || q >= (int)s->hb.size() || !s->hb[q]) { snprintf(out, cap, "no counterpart"); return -1; }
  size_t n = std::min<size_t>(hb.size, [(id<MTLBuffer>)bb length]); bool eq = memcmp([(id<MTLBuffer>)s->hb[q] contents], [(id<MTLBuffer>)bb contents], n) == 0; std::string w = "key:"; for (size_t k = 0; k + 4 <= std::min<size_t>(n, 24); k += 4) { int32_t v; memcpy(&v, (uint8_t*)[(id<MTLBuffer>)s->hb[q] contents] + k, 4); w += " " + std::to_string(v); }
  w += " | exact:"; for (size_t k = 0; k + 4 <= std::min<size_t>(n, 24); k += 4) { int32_t v; memcpy(&v, (uint8_t*)[(id<MTLBuffer>)bb contents] + k, 4); w += " " + std::to_string(v); } snprintf(out, cap, "%s", w.c_str()); return eq ? 1 : 0; }
int rp_region_patchcmp(int h1, int h2) {   // input rebinding tables of two regions built from streams of the same structure: (cmd, slot) -> (input, delta)
  Region* A = g_regions[h1]; Region* B = g_regions[h2]; int bad = 0;
  auto key = [](Patch& p) { return ((uint64_t)p.cmd << 8) | (uint64_t)p.slot; };
  std::unordered_map<uint64_t, Patch*> ma, mb; for (auto& p : A->patches) ma[key(p)] = &p; for (auto& p : B->patches) mb[key(p)] = &p;
  for (auto& kv : mb) { auto it = ma.find(kv.first); auto nm = g_psoname.find((__bridge void*)B->rec.cmds[kv.second->cmd].pso);
    if (it == ma.end()) { if (bad++ < 12) fprintf(stderr, "  patchcmp: exact stream binds input %d (+%llu) at cmd %d slot %d (%s) but the key region binds a private/external buffer there\n", kv.second->in, (unsigned long long)kv.second->delta, kv.second->cmd, kv.second->slot, nm == g_psoname.end() ? "?" : nm->second.substr(0, 40).c_str()); }
    else if (it->second->in != kv.second->in || it->second->delta != kv.second->delta) { if (bad++ < 12) fprintf(stderr, "  patchcmp: cmd %d slot %d (%s): exact input %d +%llu, key input %d +%llu\n", kv.second->cmd, kv.second->slot, nm == g_psoname.end() ? "?" : nm->second.substr(0, 40).c_str(), kv.second->in, (unsigned long long)kv.second->delta, it->second->in, (unsigned long long)it->second->delta); } }
  for (auto& kv : ma) if (!mb.count(kv.first)) { if (bad++ < 12) fprintf(stderr, "  patchcmp: key rebinds input %d at cmd %d slot %d but the exact stream does not read an input there\n", kv.second->in, kv.second->cmd, kv.second->slot); }
  fprintf(stderr, "  patchcmp: %zu key patches, %zu exact patches, %d differences\n", A->patches.size(), B->patches.size(), bad); return bad;
}
int rp_region_scan(int h, int pval, int target) {   // apply the table for pval, then list every arena / host-buffer word equal to target (is it a patched field?)
  Region* R = g_regions[h]; RSlot* s = R->slots[0]; apply_params(R, s, h, pval); int n = 0;
  std::unordered_set<uint64_t> fa, fh; for (auto& f : R->flds) { if (f.kind == 2) fa.insert(f.off); if (f.kind == 3) fh.insert(((uint64_t)f.hbi << 32) | f.off); }
  for (size_t ci = 0; ci < R->rec.cmds.size(); ++ci) for (auto& b : R->rec.cmds[ci].binds) if (b.bytes) for (uint32_t w = 0; w + 4 <= b.len; w += 4) { int32_t v; memcpy(&v, (uint8_t*)[s->arena contents] + b.arena + w, 4); if (v == target) { auto nm = g_psoname.find((__bridge void*)R->rec.cmds[ci].pso); if (n++ < 20) fprintf(stderr, "  scan: arena cmd %zu slot %d word %u = %d (%s) patched-field=%d\n", ci, b.idx, w / 4, v, nm == g_psoname.end() ? "?" : nm->second.substr(0, 40).c_str(), (int)fa.count(b.arena + w)); } }
  for (size_t q = 0; q < R->hbs.size(); ++q) if (q < s->hb.size() && s->hb[q] && R->hbs[q].size <= 256) for (size_t w = 0; w + 4 <= R->hbs[q].size; w += 4) { int32_t v; memcpy(&v, (uint8_t*)[(id<MTLBuffer>)s->hb[q] contents] + w, 4); if (v == target) { auto nm = g_psoname.find((__bridge void*)R->rec.cmds[R->hbs[q].cmd].pso); if (n++ < 20) fprintf(stderr, "  scan: hostbuf %zu cmd %d slot %d word %zu = %d (%s) patched-field=%d\n", q, R->hbs[q].cmd, R->hbs[q].idx, w / 4, v, nm == g_psoname.end() ? "?" : nm->second.substr(0, 40).c_str(), (int)fh.count(((uint64_t)q << 32) | w)); } }
  fprintf(stderr, "  scan: %d words equal to %d at p=%d\n", n, target, pval); return n;
}
int rp_region_show_cmd(int h, const char* sub, int which) {   // after an apply_params for slot 0 (call rp_region_scan first): print the bindings of the `which`-th command whose kernel name contains `sub`
  Region* R = g_regions[h]; RSlot* s = R->slots[0]; int seen = 0;
  for (size_t ci = 0; ci < R->rec.cmds.size(); ++ci) { Cmd& c = R->rec.cmds[ci]; auto nm = g_psoname.find((__bridge void*)c.pso); if (nm == g_psoname.end() || nm->second.find(sub) == std::string::npos) continue; if (seen++ != which) continue;
    fprintf(stderr, "  cmd %zu %s grid %lu,%lu,%lu tg %lu,%lu,%lu\n", ci, nm->second.substr(0, 60).c_str(), (unsigned long)c.grid.width, (unsigned long)c.grid.height, (unsigned long)c.grid.depth, (unsigned long)c.tg.width, (unsigned long)c.tg.height, (unsigned long)c.tg.depth);
    for (auto& b : c.binds) { if (b.bytes) { fprintf(stderr, "    slot %d bytes:", b.idx); for (uint32_t w = 0; w + 4 <= std::min<uint32_t>(b.len, 32); w += 4) { int32_t v; memcpy(&v, (uint8_t*)[s->arena contents] + b.arena + w, 4); fprintf(stderr, " %d", v); } fprintf(stderr, "\n"); continue; }
      int hq = -1; for (size_t q = 0; q < R->hbs.size(); ++q) if (R->hbs[q].priv && R->hbs[q].priv == b.buf) hq = (int)q;
      fprintf(stderr, "    slot %d buffer len %lu off %llu %s", b.idx, (unsigned long)[(id<MTLBuffer>)b.buf length], (unsigned long long)b.off, hq >= 0 ? "HOSTBUF words:" : "(not a host buffer) bytes now:"); if (hq < 0) for (size_t w = 0; w + 4 <= std::min<size_t>([(id<MTLBuffer>)b.buf length] - b.off, 24); w += 4) { int32_t v; memcpy(&v, (uint8_t*)[(id<MTLBuffer>)b.buf contents] + b.off + w, 4); fprintf(stderr, " %d", v); } if (hq >= 0 && hq < (int)s->hb.size() && s->hb[hq]) for (size_t w = 0; w + 4 <= std::min<size_t>(R->hbs[hq].size, 32); w += 4) { int32_t v; memcpy(&v, (uint8_t*)[(id<MTLBuffer>)s->hb[hq] contents] + w, 4); fprintf(stderr, " %d", v); } fprintf(stderr, "\n"); }
    return 0; }
  return -1;
}
long rp_region_nfields(int h) { return (long)g_regions[h]->flds.size(); }
int rp_region_new(void) { g_regions.push_back(new Region()); return (int)g_regions.size() - 1; }
void rp_rec_snap_born(int slot, int on) { g_rec[slot].snap_born = on; }
// Build region h from recording `slot`. in_ptr / in_len: data address and bytes of the function's input arrays at record time; out_ptr / out_len: its results.
// Returns 0, or a negative code: -1 ICB allocation, -2 a result not found in the recording, -3 a pipeline without ICB support, -4 residency set, -5 no MLX queue seen.
static int rp_region_build_impl(int h, int slot, int n_in, const uintptr_t* in_ptr, const uintptr_t* in_len, int n_out, const uintptr_t* out_ptr, const uintptr_t* out_len, int nslots) {
  Region* R = g_regions[h]; NSError* err = nil;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  auto newbuf = [&](NSUInteger len) { return ((NewFn_len_opt)ORIG(g_dev, "newBufferWithLength:options:"))(dev, @selector(newBufferWithLength:options:), len, MTLResourceStorageModeShared); };
  if (g_rec[slot].others) { fprintf(stderr, "rp_region_build: the call used a blit encoder or a ranged setBuffers (%d), not replayable\n", g_rec[slot].others); return -6; }
  R->rec = std::move(g_rec[slot]); R->n_in = n_in; R->n_out = n_out; Rec& r = R->rec; R->ncmds = (int)r.cmds.size();
  auto in_of = [&](uintptr_t p, uint64_t* delta) { int best = -1; for (int i = 0; i < n_in; ++i) if (p >= in_ptr[i] && p < in_ptr[i] + std::max<uintptr_t>(in_len[i], 1) && (best < 0 || in_len[i] < in_len[best])) best = i; if (best >= 0) *delta = p - in_ptr[best]; return best; };
  if (getenv("RP_BINDDBG")) { fprintf(stderr, "rp_region_build: pool-sized bindings (262144 / 65536 B buffers); inputs:"); for (int i = 0; i < n_in; ++i) fprintf(stderr, " [%d]%lx+%lu", i, (unsigned long)in_ptr[i], (unsigned long)in_len[i]); fprintf(stderr, "\n");
    for (size_t ci = 0; ci < r.cmds.size(); ++ci) for (auto& b : r.cmds[ci].binds) if (!b.bytes) { id<MTLBuffer> mb = (id<MTLBuffer>)b.buf; size_t L = [mb length]; if (L < 60000 || L > 300000) continue; uint64_t dd = 0; int in2 = in_of(buf_ptr(b.buf, b.off), &dd);
      auto nm = g_psoname.find((__bridge void*)r.cmds[ci].pso); fprintf(stderr, "   cmd %zu slot %d %s len %zu off %llu ptr %lx class %s%d born %d\n", ci, b.idx, nm == g_psoname.end() ? "?" : nm->second.substr(0, 36).c_str(), L, (unsigned long long)b.off, (unsigned long)buf_ptr(b.buf, b.off), in2 >= 0 ? "INPUT" : "other", in2, (int)r.born.count((__bridge void*)b.buf)); } }
  std::unordered_set<void*> done;
  int bad = 0;
  for (size_t ci = 0; ci < r.cmds.size(); ++ci) for (auto& b : r.cmds[ci].binds) if (!b.bytes) {
    void* key = (__bridge void*)b.buf; id<MTLBuffer> ob = (id<MTLBuffer>)b.buf; uint64_t delta = 0;
    int in = in_of(buf_ptr(b.buf, b.off), &delta);
    if (in >= 0) { R->patch_of[((uint64_t)ci << 8) | (uint64_t)b.idx] = (int)R->patches.size(); R->patches.push_back({(int)ci, b.idx, in, delta}); continue; }
    if (!done.insert(key).second) continue;
    bool hostb = false; for (auto& hb : R->hbs) if (hb.cmd == (int)ci && hb.idx == b.idx) hostb = true;
    bool written = false;                                                                                      // recycled allocator buffer (not born) that a kernel of the call writes: scratch, must be private
    if (!r.born.count(key) && !hostb) { if (rec_writes(r, key)) written = true; }
    if (written) R->n_written++;
    if (!r.born.count(key) && !hostb && !written) { R->mp[key] = ob; R->own.push_back(ob); R->nbound++; if (getenv("RP_DEBUG")) R->ext_hash[key] = hash_buf([ob contents], std::min<size_t>([ob length], 1 << 22)); continue; }                  // external (weights, persistent constants): bound in place
    id<MTLBuffer> nb = newbuf([ob length] + (R->fit_L > 0 ? 65536 : 0));                                          // born in the recorded call: private scratch (a parametrised region grows with p: padded)
    auto sn = r.snap.find(key);
    if (sn != r.snap.end() && sn->second.size() == [ob length]) memcpy([nb contents], sn->second.data(), [ob length]); else memset([nb contents], 0, [ob length]);
    { auto es1 = r.endsnap.find(key); bool constlike = sn != r.snap.end() && es1 != r.endsnap.end() && sn->second == es1->second; if (!constlike) { R->priv_scratch.push_back(nb); auto nm = g_psoname.find((__bridge void*)r.cmds[ci].pso); R->priv_info.push_back("cmd " + std::to_string(ci) + " slot " + std::to_string(b.idx) + " " + (nm == g_psoname.end() ? "?" : nm->second.substr(0, 50)) + " len " + std::to_string([ob length]) + (sn != r.snap.end() ? " snap" : " nosnap")); } }
    if (getenv("RP_RESTORE")) { std::vector<uint8_t> im; auto es2 = r.endsnap.find(key); bool constlike2 = sn != r.snap.end() && es2 != r.endsnap.end() && sn->second == es2->second;
      bool zero_mode = getenv("RP_RESTORE")[0] == 'z'; if (sn != r.snap.end() && sn->second.size() == [ob length] && (!zero_mode || constlike2)) im = sn->second; R->priv_init.push_back({nb, im}); }
    R->mp[key] = nb; R->own.push_back(nb); R->priv.push_back(nb); R->clone_bytes += [ob length]; R->nclones++;
  }
  if (getenv("RP_BINDDBG")) { std::vector<int> cnt(n_in, 0); for (auto& pt : R->patches) cnt[pt.in]++; fprintf(stderr, "rp_region_build: patches per input:"); for (int i = 0; i < n_in; ++i) fprintf(stderr, " [%d len %lu]=%d", i, (unsigned long)in_len[i], cnt[i]); fprintf(stderr, "\n"); }
  // Barriers. MLX issues a barrier only for read-after-write on a buffer it tracks; the order of an eager step also hides write-after-read and write-after-write hazards (buffers
  // donated in place, one buffer written by two independent kernels): the command processor of an ICB issues commands faster and exposes them. A barrier is added before any
  // command that binds a private scratch buffer already bound by a command since the last barrier (inputs and weights are never written inside the region).
  R->bar.assign(r.cmds.size(), 0);
  { std::unordered_set<void*> wrote, read; long unk = 0;
    for (size_t ci = 0; ci < r.cmds.size(); ++ci) {
      Cmd& c = r.cmds[ci]; bool need = false; std::vector<std::pair<void*, bool>> mine;           // (private buffer, may be written)
      auto ac = g_access.find((__bridge void*)c.pso);
      for (auto& b : c.binds) if (!b.bytes) {
        void* key = (__bridge void*)b.buf; if (!(r.born.count(key) && R->mp.count(key))) continue;
        uint8_t a = ac == g_access.end() ? 2 : ac->second.a[b.idx]; if (ac == g_access.end()) { unk++; if (getenv("RP_DEBUG")) { auto nm = g_psoname.find((__bridge void*)c.pso); fprintf(stderr, "  no reflection: cmd %zu %s\n", ci, nm == g_psoname.end() ? "?" : nm->second.substr(0, 90).c_str()); } }
        if (a == 0) continue;                                                                      // stale binding of an earlier kernel, not an argument of this one
        bool w = a == 2; mine.push_back({key, w});
        if (wrote.count(key) || (w && read.count(key))) need = true;                               // RAW, WAW, WAR
      }
      if (c.barrier) { R->bar[ci] = 1; R->n_mlx_bar++; wrote.clear(); read.clear(); }
      else if (need && !getenv("RP_NOEXTRA")) { R->bar[ci] = 1; R->n_extra_bar++; wrote.clear(); read.clear(); }
      for (auto& m : mine) (m.second ? wrote : read).insert(m.first);
    }
    if (unk) fprintf(stderr, "rp_region_build: %ld bindings of pipelines without reflection (treated as may-write)\n", unk); }
  for (auto& c : r.cmds) if (!((id<MTLComputePipelineState>)c.pso).supportIndirectCommandBuffers) bad++;
  if (bad) { fprintf(stderr, "rp_region_build: %d commands use pipelines without supportIndirectCommandBuffers\n", bad); return -3; }
  for (int j = 0; j < n_out; ++j) {                                                                               // where each result lives after the ICB ran
    uint64_t d = 0; int in = in_of(out_ptr[j], &d);
    if (in >= 0 && in_len[in] >= out_len[j]) { R->outs.push_back({true, nil, d, in, out_len[j]}); continue; }
    bool found = false;
    for (auto& kv : R->mp) { id<MTLBuffer> ob = (__bridge id<MTLBuffer>)kv.first; uintptr_t c = (uintptr_t)[ob contents];
      if (out_ptr[j] >= c && out_ptr[j] + out_len[j] <= c + [ob length]) { R->outs.push_back({false, kv.second, out_ptr[j] - c, -1, out_len[j]}); found = true; break; } }
    if (!found) { fprintf(stderr, "rp_region_build: result %d not found in the recording\n", j); return -2; }
  }
  // Resolve every binding to its final buffer and let go of the recording's own references (the recorded scratch would otherwise stay alive next to its private copy).
  { id<MTLBuffer> dummy = newbuf(256);
    for (auto& c : r.cmds) for (auto& b : c.binds) if (!b.bytes) {
      if (R->patch_of.count(((uint64_t)(&c - &r.cmds[0]) << 8) | (uint64_t)b.idx)) b.buf = dummy; else b.buf = R->mp[(__bridge void*)b.buf];
    }
    if (!getenv("RP_DEBUG")) { r.keep.clear(); r.born.clear(); r.seen.clear(); r.snap.clear(); r.endsnap.clear(); g_hash0.clear(); } }
  for (auto& hb : R->hbs) { hb.priv = nil; for (auto& b : r.cmds[hb.cmd].binds) if (!b.bytes && b.idx == hb.idx) { hb.priv = b.buf; break; } if (R->patch_of.count(((uint64_t)hb.cmd << 8) | (uint64_t)hb.idx)) { hb.priv = nil; continue; } if (!hb.priv) { fprintf(stderr, "rp_region_build: host buffer of cmd %d slot %d lost\n", hb.cmd, hb.idx); return -7; } }
  if (!g_region_rs) {
    MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new]; rd.initialCapacity = 4096;
    g_region_rs = [dev newResidencySetWithDescriptor:rd error:&err];
    if (!g_region_rs) { fprintf(stderr, "rp_region_build: residency set failed %s\n", err.description.UTF8String); return -4; }
    if (g_queues.empty()) { fprintf(stderr, "rp_region_build: MLX queue not seen yet\n"); g_region_rs = nil; return -5; }
    for (id q : g_queues) [(id<MTLCommandQueue>)q addResidencySet:g_region_rs];
    g_all_rs.push_back(g_region_rs);
  }
  for (id b : R->own) [g_region_rs addAllocation:(id<MTLAllocation>)b];
  for (int i = 0; i < nslots; ++i) { RSlot* s = make_slot(R); if (!s) return -1; R->slots.push_back(s); for (id h_ : s->hb) if (h_) [g_region_rs addAllocation:(id<MTLAllocation>)h_]; [g_region_rs addAllocation:s->arena]; [g_region_rs addAllocation:s->icb]; }
  [g_region_rs commit]; [g_region_rs requestResidency];
  R->built = true;
  return 0;
}
long rp_region_stat(int h, int what) { Region* R = g_regions[h]; switch (what) { case 0: return R->replays; case 1: return R->rebinds; case 2: return R->extra_slots; case 20: return R->n_written; case 3: return R->clone_bytes; case 4: return R->nclones; case 5: return R->nbound; case 6: return (long)R->patches.size(); case 7: return R->ncmds; case 8: return (long)R->slots.size(); case 9: return R->n_mlx_bar; case 10: return R->n_extra_bar; } return -1; }

// release a region: its scratch, ICB slots and arenas go back to the system (the GPU must be idle: callers synchronise first)
static void rp_region_free_impl(int h) {
  Region* R = g_regions[h]; if (!R || !R->built) return; R->built = false; R->freed = true;
  if (g_region_rs) { for (id b : R->priv) [g_region_rs removeAllocation:(id<MTLAllocation>)b]; for (RSlot* s : R->slots) { [g_region_rs removeAllocation:s->arena]; [g_region_rs removeAllocation:s->icb]; } [g_region_rs commit]; }
  { std::lock_guard<std::mutex> l(g_slot_mu); for (RSlot* s : R->slots) g_cb_slots.erase(std::remove(g_cb_slots.begin(), g_cb_slots.end(), s), g_cb_slots.end()); }
  for (RSlot* s : R->slots) delete s;
  R->slots.clear(); R->own.clear(); R->priv.clear(); R->mp.clear(); R->patches.clear(); R->patch_of.clear(); R->outs.clear(); R->ext_hash.clear(); R->rec = Rec();
}
void rp_region_drop(int h) { Region* R = g_regions[h]; if (R) R->built = false; }       // stop intercepting (the trigger op then dispatches its stub); memory stays

// debug: after a replay has completed, compare every private scratch buffer with its contents at the end of the recorded call; print the first `maxn` that differ, in command order of their first use
int rp_region_diff(int h, int maxn) {
  Region* R = g_regions[h]; Rec& r = R->rec; std::unordered_set<void*> seen; int shown = 0, total = 0;
  for (size_t ci = 0; ci < r.cmds.size(); ++ci) for (auto& b : r.cmds[ci].binds) if (!b.bytes) {
    void* key = (__bridge void*)b.buf; if (!r.born.count(key) || !seen.insert(key).second) continue;
    auto es = r.endsnap.find(key); auto m = R->mp.find(key); if (es == r.endsnap.end() || m == R->mp.end()) continue;
    id<MTLBuffer> nb = (id<MTLBuffer>)m->second; size_t n = std::min<size_t>(es->second.size(), [nb length]);
    if (memcmp(es->second.data(), [nb contents], n) != 0) {
      total++;
      if (shown < maxn) { auto it = g_psoname.find((__bridge void*)r.cmds[ci].pso); size_t nd = 0; const uint8_t* a = es->second.data(); const uint8_t* c = (const uint8_t*)[nb contents]; for (size_t k = 0; k < n; ++k) nd += a[k] != c[k];
        fprintf(stderr, "  diff: cmd %zu %s buf %lu B, %zu bytes differ, bound as slot %d\n", ci, it == g_psoname.end() ? "?" : it->second.substr(0, 60).c_str(), (unsigned long)[nb length], nd, b.idx); shown++; }
    }
  }
  return total;
}

int rp_region_ext_changed(int h) {            // debug: bound-in-place buffers whose first 4 MB changed since the build (a region must not write them)
  Region* R = g_regions[h]; int n = 0;
  for (auto& kv : R->ext_hash) { id<MTLBuffer> ob = (__bridge id<MTLBuffer>)kv.first; if (hash_buf([ob contents], std::min<size_t>([ob length], 1 << 22)) != kv.second) { n++; std::string where; for (size_t ci = 0; ci < R->rec.cmds.size() && where.empty(); ++ci) for (auto& b : R->rec.cmds[ci].binds) if (!b.bytes && (__bridge void*)b.buf == kv.first) { auto nm = g_psoname.find((__bridge void*)R->rec.cmds[ci].pso); where = "cmd " + std::to_string(ci) + " slot " + std::to_string(b.idx) + " " + (nm == g_psoname.end() ? "?" : nm->second.substr(0, 44)); break; }
      fprintf(stderr, "  ext buffer %p (%lu B) changed: %s\n", kv.first, (unsigned long)[ob length], where.c_str()); } }
  return n;
}

// Python calls these from a thread with no autorelease pool: whatever the ObjC runtime autoreleases inside would never be released. One pool per call.
int rp_region_build(int h, int slot, int n_in, const uintptr_t* in_ptr, const uintptr_t* in_len, int n_out, const uintptr_t* out_ptr, const uintptr_t* out_len, int nslots) { @autoreleasepool { return rp_region_build_impl(h, slot, n_in, in_ptr, in_len, n_out, out_ptr, out_len, nslots); } }
void rp_region_free(int h) { @autoreleasepool { rp_region_free_impl(h); } }
void rp_install(int flag_icb) { @autoreleasepool { rp_install_impl(flag_icb); } }
int rp_build_chain(int n, const int* slots, const uintptr_t* out_ptrs, uintptr_t out_bytes, int out_slot) { @autoreleasepool { return rp_build_chain_impl(n, slots, out_ptrs, out_bytes, out_slot); } }
void rp_activate(int h) { g_active = h >= 0 ? g_chains[h] : nil; if (g_active) g_active->pos = 0; }
long rp_replays(void) { return g_replays; }
long rp_commits(void) { return g_commits; }
long rp_dbg(int i) { return g_dbg[i]; }
long rp_clone_bytes(int h) { return g_chains[h]->clone_bytes; }
long rp_nclones(int h) { return g_chains[h]->nclones; }
long rp_nrefresh(int h) { long k = 0; for (auto* r : g_chains[h]->steps) k += r->refresh.size(); return k; }
long rp_bound_bytes(int h) { return g_chains[h]->bound_bytes; }
long rp_nbound(int h) { return g_chains[h]->nbound; }
void rp_snap_reset(void) { g_hash0.clear(); }
// back to the state at the first recorded step: every private copy re-initialised, the chain restarts at step 0 (the GPU must be idle)
void rp_chain_rewind(int h) { Chain* ch = g_chains[h]; for (auto& c : ch->clones) { id<MTLBuffer> b = (id<MTLBuffer>)c.first; if (c.second) memcpy([b contents], c.second->data(), [b length]); else memset([b contents], 0, [b length]); } ch->pos = 0; }
long rp_chain_pos(int h) { return (long)g_chains[h]->pos; }
void rp_time(int on) { pthread_mutex_lock(&g_mu); g_time_on = on; g_gpu_us.clear(); g_gpu_wall_us.clear(); pthread_mutex_unlock(&g_mu); }
int rp_gpu_times(double* out, double* wall, int cap) { pthread_mutex_lock(&g_mu); int n = (int)std::min<size_t>(cap, g_gpu_us.size()); for (int i = 0; i < n; ++i) { out[i] = g_gpu_us[i]; wall[i] = g_gpu_wall_us[i]; } g_gpu_us.clear(); g_gpu_wall_us.clear(); pthread_mutex_unlock(&g_mu); return n; }

void rp_set_pins(const uintptr_t* ptr, const uintptr_t* len, int n) { g_pins.clear(); for (int i = 0; i < n; ++i) g_pins[ptr[i]] = ptr[i] + len[i]; }
void rp_rec_begin(int slot) { if ((size_t)slot >= g_rec.size()) g_rec.resize(slot + 1); g_rec[slot] = Rec(); g_rec[slot].snap_born = true; g_cur = slot; g_es = EncState(); }
int rp_rec_end(void) {
  Rec& r = R(); int n = (int)r.cmds.size(); g_cur = -1;
  for (id b : r.keep) { id<MTLBuffer> mb = (id<MTLBuffer>)b; size_t len = [mb length]; void* key = (__bridge void*)b;
    if (len <= kSmallMax && !r.endsnap.count(key) && !is_pinned(mb)) { std::vector<uint8_t>& v = r.endsnap[key]; v.resize(len); memcpy(v.data(), [mb contents], len); } }
  return n;
}
int rp_hostconst_dump(int slot, const char* path) {   // born small buffers never written by a kernel (contents at first bind == at the end): one line per first binding
  Rec& r = g_rec[slot]; FILE* f = fopen(path, "w"); if (!f) return -1; std::unordered_set<void*> done; int n = 0, ci = 0;
  for (auto& c : r.cmds) { for (auto& b : c.binds) if (!b.bytes) {
      void* key = (__bridge void*)b.buf; if (!r.born.count(key) || !done.insert(key).second) continue;
      auto sn = r.snap.find(key); auto es = r.endsnap.find(key); if (sn == r.snap.end() || es == r.endsnap.end() || sn->second != es->second) continue;
      fprintf(f, "%d %d %zu ", ci, b.idx, sn->second.size()); for (size_t k = 0; k < sn->second.size() && k < 8192; ++k) fprintf(f, "%02x", sn->second[k]); fprintf(f, "\n"); n++; }
    ci++; }
  fclose(f); return n;
}
int rp_rec_info(int slot, int what) {
  Rec& r = g_rec[slot];
  switch (what) { case 0: return (int)r.cmds.size(); case 1: return r.encoders; case 2: return r.fences; case 3: return r.others; case 4: return (int)r.arena.size(); case 5: return (int)r.born.size(); }
  return -1;
}
// text dump of one recording for diffing: one line per dispatch
int rp_dump(int slot, const char* path) {
  Rec& r = g_rec[slot]; FILE* f = fopen(path, "w"); if (!f) return -1;
  int ci = 0;
  for (auto& c : r.cmds) {
    auto it = g_psoname.find((__bridge void*)c.pso);
    fprintf(f, "%d %s %s grid=%lu,%lu,%lu tg=%lu,%lu,%lu bar=%d", ci++, it == g_psoname.end() ? "?" : it->second.c_str(), c.threadgroups ? "TG" : "TH", (unsigned long)c.grid.width, (unsigned long)c.grid.height, (unsigned long)c.grid.depth, (unsigned long)c.tg.width, (unsigned long)c.tg.height, (unsigned long)c.tg.depth, (int)c.barrier);
    for (auto& t : c.tgmem) fprintf(f, " tgm%u=%u", t.first, t.second);
    for (auto& b : c.binds) {
      if (b.bytes) { fprintf(f, " [%d:B%u:", b.idx, b.len); for (uint32_t k = 0; k < b.len && k < 64; ++k) fprintf(f, "%02x", r.arena[b.arena + k]); fprintf(f, "]"); }
      else { id<MTLBuffer> mb = b.buf; fprintf(f, " [%d:%s%lu+%llu@%06lx]", b.idx, r.born.count((__bridge void*)b.buf) ? "n" : "x", (unsigned long)mb.length, (unsigned long long)b.off, (unsigned long)((uintptr_t)(__bridge void*)b.buf & 0xffffff)); }
    }
    fprintf(f, "\n");
  }
  fclose(f); return 0;
}
}
