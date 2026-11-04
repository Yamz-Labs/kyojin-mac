// Probe: swizzle Metal encoder / command buffer / device methods and count what MLX calls during one decode step.
// build: clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation rp_probe.mm -o librp_probe.dylib
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <map>
#include <string>
#include <cstdio>
static std::map<std::string, long> g_cnt;
static bool g_rec = false;
static void bump(const char* s) { if (g_rec) g_cnt[s]++; }
#define HOOK_SETUP(cls, selname, rettype, ...)
static std::map<std::string, IMP> g_orig;
static void swz(Class c, SEL s, IMP imp) {
  Method m = class_getInstanceMethod(c, s);
  if (!m) { fprintf(stderr, "rp_probe: %s has no %s\n", class_getName(c), sel_getName(s)); return; }
  g_orig[std::string(class_getName(c)) + "/" + sel_getName(s)] = method_getImplementation(m);
  method_setImplementation(m, imp);
}
#define ORIG(c, sel) g_orig[std::string(class_getName(c)) + "/" + sel]
static Class g_enc, g_cb, g_dev, g_q;

static id h_pso_fn(id self, SEL _cmd, id fn, NSError** err) { bump("dev.newComputePipelineStateWithFunction"); g_cnt["PSO.fn"]++; return ((id(*)(id, SEL, id, NSError**))ORIG(g_dev, "newComputePipelineStateWithFunction:error:"))(self, _cmd, fn, err); }
static id h_pso_desc(id self, SEL _cmd, id d, NSUInteger opt, id* refl, NSError** err) { g_cnt["PSO.desc.opt"]++; return ((id(*)(id, SEL, id, NSUInteger, id*, NSError**))ORIG(g_dev, "newComputePipelineStateWithDescriptor:options:reflection:error:"))(self, _cmd, d, opt, refl, err); }
static id h_pso_desc2(id self, SEL _cmd, id d, NSError** err) { g_cnt["PSO.desc"]++; return ((id(*)(id, SEL, id, NSError**))ORIG(g_dev, "newComputePipelineStateWithDescriptor:error:"))(self, _cmd, d, err); }

#define E(sel) ORIG(g_enc, sel)
static void h_setPSO(id self, SEL c, id p) { bump("enc.setComputePipelineState"); ((void(*)(id, SEL, id))E("setComputePipelineState:"))(self, c, p); }
static void h_setBuf(id self, SEL c, id b, NSUInteger o, NSUInteger i) { bump("enc.setBuffer"); ((void(*)(id, SEL, id, NSUInteger, NSUInteger))E("setBuffer:offset:atIndex:"))(self, c, b, o, i); }
static void h_setBufs(id self, SEL c, const id* b, const NSUInteger* o, NSRange r) { bump("enc.setBuffers"); ((void(*)(id, SEL, const id*, const NSUInteger*, NSRange))E("setBuffers:offsets:withRange:"))(self, c, b, o, r); }
static void h_setBytes(id self, SEL c, const void* p, NSUInteger l, NSUInteger i) { bump("enc.setBytes"); ((void(*)(id, SEL, const void*, NSUInteger, NSUInteger))E("setBytes:length:atIndex:"))(self, c, p, l, i); }
static void h_setTG(id self, SEL c, NSUInteger l, NSUInteger i) { bump("enc.setThreadgroupMemoryLength"); ((void(*)(id, SEL, NSUInteger, NSUInteger))E("setThreadgroupMemoryLength:atIndex:"))(self, c, l, i); }
static void h_dT(id self, SEL c, MTLSize g, MTLSize t) { bump("enc.dispatchThreads"); ((void(*)(id, SEL, MTLSize, MTLSize))E("dispatchThreads:threadsPerThreadgroup:"))(self, c, g, t); }
static void h_dTG(id self, SEL c, MTLSize g, MTLSize t) { bump("enc.dispatchThreadgroups"); ((void(*)(id, SEL, MTLSize, MTLSize))E("dispatchThreadgroups:threadsPerThreadgroup:"))(self, c, g, t); }
static void h_barS(id self, SEL c, NSUInteger s) { bump("enc.memoryBarrierWithScope"); ((void(*)(id, SEL, NSUInteger))E("memoryBarrierWithScope:"))(self, c, s); }
static void h_barR(id self, SEL c, const id* r, NSUInteger n) { bump("enc.memoryBarrierWithResources"); ((void(*)(id, SEL, const id*, NSUInteger))E("memoryBarrierWithResources:count:"))(self, c, r, n); }
static void h_upF(id self, SEL c, id f) { bump("enc.updateFence"); ((void(*)(id, SEL, id))E("updateFence:"))(self, c, f); }
static void h_waF(id self, SEL c, id f) { bump("enc.waitForFence"); ((void(*)(id, SEL, id))E("waitForFence:"))(self, c, f); }
static void h_end(id self, SEL c) { bump("enc.endEncoding"); ((void(*)(id, SEL))E("endEncoding"))(self, c); }
static void h_useR(id self, SEL c, id r, NSUInteger u) { bump("enc.useResource"); ((void(*)(id, SEL, id, NSUInteger))E("useResource:usage:"))(self, c, r, u); }
static void h_useRs(id self, SEL c, const id* r, NSUInteger n, NSUInteger u) { bump("enc.useResources"); ((void(*)(id, SEL, const id*, NSUInteger, NSUInteger))E("useResources:count:usage:"))(self, c, r, n, u); }
#define CB(sel) ORIG(g_cb, sel)
static id h_cenc(id self, SEL c, NSUInteger t) { bump("cb.computeCommandEncoderWithDispatchType"); if (g_rec) g_cnt[std::string("enc.class=") + class_getName(object_getClass(self))]++; return ((id(*)(id, SEL, NSUInteger))CB("computeCommandEncoderWithDispatchType:"))(self, c, t); }
static id h_cenc0(id self, SEL c) { bump("cb.computeCommandEncoder"); return ((id(*)(id, SEL))CB("computeCommandEncoder"))(self, c); }
static id h_benc(id self, SEL c) { bump("cb.blitCommandEncoder"); return ((id(*)(id, SEL))CB("blitCommandEncoder"))(self, c); }
static void h_commit(id self, SEL c) { bump("cb.commit"); ((void(*)(id, SEL))CB("commit"))(self, c); }
static void h_sig(id self, SEL c, id e, uint64_t v) { bump("cb.encodeSignalEvent"); ((void(*)(id, SEL, id, uint64_t))CB("encodeSignalEvent:value:"))(self, c, e, v); }
static void h_wai(id self, SEL c, id e, uint64_t v) { bump("cb.encodeWaitForEvent"); ((void(*)(id, SEL, id, uint64_t))CB("encodeWaitForEvent:value:"))(self, c, e, v); }
#define Q(sel) ORIG(g_q, sel)
static id h_qcb(id self, SEL c) { bump("q.commandBuffer"); return ((id(*)(id, SEL))Q("commandBuffer"))(self, c); }
static id h_qcbu(id self, SEL c) { bump("q.commandBufferWithUnretainedReferences"); return ((id(*)(id, SEL))Q("commandBufferWithUnretainedReferences"))(self, c); }

extern "C" {
void rp_probe_install(void) {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  g_dev = object_getClass(dev); g_q = object_getClass(q); g_cb = object_getClass(cb); g_enc = object_getClass(e);
  fprintf(stderr, "rp_probe: device %s queue %s cb %s enc %s\n", class_getName(g_dev), class_getName(g_q), class_getName(g_cb), class_getName(g_enc));
  [e endEncoding]; [cb commit];
  swz(g_dev, @selector(newComputePipelineStateWithFunction:error:), (IMP)h_pso_fn);
  swz(g_dev, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)h_pso_desc);
  swz(g_dev, @selector(newComputePipelineStateWithDescriptor:error:), (IMP)h_pso_desc2);
  swz(g_enc, @selector(setComputePipelineState:), (IMP)h_setPSO);
  swz(g_enc, @selector(setBuffer:offset:atIndex:), (IMP)h_setBuf);
  swz(g_enc, @selector(setBuffers:offsets:withRange:), (IMP)h_setBufs);
  swz(g_enc, @selector(setBytes:length:atIndex:), (IMP)h_setBytes);
  swz(g_enc, @selector(setThreadgroupMemoryLength:atIndex:), (IMP)h_setTG);
  swz(g_enc, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)h_dT);
  swz(g_enc, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)h_dTG);
  swz(g_enc, @selector(memoryBarrierWithScope:), (IMP)h_barS);
  swz(g_enc, @selector(memoryBarrierWithResources:count:), (IMP)h_barR);
  swz(g_enc, @selector(updateFence:), (IMP)h_upF);
  swz(g_enc, @selector(waitForFence:), (IMP)h_waF);
  swz(g_enc, @selector(endEncoding), (IMP)h_end);
  swz(g_enc, @selector(useResource:usage:), (IMP)h_useR);
  swz(g_enc, @selector(useResources:count:usage:), (IMP)h_useRs);
  swz(g_cb, @selector(computeCommandEncoderWithDispatchType:), (IMP)h_cenc);
  swz(g_cb, @selector(computeCommandEncoder), (IMP)h_cenc0);
  swz(g_cb, @selector(blitCommandEncoder), (IMP)h_benc);
  swz(g_cb, @selector(commit), (IMP)h_commit);
  swz(g_cb, @selector(encodeSignalEvent:value:), (IMP)h_sig);
  swz(g_cb, @selector(encodeWaitForEvent:value:), (IMP)h_wai);
  swz(g_q, @selector(commandBuffer), (IMP)h_qcb);
  swz(g_q, @selector(commandBufferWithUnretainedReferences), (IMP)h_qcbu);
}
void rp_probe_begin(void) { for (auto& kv : g_cnt) fprintf(stderr, "  [startup] %-44s %ld\n", kv.first.c_str(), kv.second); g_cnt.clear(); g_rec = true; }
void rp_probe_end(void) { g_rec = false; for (auto& kv : g_cnt) fprintf(stderr, "  %-52s %ld\n", kv.first.c_str(), kv.second); fflush(stderr); }
}
