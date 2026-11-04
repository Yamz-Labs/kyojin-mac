// Counts the MTLResidencySet traffic MLX causes (swizzles addAllocation:, addAllocations:count:, removeAllocation:, removeAllAllocations, commit,
// requestResidency on the class of a real residency set). Companion of cbcount.mm (commits); both can be installed in one process.
// build: clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation tools/cbcount/rscount.mm -o tools/cbcount/librscount.dylib
// Needs macOS 15 (MTLResidencySet). rs_install() returns 0 on success, -1 when the OS has no residency sets (all counters stay 0).
// Not built or run by the author (no Mac in the authoring sandbox): tools/cbstat.py reports "unavailable" if the load or the install fails.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>

static long g_add = 0, g_add_n = 0, g_rm = 0, g_rm_all = 0, g_commit = 0, g_request = 0;
static IMP o_add, o_addn, o_rm, o_rmall, o_commit, o_req;

static void h_add(id self, SEL c, id a) { __atomic_fetch_add(&g_add, 1, __ATOMIC_RELAXED); ((void (*)(id, SEL, id))o_add)(self, c, a); }
static void h_addn(id self, SEL c, const id* a, NSUInteger n) {
  __atomic_fetch_add(&g_add, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&g_add_n, (long)n, __ATOMIC_RELAXED);
  ((void (*)(id, SEL, const id*, NSUInteger))o_addn)(self, c, a, n);
}
static void h_rm(id self, SEL c, id a) { __atomic_fetch_add(&g_rm, 1, __ATOMIC_RELAXED); ((void (*)(id, SEL, id))o_rm)(self, c, a); }
static void h_rmall(id self, SEL c) { __atomic_fetch_add(&g_rm_all, 1, __ATOMIC_RELAXED); ((void (*)(id, SEL))o_rmall)(self, c); }
static void h_commit(id self, SEL c) { __atomic_fetch_add(&g_commit, 1, __ATOMIC_RELAXED); ((void (*)(id, SEL))o_commit)(self, c); }
static void h_req(id self, SEL c) { __atomic_fetch_add(&g_request, 1, __ATOMIC_RELAXED); ((void (*)(id, SEL))o_req)(self, c); }

static IMP swiz(Class k, SEL s, IMP h) {
  Method m = class_getInstanceMethod(k, s);
  if (!m) return nullptr;
  IMP o = method_getImplementation(m);
  method_setImplementation(m, h);
  return o;
}

extern "C" {
int rs_install(void) {
  if (o_commit) return 0;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (![dev respondsToSelector:@selector(newResidencySetWithDescriptor:error:)]) return -1;
  MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
  NSError* err = nil;
  id rs = [dev newResidencySetWithDescriptor:d error:&err];
  if (!rs) return -1;
  Class k = object_getClass(rs);
  o_add = swiz(k, @selector(addAllocation:), (IMP)h_add);
  o_addn = swiz(k, @selector(addAllocations:count:), (IMP)h_addn);
  o_rm = swiz(k, @selector(removeAllocation:), (IMP)h_rm);
  o_rmall = swiz(k, @selector(removeAllAllocations), (IMP)h_rmall);
  o_commit = swiz(k, @selector(commit), (IMP)h_commit);
  o_req = swiz(k, @selector(requestResidency), (IMP)h_req);
  return o_commit ? 0 : -1;
}
// out[0..5] = addAllocation(s) calls, allocations added by the array form, removeAllocation calls, removeAll calls, commits, requestResidency calls
void rs_counts(long* out) { out[0] = g_add; out[1] = g_add_n; out[2] = g_rm; out[3] = g_rm_all; out[4] = g_commit; out[5] = g_request; }
}
