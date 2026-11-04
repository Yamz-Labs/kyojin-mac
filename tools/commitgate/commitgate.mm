// Commit gate for the Engram disk path: holds the command buffers MLX commits for one decode step until the host has written the staged rows, so the step
// can be encoded (and sit committed-but-held) while the previous step still runs. Metal makes CPU writes made before -[MTLCommandBuffer commit] visible to
// the GPU; a GPU kernel that polls a CPU-written flag does not get that (see ULTRA.md s.14), which is why the wait is on the commit and not in a kernel.
// Swizzles -[MTLCommandBuffer commit] like tools/cbcount. API (ctypes):
//   cg_install()      once
//   cg_hold(n)        the next n commits are held (n = command buffers per step)
//   cg_release()      commit the held buffers in order; commits of the step still to arrive (n - held) pass through untouched
//   cg_seen()         commits seen so far (held or passed)
// Ordering: commits are issued in arrival order; a commit arriving while cg_release flushes waits for it, so the queue order never changes.
// build: clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation tools/commitgate/commitgate.mm -o tools/commitgate/libcommitgate.dylib
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <pthread.h>

static IMP g_orig = nullptr;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static long g_seen = 0, g_tokens = 0, g_credit = 0;
static NSMutableArray* g_held = nil;

static void h_commit(id self, SEL c) {
  pthread_mutex_lock(&g_mu);
  g_seen++;
  if (g_credit > 0) g_credit--;
  else if (g_tokens > 0) { g_tokens--; [g_held addObject:self]; pthread_mutex_unlock(&g_mu); return; }
  ((void (*)(id, SEL))g_orig)(self, c);                                  // under the lock: keeps the queue order against a concurrent cg_release
  pthread_mutex_unlock(&g_mu);
}
extern "C" {
void cg_install(void) {
  if (g_orig) return;
  g_held = [NSMutableArray new];
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  Method m = class_getInstanceMethod(object_getClass(cb), @selector(commit));
  g_orig = method_getImplementation(m);
  method_setImplementation(m, (IMP)h_commit);
  [cb commit];
}
void cg_hold(long n) { pthread_mutex_lock(&g_mu); g_tokens = n; g_credit = 0; pthread_mutex_unlock(&g_mu); }
long cg_release(void) {
  pthread_mutex_lock(&g_mu);
  long n = (long)[g_held count];
  g_credit = g_tokens; g_tokens = 0;
  for (id cb in g_held) ((void (*)(id, SEL))g_orig)(cb, @selector(commit));
  [g_held removeAllObjects];
  pthread_mutex_unlock(&g_mu);
  return n;
}
long cg_seen(void) { pthread_mutex_lock(&g_mu); long v = g_seen; pthread_mutex_unlock(&g_mu); return v; }
}
