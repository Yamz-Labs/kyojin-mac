// SPDX-License-Identifier: Apache-2.0
// Scattered fixed-width row reads for kyojin_mlx/engram_rows.py: one call reads rows[i] (width bytes at off + rows[i] * width) into
// dst[i * width] with `threads` POSIX threads doing plain preads. Called through ctypes (the GIL is released for the whole call).
// A prefill-sized DS4.1 Engram read (~196k value rows of one layer, cold) is NVMe-IOPS bound: 0.40-0.68 s here against 0.60-1.22 s through page faults on a
// memory map (np.take), whose fault path does not scale past ~16 threads.
#include <pthread.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct { int fd; int64_t off, width; const int64_t *rows; int64_t n; uint8_t *dst; int err; } rr_job;

static void *rr_run(void *arg) {
    rr_job *j = (rr_job *)arg;
    for (int64_t i = 0; i < j->n; i++) {
        uint8_t *d = j->dst + i * j->width;
        int64_t pos = j->off + j->rows[i] * j->width, got = 0;
        while (got < j->width) {
            ssize_t r = pread(j->fd, d + got, (size_t)(j->width - got), pos + got);
            if (r <= 0) { j->err = 1; return 0; }
            got += r;
        }
    }
    return 0;
}

// 0 = all rows read, -1 = a short read (dst is then incomplete). A thread that cannot start runs its share in the caller.
int rr_read(int fd, int64_t off, const int64_t *rows, int64_t n, int64_t width, uint8_t *dst, int threads) {
    enum { MAXT = 256 };
    if (threads > MAXT) threads = MAXT;
    if (threads > n) threads = (int)n;
    if (threads < 1) threads = 1;
    pthread_t th[MAXT];
    int live[MAXT];
    rr_job jobs[MAXT];
    for (int t = 0; t < threads; t++) {
        int64_t a = n * t / threads, b = n * (t + 1) / threads;
        jobs[t] = (rr_job){fd, off, width, rows + a, b - a, dst + a * width, 0};
        live[t] = t < threads - 1 && pthread_create(&th[t], 0, rr_run, &jobs[t]) == 0;
        if (!live[t]) rr_run(&jobs[t]);
    }
    int err = 0;
    for (int t = 0; t < threads; t++) {
        if (live[t]) pthread_join(th[t], 0);
        err |= jobs[t].err;
    }
    return err ? -1 : 0;
}

// Rows (of `sample` evenly spaced ones) whose pages are all in the page cache; map = the table's memory map (row 0 at map).
int64_t rr_hot(const uint8_t *map, const int64_t *rows, int64_t n, int64_t width, int64_t sample) {
    int64_t page = (int64_t)getpagesize(), hot = 0;
    if (sample > n) sample = n;
    for (int64_t k = 0; k < sample; k++) {
        const uint8_t *p = map + rows[n * k / sample] * width;
        uintptr_t a = (uintptr_t)p & ~(uintptr_t)(page - 1), b = (uintptr_t)p + (uintptr_t)width;
        char vec[4];
        int64_t np = (int64_t)((b - a + page - 1) / page), ok = np <= 4 && mincore((void *)a, b - a, vec) == 0;
        for (int64_t q = 0; ok && q < np; q++) ok = vec[q] & MINCORE_INCORE;
        hot += ok;
    }
    return hot;
}
