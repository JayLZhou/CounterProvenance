/* Time a Treelite (tl2cgen) model's own predict() over a file of query rows, one call per row.
   usage: tl_loop model.so n_feature n_query entry_bytes rows.bin out.bin
   rows.bin: n_query*n_feature doubles; out.bin receives n_query doubles.  Prints ns per query. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*predict_fn)(void *, int, double *);

int main(int argc, char **argv) {
    if (argc != 7) { fprintf(stderr, "usage\n"); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    predict_fn predict = (predict_fn)dlsym(lib, "predict");
    long nf = atol(argv[2]), nq = atol(argv[3]), eb = atol(argv[4]);
    double *rows = malloc(sizeof(double) * nf * nq), *out = malloc(sizeof(double) * nq);
    FILE *f = fopen(argv[5], "rb");
    if (fread(rows, sizeof(double), nf * nq, f) != (size_t)(nf * nq)) return 1;
    fclose(f);
    char *entries = calloc(nf * nq, eb);
    for (long i = 0; i < nf * nq; i++) {
        if (eb == 8) { double v = rows[i]; memcpy(entries + 8 * i, &v, 8); }
        else { float v = (float)rows[i]; memcpy(entries + 4 * i, &v, 4); }
    }
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (long q = 0; q < nq; q++) { out[q] = 0.0; predict(entries + eb * nf * q, 0, &out[q]); }
        clock_gettime(CLOCK_MONOTONIC, &b);
        double ns = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / nq;
        if (ns < best) best = ns;
    }
    f = fopen(argv[6], "wb");
    fwrite(out, sizeof(double), nq, f);
    fclose(f);
    printf("%.1f\n", best);
    return 0;
}
