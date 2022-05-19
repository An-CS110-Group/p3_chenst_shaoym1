#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <xmmintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "thpool.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159

#define MIN(x, y) (((x) < (y)) ? (x) : (y))


typedef struct FVec {
    int length;
    int min_length;
    int min_deta;
    float *data;
    float *sum;
} FVec;

typedef struct Image {
    int dimX, dimY, numChannels;
    float *data;
} Image;

threadpool thPool;

void normalize_FVec(FVec v) {
    unsigned int i, j;
    int ext = v.length / 2;
    v.sum[0] = v.data[ext];
    for (i = ext + 1, j = 1; i < v.length; i++, j++) { v.sum[j] = v.sum[j - 1] + v.data[i] * 2; }
}

float *get_pixel(Image *img, int x, int y) {
    if (x < 0) { x = 0; }
    if (x >= img->dimX) { x = img->dimX - 1; }
    if (y < 0) { y = 0; }
    if (y >= img->dimY) { y = img->dimY - 1; }
    return img->data + img->numChannels * (y * img->dimX + x);
}

float gd(float a, float b, float x) {
    float c = (x - b) / a;
    return (float) (expf((-.5f) * c * c) / (a * sqrt(2 * PI)));
}

FVec make_gv(float a, float x0, float x1, int length, int min_length) {
    FVec v;
    v.length = length;
    v.min_length = min_length;
    if (v.min_length > v.length) {
        v.min_deta = 0;
    } else {
        v.min_deta = ((v.length - v.min_length) / 2);
    }
    v.data = malloc(length * sizeof(float) + sizeof(float));
    v.sum = malloc((length / 2 + 1) * sizeof(float) + sizeof(float));
    float step = (x1 - x0) / ((float) length);
    int offset = length / 2;

    for (int i = 0; i < length; i++) { v.data[i] = gd(a, 0.0f, (float) (i - offset) * step); }
    normalize_FVec(v);
    return v;
}

typedef struct transExeUnit {
    Image *src;
    Image *dst;
    int i;
} transExeUnit;

void processTrans(transExeUnit *var) {
    for (int j = 0; j < var->src->dimY; ++j) {
        var->dst->data[(var->dst->dimX * var->i + j) * var->dst->numChannels + 0] = var->src->data[(var->src->dimX * j + var->i) * var->src->numChannels + 0];
        var->dst->data[(var->dst->dimX * var->i + j) * var->dst->numChannels + 1] = var->src->data[(var->src->dimX * j + var->i) * var->src->numChannels + 1];
        var->dst->data[(var->dst->dimX * var->i + j) * var->dst->numChannels + 2] = var->src->data[(var->src->dimX * j + var->i) * var->src->numChannels + 2];
    }
}

void transpose_block(Image *src, Image *dst) {
    dst->dimX = src->dimY;
    dst->dimY = src->dimX;
    dst->numChannels = src->numChannels;
    //#pragma omp parallel for schedule(dynamic) default(none) shared(src, dst)
    for (int i = 0; i < src->dimX; ++i) {
        processTrans(&(transExeUnit){.src = src, .dst = dst, .i = i});
        //        There could be data race if use thpool
        //        thpool_add_work(thPool, (void (*)(void *)) processTrans, &(transExeUnit){.src=src, .dst=dst, .i=i});
    }
    thpool_wait(thPool);
}

Image img_sc(Image a) {
    Image b = a;
    b.data = malloc(b.dimX * b.dimY * b.numChannels * sizeof(float) + sizeof(float));
    return b;
}

typedef struct exeUnit {
    int y;
    Image *a;
    Image *b;
    FVec *gv;
    int ext;
    float *gvData;
    float *pixels;
} exeUnit;

void thread(exeUnit *var) {
    for (int x = 0; x < var->a->dimX; x++) {
        int deta = MIN(MIN(MIN(var->a->dimY - var->y - 1, var->y), MIN(var->a->dimX - x - 1, x)), var->gv->min_deta);
        __m256 sum[3] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        int i;
        for (i = deta; i < var->gv->length - deta - 8; i += 8) {
            sum[0] = _mm256_fmadd_ps(_mm256_loadu_ps(&var->pixels[3 * (x + i + 0) + 0 + 3 * var->y * (var->a->dimX + 2 * var->ext + 1)]), _mm256_loadu_ps(&var->gvData[3 * i + 0]),
                                     sum[0]);
            sum[1] = _mm256_fmadd_ps(_mm256_loadu_ps(&var->pixels[3 * (x + i + 2) + 2 + 3 * var->y * (var->a->dimX + 2 * var->ext + 1)]), _mm256_loadu_ps(&var->gvData[3 * i + 8]),
                                     sum[1]);
            sum[2] = _mm256_fmadd_ps(_mm256_loadu_ps(&var->pixels[3 * (x + i + 5) + 1 + 3 * var->y * (var->a->dimX + 2 * var->ext + 1)]), _mm256_loadu_ps(&var->gvData[3 * i + 16]),
                                     sum[2]);
        }
        float fsum1 = 0, fsum2 = 0, fsum3 = 0;
        for (; i < var->gv->length - deta; ++i) {
            fsum1 += var->gv->data[i] * get_pixel(var->a, x - var->ext + i, var->y)[0];
            fsum2 += var->gv->data[i] * get_pixel(var->a, x - var->ext + i, var->y)[1];
            fsum3 += var->gv->data[i] * get_pixel(var->a, x - var->ext + i, var->y)[2];
        }
        get_pixel(var->b, x, var->y)[0] = (sum[0][0] + sum[0][3] + sum[0][6] + sum[1][1] + sum[1][4] + sum[1][7] + sum[2][2] + sum[2][5] + fsum1) / var->gv->sum[var->ext - deta];
        get_pixel(var->b, x, var->y)[1] = (sum[0][1] + sum[0][4] + sum[0][7] + sum[1][2] + sum[1][5] + sum[2][0] + sum[2][3] + sum[2][6] + fsum2) / var->gv->sum[var->ext - deta];
        get_pixel(var->b, x, var->y)[2] = (sum[0][2] + sum[0][5] + sum[1][0] + sum[1][3] + sum[1][6] + sum[2][1] + sum[2][4] + sum[2][7] + fsum3) / var->gv->sum[var->ext - deta];
    }
}

Image gb_h(Image *a, FVec gv, float *gvData) {
    Image b = img_sc(*a);
    int ext = gv.length / 2;

    float *pixels = malloc(3 * (a->dimX + 2 * ext + 1) * a->dimY * sizeof(float));

    //#pragma omp parallel for schedule(dynamic) default(none) shared(a, ext, pixels)
    for (int j = 0; j < a->dimY; ++j) {
        for (int i = -ext; i < a->dimX + ext; ++i) {
            pixels[3 * i + 3 * ext + 3 * j * (a->dimX + 2 * ext + 1) + 0] = get_pixel(a, i, j)[0];
            pixels[3 * i + 3 * ext + 3 * j * (a->dimX + 2 * ext + 1) + 1] = get_pixel(a, i, j)[1];
            pixels[3 * i + 3 * ext + 3 * j * (a->dimX + 2 * ext + 1) + 2] = get_pixel(a, i, j)[2];
        }
    }

    //#pragma omp parallel for schedule(dynamic) default(none) shared(a, b, gv, ext, gvData, pixels)
    for (int y = 0; y < a->dimY; ++y) {
        //        thread(&work);
        thpool_add_work(thPool, (void (*)(void *)) thread, &(exeUnit){.y = y, .a = a, .b = &b, .gv = &gv, .ext = ext, .gvData = gvData, .pixels = pixels});
    }
    thpool_wait(thPool);
    free(pixels);
    return b;
}

Image apply_gb(Image a, FVec gv) {
    __attribute__((aligned(64))) float gvData[3 * gv.length + 10];
    //#pragma omp parallel for schedule(dynamic) default(none) shared(gv, gvData)
    for (int i = 0; i < gv.length; ++i) {
        gvData[3 * i + 0] = gv.data[i];
        gvData[3 * i + 1] = gv.data[i];
        gvData[3 * i + 2] = gv.data[i];
    }
    Image b = gb_h(&a, gv, gvData);
    transpose_block(&b, &a);
    b = gb_h(&a, gv, gvData);
    Image c = img_sc(a);
    transpose_block(&b, &c);
    free(b.data);
    free(a.data);
    return c;
}


int main(int argc, char **argv) {
    assert(sysconf(_SC_NPROCESSORS_ONLN) >= 4);
    assert(sysconf(_SC_NPROCESSORS_ONLN) == 4);
    return 0;
}


/*.......................Some notes.........................................*/


//dynamic is very useful instead of choosing static #pragma omp parallel for schedule(dynamic)
//because the time of each iteration is inbalancing
//but whether we can use static//pragma omp parallel for schedule(static)//since each iteration is not such inbalancing
//Dynamic scheduling has some overhead, but can result in better load balancing if iterations not all equal sized.


// OpenMP has a shared memory programming model.
// Some variables are shared and accessible by all threads.
// Other threads are private, and each thread has its own copy.
// Most variables are shared by default.
// Global and static variables are shared.
// Variables declared in master thread shared by default.
// Some variables parallel blocks private by default.
// Loop index of for / parallel for construct.
// Stack variables (e.g. function argument or local variable) created during execution of a parallel region.
// Automatic variables in functions called in parallel region.

//can we use #pragma omp parallel for reduction(+:global_sum)
//

// Reply to @sirius: All above are carefully tested, some of which helps a lot!
//
// FIXED: gb_v is SUPER slow! Much slower than expected due to cache miss issues.
//
// DONE: transpose matrix in gb_v
//      ABORT: Speedup transpose
//          Comment: No need to speedup, it takes a very small portion of program.
// ABORT: change sequence of loop in gb_v
//          Comment: Transposing is far faster than expected.
// DONE: try __mm256, it shall be fast on Autolab.
// DONE: test with a 4999x4999 pic