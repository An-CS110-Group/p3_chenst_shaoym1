#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <xmmintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


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

void normalize_FVec(FVec v) {
    unsigned int i, j;
    int ext = v.length / 2;
    v.sum[0] = v.data[ext];
    for (i = ext + 1, j = 1; i < v.length; i++, j++) { v.sum[j] = v.sum[j - 1] + v.data[i] * 2; }
}

float *get_pixel(Image img, int x, int y) {
    if (x < 0) { x = 0; }
    if (x >= img.dimX) { x = img.dimX - 1; }
    if (y < 0) { y = 0; }
    if (y >= img.dimY) { y = img.dimY - 1; }
    return img.data + img.numChannels * (y * img.dimX + x);
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

void transpose_block(Image *src, Image *dst) {
    dst->dimX = src->dimY;
    dst->dimY = src->dimX;
    dst->numChannels = src->numChannels;
#pragma omp parallel for schedule(dynamic) default(none) shared(src, dst)
    for (int i = 0; i < src->dimX; ++i) {
        for (int j = 0; j < src->dimY; ++j) {
            dst->data[(dst->dimX * i + j) * dst->numChannels + 0] = src->data[(src->dimX * j + i) * src->numChannels + 0];
            dst->data[(dst->dimX * i + j) * dst->numChannels + 1] = src->data[(src->dimX * j + i) * src->numChannels + 1];
            dst->data[(dst->dimX * i + j) * dst->numChannels + 2] = src->data[(src->dimX * j + i) * src->numChannels + 2];
        }
    }
}

Image img_sc(Image a) {
    Image b = a;
    b.data = malloc(b.dimX * b.dimY * b.numChannels * sizeof(float) + sizeof(float));
    return b;
}

Image gb_h(Image a, FVec gv) {
    Image b = img_sc(a);
    int ext = gv.length / 2;

#pragma omp parallel for schedule(dynamic) default(none) shared(a, b, gv, ext)
    for (int y = 0; y < a.dimY; y++) {
        for (int x = 0; x < a.dimX; x++) {
            int deta = (int) MIN(MIN(MIN(a.dimY - y - 1, y), MIN(a.dimX - x - 1, x)), (float) gv.min_deta);
            float fsum1 = 0, fsum2 = 0, fsum3 = 0;
            __m256 sum[8] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                             _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            int i;
            for (i = deta; i < gv.length - deta - 8; i += 8) {
                sum[0] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i, y)), _mm256_set1_ps(gv.data[i]), sum[0]);
                //                sum[1] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 1, y)), _mm256_set1_ps(gv.data[i + 1]), sum[1]);
                sum[2] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 2, y)), _mm256_set1_ps(gv.data[i + 2]), sum[2]);
                //                sum[3] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 3, y)), _mm256_set1_ps(gv.data[i + 3]), sum[3]);
                sum[4] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 4, y)), _mm256_set1_ps(gv.data[i + 4]), sum[4]);
                //                sum[5] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 5, y)), _mm256_set1_ps(gv.data[i + 5]), sum[5]);
                sum[6] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 6, y)), _mm256_set1_ps(gv.data[i + 6]), sum[6]);
                //                sum[7] = _mm256_fmadd_ps(_mm256_loadu_ps(get_pixel(a, x - ext + i + 7, y)), _mm256_set1_ps(gv.data[i + 7]), sum[7]);
            }

            for (; i < gv.length - deta; ++i) {
                fsum1 += gv.data[i] * get_pixel(a, x - ext + i, y)[0];
                fsum2 += gv.data[i] * get_pixel(a, x - ext + i, y)[1];
                fsum3 += gv.data[i] * get_pixel(a, x - ext + i, y)[2];
            }
            get_pixel(b, x, y)[0] =
                    (sum[0][0] + sum[1][0] + sum[2][0] + sum[3][0] + sum[4][0] + sum[5][0] + sum[6][0] + sum[7][0] + fsum1) / gv.sum[ext - deta];
            get_pixel(b, x, y)[1] =
                    (sum[0][1] + sum[1][1] + sum[2][1] + sum[3][1] + sum[4][1] + sum[5][1] + sum[6][1] + sum[7][1] + fsum2) / gv.sum[ext - deta];
            get_pixel(b, x, y)[2] =
                    (sum[0][2] + sum[1][2] + sum[2][2] + sum[3][2] + sum[4][2] + sum[5][2] + sum[6][2] + sum[7][2] + fsum3) / gv.sum[ext - deta];
        }
    }
    return b;
}

Image apply_gb(Image a, FVec gv) {
    Image b = gb_h(a, gv);
    transpose_block(&b, &a);
    b = gb_h(a, gv);
    Image c = img_sc(a);
    transpose_block(&b, &c);
    free(b.data);
    free(a.data);
    return c;
}

int main(int argc, char **argv) {
    struct timeval start_time, stop_time, elapsed_time;
    gettimeofday(&start_time, NULL);
    if (argc < 6) {
        printf("Usage: ./gb.exe <inputjpg> <outputname> <float: a> <float: x0> <float: x1> <unsigned int: dim>\n");
        exit(0);
    }

    float a, x0, x1;
    int dim, min_dim;

    sscanf(argv[3], "%f", &a);       /* 0.6 */
    sscanf(argv[4], "%f", &x0);      /* -2.0 */
    sscanf(argv[5], "%f", &x1);      /* 2.0 */
    sscanf(argv[6], "%d", &dim);     /* 1001 */
    sscanf(argv[7], "%d", &min_dim); /* 201 */

    FVec v = make_gv(a, x0, x1, dim, min_dim);

    //    print_fvec(v);
    Image img;
    img.data = stbi_loadf(argv[1], &(img.dimX), &(img.dimY), &(img.numChannels), 0);

    Image imgOut = apply_gb(img, v);
    stbi_write_jpg(argv[2], imgOut.dimX, imgOut.dimY, imgOut.numChannels, imgOut.data, 90);
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);
    printf("%f \n", (double) elapsed_time.tv_sec + (double) elapsed_time.tv_usec / 1000000.0);
    free(imgOut.data);
    free(v.data);
    free(v.sum);
    return 0;
}


/*.......................Some notes.........................................*/


// __m128i _mm_set1_epi32(int i):
// Set the four signed 32-bit integers within the vector to i.
//  __m128i _mm_loadu_si128( __m128i *p):
// Load the 4 successive ints pointed to by p into a 128-bit vector.
//  __m128i _mm_mullo_epi32(__m128i a, __m128i b):
// Return vector (a0 · b0, a1 · b1, a2 · b2, a3 · b3).
// __m128i _mm_add_epi32(__m128i a, __m128i b):
// Return vector (a0 + b0, a1 + b1, a2 + b2, a3 + b3)
//  void _mm_storeu_si128( __m128i *p, __m128i a):
// Store 128-bit vector a at pointer p.
// __m128i _mm_and_si128(__m128i a, __m128i b):
// Perform a bitwise AND of 128 bits in a and b, and return the result.
//  __m128i _mm_cmpeq_epi32(__m128i a, __m128i b):
// The ith element of the return vector will be set to 0xFFFFFFFF if the ith
// elements of a and b are equal, otherwise it’ll be set to 0.

//

//can l change all float to float
//do some minus optimization (change all exp to expf and fmin to fminf)
//“Pragma”: stands for “pragmatic information.
//A pragma is a way to communicate the information to the compiler.

/*can we parallel manually instesd of using #pragma omp parallel for schedule(dynamic) default(none) private(y) shared(a, x, b, pc, gv, ext, deta, i)*/
//somethinglike
// #pragma omp parallel {
// int id, i, Nthreads, start, end;
// id = omp_get_thread_num();
// Nthreads = omp_get_num_threads();
// start = id * N / Nthreads;
// end = (id + 1) * N / Nthreads;
// for (i = start; i < end; i++) {
// a[i] = a[i] + b[i];
// }
// }

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
// FIXME: gb_v is SUPER slow! Much slower than expected due to cache miss issues.
//
// DONE: transpose matrix in gb_v
//      ABORT: Speedup transpose
//          Comment: No need to speedup, it takes a very small portion of program.
// TODO: change sequence of loop in gb_v
// TODO: try __mm256, it shall be fast on Autolab.
// DONE: test with a 4999x4999 pic