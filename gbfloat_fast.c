#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <xmmintrin.h>
//implement dynamic

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159


typedef struct FVec {
    unsigned int length;
    unsigned int min_length;
    unsigned int min_deta;
    float *data;
    float *sum;
} FVec;

typedef struct Image {
    unsigned int dimX, dimY, numChannels;
    float *data;
} Image;

void normalize_FVec(FVec v) {
    // float sum = 0.0;
    unsigned int i, j;
    int ext = v.length / 2;
    v.sum[0] = v.data[ext];
    for (i = ext + 1, j = 1; i < v.length; i++, j++) { v.sum[j] = v.sum[j - 1] + v.data[i] * 2; }
    // for (i = 0; i <= ext; i++)
    // {
    //      v.data[i] /= v.sum[v.length - ext - 1 ] ;
    //      printf("%lf ",v.sum[i]);
    // }
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
    return expf((-.5f) * c * c) / (a * sqrt(2 * PI));
}

size_t allocSize(int a) { return 1 << (32 - __builtin_clz(a)); }

FVec make_gv(float a, float x0, float x1, unsigned int length, unsigned int min_length) {
    FVec v;
    v.length = length;
    v.min_length = min_length;
    if (v.min_length > v.length) {
        v.min_deta = 0;
    } else {
        v.min_deta = ((v.length - v.min_length) / 2);
    }
    v.data = malloc(length * sizeof(float));
    //    v.data = aligned_alloc(1024, allocSize(length * sizeof(float)));
    v.sum = malloc((length / 2 + 1) * sizeof(float));
    //    v.sum = aligned_alloc(1024, allocSize((length / 2 + 1) * sizeof(float)));
    float step = (x1 - x0) / ((float) length);
    int offset = length / 2;

    for (int i = 0; i < length; i++) { v.data[i] = gd(a, 0.0f, (i - offset) * step); }
    normalize_FVec(v);
    return v;
}

void print_fvec(FVec v) {
    unsigned int i;
    printf("\n");
    for (i = 0; i < v.length; i++) { printf("%f ", v.data[i]); }
    printf("\n");
}

Image img_sc(Image a) {
    Image b = a;
    b.data = malloc(b.dimX * b.dimY * b.numChannels * sizeof(float));
    //    b.data = aligned_alloc(1024, b.dimX * b.dimY * b.numChannels * sizeof(float));
    return b;
}

Image gb_h(Image a, FVec gv) {
    Image b = img_sc(a);
    int ext = gv.length / 2;

#pragma omp parallel for schedule(dynamic) default(none) shared(a, b, gv, ext)
    for (int y = 0; y < a.dimY; y++) {
        for (int x = 0; x < a.dimX; x++) {
            int deta = fminf(fminf(fminf(a.dimY - y - 1, y), fminf(a.dimX - x - 1, x)), gv.min_deta);
            float fsum1 = 0, fsum2 = 0, fsum3 = 0;
            __m128 pixel0, pixel1, pixel2, pixel3;
            __m128 gvData0, gvData1, gvData2, gvData3;
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();
            __m128 sum2 = _mm_setzero_ps();
            __m128 sum3 = _mm_setzero_ps();
            int i;
            for (i = deta; i < gv.length - deta - 4; i += 4) {
                pixel0 = _mm_loadu_ps(get_pixel(a, x - ext + i, y));
                pixel1 = _mm_loadu_ps(get_pixel(a, x - ext + i + 1, y));
                pixel2 = _mm_loadu_ps(get_pixel(a, x - ext + i + 2, y));
                pixel3 = _mm_loadu_ps(get_pixel(a, x - ext + i + 3, y));

                gvData0 = _mm_load1_ps(&gv.data[i]);
                gvData1 = _mm_load1_ps(&gv.data[i + 1]);
                gvData2 = _mm_load1_ps(&gv.data[i + 2]);
                gvData3 = _mm_load1_ps(&gv.data[i + 3]);

                sum0 = _mm_fmadd_ps(pixel0, gvData0, sum0);
                sum1 = _mm_fmadd_ps(pixel1, gvData1, sum1);
                sum2 = _mm_fmadd_ps(pixel2, gvData2, sum2);
                sum3 = _mm_fmadd_ps(pixel3, gvData3, sum3);

                //                sum0 = _mm_add_ps(sum0, _mm_mul_ps(pixel0, gvData0));
                //                sum1 = _mm_add_ps(sum1, _mm_mul_ps(pixel1, gvData1));
                //                sum2 = _mm_add_ps(sum2, _mm_mul_ps(pixel2, gvData2));
                //                sum3 = _mm_add_ps(sum3, _mm_mul_ps(pixel3, gvData3));
            }

            for (; i < gv.length - deta; ++i) {
                fsum1 += gv.data[i] * get_pixel(a, x - ext + i, y)[0];
                fsum2 += gv.data[i] * get_pixel(a, x - ext + i, y)[1];
                fsum3 += gv.data[i] * get_pixel(a, x - ext + i, y)[2];
            }
            get_pixel(b, x, y)[0] = (sum0[0] + sum1[0] + sum2[0] + sum3[0] + fsum1) / gv.sum[ext - deta];
            get_pixel(b, x, y)[1] = (sum0[1] + sum1[1] + sum2[1] + sum3[1] + fsum2) / gv.sum[ext - deta];
            get_pixel(b, x, y)[2] = (sum0[2] + sum1[2] + sum2[2] + sum3[2] + fsum3) / gv.sum[ext - deta];
        }
    }
    return b;
}

Image gb_v(Image a, FVec gv) {
    Image b = img_sc(a);
    int ext = gv.length / 2;

#pragma omp parallel for schedule(dynamic) default(none) shared(a, b, gv, ext)
    for (int y = 0; y < a.dimY; y++) {
        for (int x = 0; x < a.dimX; x++) {
            int deta = fminf(fminf(fminf(a.dimY - y - 1, y), fminf(a.dimX - x - 1, x)), gv.min_deta);
            float fsum1 = 0, fsum2 = 0, fsum3 = 0;
            __m128 pixel0, pixel1, pixel2, pixel3;
            __m128 gvData0, gvData1, gvData2, gvData3;
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();
            __m128 sum2 = _mm_setzero_ps();
            __m128 sum3 = _mm_setzero_ps();
            int i;
            for (i = deta; i < gv.length - deta - 4; i += 4) {
                pixel0 = _mm_loadu_ps(get_pixel(a, x, y - ext + i));
                pixel1 = _mm_loadu_ps(get_pixel(a, x, y - ext + i + 1));
                pixel2 = _mm_loadu_ps(get_pixel(a, x, y - ext + i + 2));
                pixel3 = _mm_loadu_ps(get_pixel(a, x, y - ext + i + 3));

                gvData0 = _mm_load1_ps(&gv.data[i]);
                gvData1 = _mm_load1_ps(&gv.data[i + 1]);
                gvData2 = _mm_load1_ps(&gv.data[i + 2]);
                gvData3 = _mm_load1_ps(&gv.data[i + 3]);

                sum0 = _mm_fmadd_ps(pixel0, gvData0, sum0);
                sum1 = _mm_fmadd_ps(pixel1, gvData1, sum1);
                sum2 = _mm_fmadd_ps(pixel2, gvData2, sum2);
                sum3 = _mm_fmadd_ps(pixel3, gvData3, sum3);
            }

            for (; i < gv.length - deta; ++i) {
                fsum1 += gv.data[i] * get_pixel(a, x, y - ext + i)[0];
                fsum2 += gv.data[i] * get_pixel(a, x, y - ext + i)[1];
                fsum3 += gv.data[i] * get_pixel(a, x, y - ext + i)[2];
            }
            get_pixel(b, x, y)[0] = (sum0[0] + sum1[0] + sum2[0] + sum3[0] + fsum1) / gv.sum[ext - deta];
            get_pixel(b, x, y)[1] = (sum0[1] + sum1[1] + sum2[1] + sum3[1] + fsum2) / gv.sum[ext - deta];
            get_pixel(b, x, y)[2] = (sum0[2] + sum1[2] + sum2[2] + sum3[2] + fsum3) / gv.sum[ext - deta];
        }
    }
    return b;
}

Image apply_gb(Image a, FVec gv) {
    Image b = gb_h(a, gv);
    Image c = gb_v(b, gv);
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
    unsigned int dim, min_dim;

    sscanf(argv[3], "%f", &a);       /* 0.6 */
    sscanf(argv[4], "%f", &x0);      /* -2.0 */
    sscanf(argv[5], "%f", &x1);      /* 2.0 */
    sscanf(argv[6], "%u", &dim);     /* 1001 */
    sscanf(argv[7], "%u", &min_dim); /* 201 */

    FVec v = make_gv(a, x0, x1, dim, min_dim);

    //    print_fvec(v);
    Image img;
    img.data = stbi_loadf(argv[1], &(img.dimX), &(img.dimY), &(img.numChannels), 0);

    Image imgOut = apply_gb(img, v);
    stbi_write_jpg(argv[2], imgOut.dimX, imgOut.dimY, imgOut.numChannels, imgOut.data, 90);
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);
    printf("%f \n", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);
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