#ifndef WKV_CUDA_H
#define WKV_CUDA_H

#ifndef Tmax
#define Tmax 512*64
#endif

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv);

#endif
