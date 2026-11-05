// NovaGPUFast.c — see NovaGPUFast.h for the FP exactness contract.

#include "NovaGPUFast.h"

void ngp_add_f32(float *d, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] + b[i];
}
void ngp_sub_f32(float *d, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] - b[i];
}
void ngp_mul_f32(float *d, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] * b[i];
}
void ngp_div_f32(float *d, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] / b[i];
}
void ngp_mad_f32(float *d, const float *a, const float *b, int n) {
    // Two statements on purpose: the product must round to binary32
    // before the add, exactly as the Swift expression evaluated.
    for (int i = 0; i < n; i++) { float t = a[i] * b[i]; d[i] = t + d[i]; }
}
void ngp_rcp_f32(float *d, const float *a, int n) {
    for (int i = 0; i < n; i++) d[i] = 1.0f / a[i];
}
void ngp_mov_f32(float *d, const float *a, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i];
}
