// NovaGPUFast.c — see NovaGPUFast.h for the FP exactness contract.

#include "NovaGPUFast.h"

#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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
void ngp_min_f32(float *d, const float *a, const float *b, int n) {
    // Swift.min(a, b) for Comparable: (b < a) ? b : a.
    for (int i = 0; i < n; i++) d[i] = b[i] < a[i] ? b[i] : a[i];
}
void ngp_max_f32(float *d, const float *a, const float *b, int n) {
    // Swift.max(a, b) for Comparable: (a < b) ? b : a.
    for (int i = 0; i < n; i++) d[i] = a[i] < b[i] ? b[i] : a[i];
}
void ngp_sqrt_f32(float *d, const float *a, int n) {
    for (int i = 0; i < n; i++) { float x = a[i]; d[i] = x <= 0 ? 0.0f : sqrtf(x); }
}
void ngp_rsq_f32(float *d, const float *a, int n) {
    for (int i = 0; i < n; i++) { float x = a[i]; d[i] = x <= 0 ? 0.0f : 1.0f / sqrtf(x); }
}
void ngp_cmplt_f32(float *d, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] < b[i] ? 1.0f : 0.0f;
}
void ngp_sel_f32(float *d, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) { if (a[i] != 0) d[i] = b[i]; }
}

// MARK: - Per-lane vec4 reductions (v4)
//
// Scalar reference order for a dot over components c = 0..<lim:
//   s = 0; for c: s += a[c] * b[c]
// Each product rounds once, each add rounds once, in that order.
// The NEON path keeps exactly this per-lane sequence: it works on
// component PLANES across 4 lanes at a time (vld4q deinterleaves
// [lane][4] into 4 plane vectors), and starts the accumulator at
// zero before adding p0 — so 0 + -0 = +0 matches the scalar loop.

static void ngp_dot_scalar(float *d, const float *a, const float *b,
                           int lanes, int lim) {
    for (int l = 0; l < lanes; l++) {
        const float *av = a + l * 4;
        const float *bv = b + l * 4;
        float s = 0;
        for (int c = 0; c < lim; c++) s += av[c] * bv[c];
        float *dv = d + l * 4;
        dv[0] = s; dv[1] = s; dv[2] = s; dv[3] = s;
    }
}

void ngp_dot3_f32(float *d, const float *a, const float *b, int lanes) {
#if defined(__aarch64__)
    int l = 0;
    for (; l + 4 <= lanes; l += 4) {
        float32x4x4_t va = vld4q_f32(a + l * 4);
        float32x4x4_t vb = vld4q_f32(b + l * 4);
        float32x4_t s = vdupq_n_f32(0);
        s = vaddq_f32(s, vmulq_f32(va.val[0], vb.val[0]));
        s = vaddq_f32(s, vmulq_f32(va.val[1], vb.val[1]));
        s = vaddq_f32(s, vmulq_f32(va.val[2], vb.val[2]));
        float32x4x4_t out = { s, s, s, s };
        vst4q_f32(d + l * 4, out);
    }
    if (l < lanes) ngp_dot_scalar(d + l * 4, a + l * 4, b + l * 4, lanes - l, 3);
#else
    ngp_dot_scalar(d, a, b, lanes, 3);
#endif
}

void ngp_dot4_f32(float *d, const float *a, const float *b, int lanes) {
#if defined(__aarch64__)
    int l = 0;
    for (; l + 4 <= lanes; l += 4) {
        float32x4x4_t va = vld4q_f32(a + l * 4);
        float32x4x4_t vb = vld4q_f32(b + l * 4);
        float32x4_t s = vdupq_n_f32(0);
        s = vaddq_f32(s, vmulq_f32(va.val[0], vb.val[0]));
        s = vaddq_f32(s, vmulq_f32(va.val[1], vb.val[1]));
        s = vaddq_f32(s, vmulq_f32(va.val[2], vb.val[2]));
        s = vaddq_f32(s, vmulq_f32(va.val[3], vb.val[3]));
        float32x4x4_t out = { s, s, s, s };
        vst4q_f32(d + l * 4, out);
    }
    if (l < lanes) ngp_dot_scalar(d + l * 4, a + l * 4, b + l * 4, lanes - l, 4);
#else
    ngp_dot_scalar(d, a, b, lanes, 4);
#endif
}

void ngp_cross_f32(float *d, const float *a, const float *b, int lanes) {
#if defined(__aarch64__)
    int l = 0;
    for (; l + 4 <= lanes; l += 4) {
        float32x4x4_t va = vld4q_f32(a + l * 4);
        float32x4x4_t vb = vld4q_f32(b + l * 4);
        float32x4_t cx = vsubq_f32(vmulq_f32(va.val[1], vb.val[2]),
                                   vmulq_f32(va.val[2], vb.val[1]));
        float32x4_t cy = vsubq_f32(vmulq_f32(va.val[2], vb.val[0]),
                                   vmulq_f32(va.val[0], vb.val[2]));
        float32x4_t cz = vsubq_f32(vmulq_f32(va.val[0], vb.val[1]),
                                   vmulq_f32(va.val[1], vb.val[0]));
        float32x4x4_t out = { cx, cy, cz, vdupq_n_f32(0) };
        vst4q_f32(d + l * 4, out);
    }
    for (; l < lanes; l++) {
        const float *av = a + l * 4;
        const float *bv = b + l * 4;
        float *dv = d + l * 4;
        // Locals first: the op must be correct when rd aliases ra/rb.
        float ax = av[0], ay = av[1], az = av[2];
        float bx = bv[0], by = bv[1], bz = bv[2];
        dv[0] = ay * bz - az * by;
        dv[1] = az * bx - ax * bz;
        dv[2] = ax * by - ay * bx;
        dv[3] = 0;
    }
#else
    for (int l = 0; l < lanes; l++) {
        const float *av = a + l * 4;
        const float *bv = b + l * 4;
        float *dv = d + l * 4;
        float ax = av[0], ay = av[1], az = av[2];
        float bx = bv[0], by = bv[1], bz = bv[2];
        dv[0] = ay * bz - az * by;
        dv[1] = az * bx - ax * bz;
        dv[2] = ax * by - ay * bx;
        dv[3] = 0;
    }
#endif
}

void ngp_vnorm_f32(float *d, const float *a, int lanes) {
    // len2 per lane is accumulated in DOT3 order ((0 + x*x) + y*y)
    // + z*z; the reciprocal root is a per-lane scalar op (IEEE
    // sqrtf + division, identical to the Swift expression); the
    // final scale is one multiply per component.
    for (int l = 0; l < lanes; l++) {
        const float *av = a + l * 4;
        float *dv = d + l * 4;
        float len2 = 0;
        len2 += av[0] * av[0];
        len2 += av[1] * av[1];
        len2 += av[2] * av[2];
        float inv = len2 <= 0 ? 0.0f : 1.0f / sqrtf(len2);
        dv[0] = av[0] * inv;
        dv[1] = av[1] * inv;
        dv[2] = av[2] * inv;
        dv[3] = av[3];
    }
}

void ngp_mat4_f32(float *d, const float *a, const float *u, int lanes) {
#if defined(__aarch64__)
    int l = 0;
    for (; l + 4 <= lanes; l += 4) {
        float32x4x4_t va = vld4q_f32(a + l * 4);
        float32x4_t r0 = vdupq_n_f32(0), r1 = vdupq_n_f32(0);
        float32x4_t r2 = vdupq_n_f32(0), r3 = vdupq_n_f32(0);
        // Component j = dot4(va, row j), accumulated in DOT4 order.
        r0 = vaddq_f32(r0, vmulq_f32(va.val[0], vdupq_n_f32(u[0])));
        r0 = vaddq_f32(r0, vmulq_f32(va.val[1], vdupq_n_f32(u[1])));
        r0 = vaddq_f32(r0, vmulq_f32(va.val[2], vdupq_n_f32(u[2])));
        r0 = vaddq_f32(r0, vmulq_f32(va.val[3], vdupq_n_f32(u[3])));
        r1 = vaddq_f32(r1, vmulq_f32(va.val[0], vdupq_n_f32(u[4])));
        r1 = vaddq_f32(r1, vmulq_f32(va.val[1], vdupq_n_f32(u[5])));
        r1 = vaddq_f32(r1, vmulq_f32(va.val[2], vdupq_n_f32(u[6])));
        r1 = vaddq_f32(r1, vmulq_f32(va.val[3], vdupq_n_f32(u[7])));
        r2 = vaddq_f32(r2, vmulq_f32(va.val[0], vdupq_n_f32(u[8])));
        r2 = vaddq_f32(r2, vmulq_f32(va.val[1], vdupq_n_f32(u[9])));
        r2 = vaddq_f32(r2, vmulq_f32(va.val[2], vdupq_n_f32(u[10])));
        r2 = vaddq_f32(r2, vmulq_f32(va.val[3], vdupq_n_f32(u[11])));
        r3 = vaddq_f32(r3, vmulq_f32(va.val[0], vdupq_n_f32(u[12])));
        r3 = vaddq_f32(r3, vmulq_f32(va.val[1], vdupq_n_f32(u[13])));
        r3 = vaddq_f32(r3, vmulq_f32(va.val[2], vdupq_n_f32(u[14])));
        r3 = vaddq_f32(r3, vmulq_f32(va.val[3], vdupq_n_f32(u[15])));
        float32x4x4_t out = { r0, r1, r2, r3 };
        vst4q_f32(d + l * 4, out);
    }
    for (; l < lanes; l++) {
        const float *av = a + l * 4;
        float *dv = d + l * 4;
        float v0 = av[0], v1 = av[1], v2 = av[2], v3 = av[3];
        for (int j = 0; j < 4; j++) {
            const float *row = u + j * 4;
            float s = 0;
            s += v0 * row[0];
            s += v1 * row[1];
            s += v2 * row[2];
            s += v3 * row[3];
            dv[j] = s;
        }
    }
#else
    for (int l = 0; l < lanes; l++) {
        const float *av = a + l * 4;
        float *dv = d + l * 4;
        float v0 = av[0], v1 = av[1], v2 = av[2], v3 = av[3];
        for (int j = 0; j < 4; j++) {
            const float *row = u + j * 4;
            float s = 0;
            s += v0 * row[0];
            s += v1 * row[1];
            s += v2 * row[2];
            s += v3 * row[3];
            dv[j] = s;
        }
    }
#endif
}

// MARK: - Integer ops (UInt32 view of the register file)

void ngp_f2i_u32(uint32_t *d, const float *a, int n) {
    for (int i = 0; i < n; i++) {
        float v = a[i];
        float t = isfinite(v) ? v : 0.0f;
        d[i] = (uint32_t)(int32_t)t;
    }
}
void ngp_i2f_f32(float *d, const uint32_t *a, int n) {
    for (int i = 0; i < n; i++) d[i] = (float)(int32_t)a[i];
}
void ngp_iadd_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] + b[i];
}
void ngp_isub_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] - b[i];
}
void ngp_imul_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] * b[i];
}
void ngp_iand_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] & b[i];
}
void ngp_ior_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] | b[i];
}
void ngp_ixor_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    for (int i = 0; i < n; i++) d[i] = a[i] ^ b[i];
}
void ngp_ishl_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    // Bit pattern of Int32(x) << (y & 31) — identical bits whether
    // the shift is performed signed or unsigned.
    for (int i = 0; i < n; i++) d[i] = a[i] << (b[i] & 31);
}
void ngp_ishr_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n) {
    // Arithmetic shift of the signed view (clang/gcc define >> on
    // negative int32 as arithmetic, as Swift's Int32 >> is).
    for (int i = 0; i < n; i++) d[i] = (uint32_t)((int32_t)a[i] >> (b[i] & 31));
}
