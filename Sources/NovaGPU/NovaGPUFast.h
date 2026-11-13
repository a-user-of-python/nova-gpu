// NovaGPUFast.h — C hot core for NovaGPU's warp executor (speed round 3,
// extended in v4).
//
// These functions implement the executor's vector ops. FP EXACTNESS
// RULES (the framebuffer checksums are the judge):
//   * Never compiled with fast-math; builds pass -ffp-contract=off.
//   * Every function performs exactly the IEEE-754 binary32 operations
//     the Swift executor performed, in the same order: one rounding
//     per operation, no FMA fusion, no reassociation. MAD is written
//     as two statements so the multiply rounds before the add under
//     ANY compiler flags (validated bit-exact even with contraction
//     forced on).
//   * Auto-vectorisation of these loops (NEON on arm64, SSE/AVX on
//     x86) is bit-safe: each output element is an independent single
//     IEEE operation, and no conforming compiler may reassociate FP
//     without fast-math flags.
//   * The v4 reduction ops (DOT3/DOT4/VCROSS/VNORM/VMAT4) use explicit
//     NEON intrinsics on arm64. Their per-lane accumulation ORDER is
//     the scalar order (s = 0; s += p0; s += p1; ...) — the vector
//     code starts from a zero vector and adds each product in the
//     same sequence, so every lane's result is bit-identical to the
//     scalar loop, including the +0/-0 corner (0 + -0 = +0). A
//     portable scalar fallback with the same order covers non-arm64.
//
// Swift binds these via @_silgen_name declarations in
// NovaGPUShader.swift; every build that compiles the Swift sources
// also compiles and links NovaGPUFast.c (CI swiftc lines and the app
// target both do).

#ifndef NOVAGPU_FAST_H
#define NOVAGPU_FAST_H

#include <stdint.h>

// Component-wise ops over n floats (n = active lanes * 4 in the warp
// executor; the pointers address the executor's [reg][lane] register
// file, where a register occupies 128 consecutive floats).
void ngp_add_f32(float *d, const float *a, const float *b, int n);
void ngp_sub_f32(float *d, const float *a, const float *b, int n);
void ngp_mul_f32(float *d, const float *a, const float *b, int n);
void ngp_div_f32(float *d, const float *a, const float *b, int n);
// rd = ra * rb + rd (accumulating multiply-add, two roundings).
void ngp_mad_f32(float *d, const float *a, const float *b, int n);
void ngp_rcp_f32(float *d, const float *a, int n);
void ngp_mov_f32(float *d, const float *a, int n);
// v4: the remaining component-wise ops, same expressions as the v1
// Swift executor (MIN/MAX use the Comparable comparison form, so a
// NaN operand propagates exactly as it did in Swift).
void ngp_min_f32(float *d, const float *a, const float *b, int n);
void ngp_max_f32(float *d, const float *a, const float *b, int n);
void ngp_sqrt_f32(float *d, const float *a, int n);   // x <= 0 ? 0 : sqrtf(x)
void ngp_rsq_f32(float *d, const float *a, int n);    // x <= 0 ? 0 : 1/sqrtf(x)
void ngp_cmplt_f32(float *d, const float *a, const float *b, int n);
void ngp_sel_f32(float *d, const float *a, const float *b, int n); // d = (a != 0) ? b : d

// Per-lane vec4 ops over the [lane][4] register layout (lanes =
// active lanes in the warp). DOT results are splatted to all four
// components, as the ISA specifies.
void ngp_dot3_f32(float *d, const float *a, const float *b, int lanes);
void ngp_dot4_f32(float *d, const float *a, const float *b, int lanes);
// VCROSS: rd.xyz = cross(ra.xyz, rb.xyz) (per-component mul/sub,
// two roundings each), rd.w = 0.
void ngp_cross_f32(float *d, const float *a, const float *b, int lanes);
// VNORM: rd.xyz = ra.xyz * inv, inv = (len2 <= 0) ? 0 : 1/sqrtf(len2),
// len2 accumulated in DOT3 order; rd.w = ra.w (pass-through).
void ngp_vnorm_f32(float *d, const float *a, int lanes);
// VMAT4: rd = M * ra, M = 4 uniform rows u[0..15] (row-major);
// component j = dot4(ra, row j) in DOT4 accumulation order.
void ngp_mat4_f32(float *d, const float *a, const float *u, int lanes);

// Integer ops over the register file's UInt32 view. F2I matches the
// v1 expression UInt32(bitPattern: Int32(v.isFinite ? v : 0)) for
// in-range values; shifts mask counts to 31, ISHR is arithmetic.
void ngp_f2i_u32(uint32_t *d, const float *a, int n);
void ngp_i2f_f32(float *d, const uint32_t *a, int n);
void ngp_iadd_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_isub_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_imul_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_iand_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_ior_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_ixor_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_ishl_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);
void ngp_ishr_u32(uint32_t *d, const uint32_t *a, const uint32_t *b, int n);

#endif
