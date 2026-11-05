// NovaGPUFast.h — C hot core for NovaGPU's warp executor (speed round 3).
//
// These functions implement the executor's component-wise FP32 vector
// ops. FP EXACTNESS RULES (the framebuffer checksums are the judge):
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
//
// Swift binds these via @_silgen_name declarations in
// NovaGPUShader.swift; every build that compiles the Swift sources
// also compiles and links NovaGPUFast.c (CI swiftc lines and the app
// target both do).

#ifndef NOVAGPU_FAST_H
#define NOVAGPU_FAST_H

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

#endif
