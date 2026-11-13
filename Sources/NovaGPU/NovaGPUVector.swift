// NovaGPUVector.swift — the v4 vector hardware: grab unit, VPU
// (vector processing unit) pool, and the plot unit.
//
// Number formats (Sean's spec, 2026-11-13 — superseding the brief
// 6-bit-per-axis idea, which never reached code):
//   * World-space math: FP32 — the same precision as the shader
//     pipeline. Chosen over 16.16 fixed-point because it matches
//     the pipeline bit-for-bit, which is what makes the plot path
//     provably pixel-identical to projecting through the full
//     triangle pipeline (rounding the final clip product once,
//     in the same order, yields the same vertex).
//   * Screen-space coordinates: SCREEN_AXIS_BITS = 16 bits per
//     axis — the plot unit converts projected endpoints into a
//     Q(16-6).6 fixed-point grid (SCREEN_FRAC_BITS fractional
//     bits) and rasterises in integer space on that grid.
//
// Architecture: a VPU processes ONE triangle's vertex positions.
// It fetches 3 vertices (grab unit), runs the 3-axis calculator
// over the 9 axis values (3 points × X/Y/Z) — the same FP32
// matrix math (ngp_mat4) the shader stage's VMAT4 executes — and
// hands the positioned triangle onward. The VPUs are grouped
// into MGVPU clusters (Sean's numbers, 2026-11-13):
//
//     MAX_SCREEN_VECTORS = 65,536          (per-frame plot budget)
//     MGVPU_COUNT = 50                     (clusters)
//     VPUS_PER_MGVPU = 131,072             (VPUs per cluster)
//     addressable pool = 50 × 131,072 = 6,553,600 VPUs
//
// All four numbers live on NovaGPU so they change in exactly one
// place. The pool is virtual: a VPU is a stateless datapath (the
// 48-byte struct below — one workload's registers), demand-
// allocated on the stack of the host thread running the workload
// and scheduled by whole MGVPU cluster sub-ranges. There is no
// array of VPU objects and no idle cost; the capacity is an
// addressable space, and MAX_SCREEN_VECTORS bounds the plot
// unit's per-frame vector budget.

/// The grab unit (v4): a vertex-fetch path specialised for pulling
/// 3D vector attributes out of VRAM. Where the v3 vertex stage
/// fetched through a per-lane Swift closure (one call per lane per
/// attribute slot), the grab unit performs wide sequential loads
/// from the draw's vertex snapshot and expands each vertex into a
/// fixed [slot][vec4] record — vec3 attributes padded to vec4
/// (position w = 1, normal w = 0, uv zw = 0), colour as-is — which
/// the executor then reads through its pointer fast path with
/// zero per-lane calls. The values are bit-identical to what the
/// v3 closure returned for every slot.
enum GrabUnit {
    /// Floats per grabbed vertex record (4 attribute slots × vec4).
    static let recordStride = 16

    /// Grab `count` vertices starting at vertex `base` from the
    /// interleaved snapshot (12 floats per vertex: position xyz,
    /// normal xyz, uv, colour rgba) into `dst` as grabbed records.
    static func grabWarp(snapshot src: UnsafePointer<Float>, base: Int,
                         count: Int, into dst: UnsafeMutablePointer<Float>) {
        for i in 0..<count {
            let s = src + (base + i) * 12
            let d = dst + i * recordStride
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 1        // position, w = 1
            d[4] = s[3]; d[5] = s[4]; d[6] = s[5]; d[7] = 0        // normal, w = 0
            d[8] = s[6]; d[9] = s[7]; d[10] = 0; d[11] = 0         // uv -> (u, v, 0, 0)
            d[12] = s[8]; d[13] = s[9]; d[14] = s[10]; d[15] = s[11] // colour
        }
    }

    /// Grab just the positions of `count` vertices (the VPU/plot
    /// fetch): [vertex][vec4] with w = 1, matching the position
    /// slot of a full grab.
    static func grabPositions(snapshot src: UnsafePointer<Float>, base: Int,
                              count: Int, into dst: UnsafeMutablePointer<Float>) {
        for i in 0..<count {
            let s = src + (base + i) * 12
            let d = dst + i * 4
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 1
        }
    }

    /// Grab one triangle workload's positions: 3 INDEXED vertices
    /// (the VPU's fetch — a line loads its second endpoint twice,
    /// a point three times) into [3][vec4] with w = 1.
    static func grabTrianglePositions(snapshot src: UnsafePointer<Float>,
                                      _ i0: Int, _ i1: Int, _ i2: Int,
                                      into dst: UnsafeMutablePointer<Float>) {
        let idx = (i0, i1, i2)
        for v in 0..<3 {
            let vi = v == 0 ? idx.0 : (v == 1 ? idx.1 : idx.2)
            let s = src + vi * 12
            let d = dst + v * 4
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = 1
        }
    }
}

/// Screen-space coordinate format for the plot unit: each axis is
/// a SCREEN_AXIS_BITS-wide fixed-point value with SCREEN_FRAC_BITS
/// fractional bits (Q10.6 at the defaults: integer range 0..<1024
/// pixels, 64 sub-pixel steps per pixel — covers the framebuffer
/// sizes this GPU presents; the grid itself is a named parameter).
enum ScreenSpace {
    static let axisBits = 16        // SCREEN_AXIS_BITS
    static let fracBits = 6         // SCREEN_FRAC_BITS
    static let scale: Float = 64    // 1 << fracBits

    /// Convert a projected NDC point to grid coordinates: NDC → pixel
    /// float (the triangle pipeline's viewport transform — the same
    /// expressions screenOf uses in draw()) → grid by one FP32
    /// multiply and a truncation (Float→Int truncates toward zero).
    /// The pixel float is clamped
    /// to a generous margin around the framebuffer first: Float→Int
    /// traps on out-of-range or non-finite values, and off-screen
    /// endpoints only need to stay ordered for the rasteriser —
    /// the binning stage clips them, as it clips triangles.
    @inline(__always)
    static func gridX(_ ndcX: Float, width: Int) -> Int {
        let px = (ndcX + 1) * 0.5 * Float(width)
        guard px.isFinite else { return -(1 << 20) }
        return Int(Swift.min(Swift.max(px, -4096), Float(width) + 4096) * scale)
    }
    @inline(__always)
    static func gridY(_ ndcY: Float, height: Int) -> Int {
        let py = (1 - ndcY) * 0.5 * Float(height)
        guard py.isFinite else { return -(1 << 20) }
        return Int(Swift.min(Swift.max(py, -4096), Float(height) + 4096) * scale)
    }
    /// Grid coordinate → pixel index (arithmetic shift: the grid
    /// value of a visible pixel is non-negative, so this is exact
    /// division by the sub-pixel scale).
    @inline(__always)
    static func pixel(_ grid: Int) -> Int { grid >> fracBits }
}

/// The MGVPU cluster layer (Sean's grouping, 2026-11-13): the
/// VPU pool is organised as MGVPU_COUNT = 50 clusters of
/// VPUS_PER_MGVPU = 131,072 VPUs — 6,553,600 addressable VPUs in
/// total. An MGVPU cluster is the unit of work distribution: a
/// plot command's vector range is split into one contiguous
/// sub-range per cluster and the sub-ranges run concurrently
/// across host threads, which keeps each thread's grab traffic
/// sequential and local. A VPU address is (mgvpu, vpu); the
/// global id is mgvpu * VPUS_PER_MGVPU + vpu. Clusters hold no
/// state of their own — like the VPUs, they are an address space
/// and a scheduling partition, materialized only as the stack
/// work slots of the threads running them.
enum MGVPU {
    static var count: Int { NovaGPU.mgvpuCount }              // 50
    static var vpusPerCluster: Int { NovaGPU.vpusPerMgvpu }   // 131,072
    static var totalVPUs: Int { NovaGPU.vpuCount }            // 6,553,600

    /// Global VPU id for an (mgvpu, vpu) address.
    static func globalID(mgvpu: Int, vpu: Int) -> Int { mgvpu * vpusPerCluster + vpu }
    /// The (mgvpu, vpu) address of a global VPU id.
    static func address(ofGlobalID id: Int) -> (mgvpu: Int, vpu: Int) {
        (id / vpusPerCluster, id % vpusPerCluster)
    }

    /// Split `count` workloads into one contiguous sub-range per
    /// cluster, in order (range k covers
    /// [count*k/clusters, count*(k+1)/clusters)).
    static func subRanges(count: Int) -> [(Int, Int)] {
        let n = Swift.max(count, 0)
        let c = Swift.max(self.count, 1)
        return (0..<c).map { k in (n * k / c, n * (k + 1) / c) }
    }
}

/// One VPU: positions ONE triangle. Holds the triangle's 9 axis
/// values (3 vertices × X/Y/Z) plus homogeneous w, runs the 3-axis
/// calculator (the shared ngp_mat4 FP32 transform), and yields the
/// positioned (clip-space) triangle.
struct VPU {
    /// The 9 axis values of the current triangle workload:
    /// vertices 0..2 × axes X, Y, Z, at [vertex * 4 + axis] with
    /// the homogeneous w interleaved at +3 (grab-unit layout).
    private(set) var axes: (Float, Float, Float, Float,
                            Float, Float, Float, Float,
                            Float, Float, Float, Float) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    /// Load a triangle workload: 3 grabbed position records
    /// ([vertex][vec4], w = 1) from the grab unit.
    mutating func load(positions p: UnsafePointer<Float>) {
        axes = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11])
    }

    /// Run the 3-axis calculator: clip = M * position for each of
    /// the 3 vertices, M = 16 row-major FP32 uniforms. Writes the
    /// 3 positioned vertices ([vertex][vec4] clip space) to `out`.
    /// The math is ngp_mat4 — bit-identical to the shader stage's
    /// VMAT4 and to a DOT4-row chain (proven in host tests).
    func position(matrix m: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>) {
        withUnsafePointer(to: axes) { ap in
            ngp_mat4_f32(out, UnsafeRawPointer(ap).assumingMemoryBound(to: Float.self), m, 3)
        }
    }
}

/// The plot unit (v4): a fast path for plotting 3D vectors as
/// points and lines (vector-display style) that bypasses the
/// triangle pipeline. For each vector: the grab unit fetches the
/// endpoint positions, a VPU positions them (FP32 clip space),
/// endpoints convert to the 16-bit-per-axis screen grid, and a
/// line/point rasteriser draws straight into the tile bins —
/// fragments then shade through the normal fragment path with a
/// constant-colour program, so plotted vectors blend with the 3D
/// scene exactly like triangles do.
///
/// Pixel-exactness contract (host-tested): a plotted vector's
/// endpoint pixels equal the pixels the same endpoints land on
/// when projected through the full pipeline and rounded to the
/// grid; the drawn pixel SET equals the set produced by a
/// Double-precision implementation of the same DDA spec, which
/// the host tests carry as the reference.
enum PlotUnit {
    /// Rasterise one vector (two grid endpoints) with a DDA that
    /// steps at PIXEL resolution over the grid: steps = the
    /// Chebyshev distance between the endpoints measured in whole
    /// pixels (grid distance / 64, rounded up), sample positions
    /// interpolated in grid space (Float) and rounded via
    /// Int(value + 0.5) truncation, so endpoints land exactly on
    /// their grid coordinates and the chain advances at most one
    /// pixel per step on the major axis. (Sampling every grid step
    /// instead would visit ~64× more points than there are pixels
    /// — the grid is the coordinate precision, not the sampling
    /// rate.) If a rounding boundary ever makes a sample jump two
    /// pixels, the midpoint pixel is emitted too, so the drawn
    /// chain is 8-connected by construction. Samples outside the
    /// framebuffer are dropped by the caller, as the triangle
    /// pipeline's binning drops them. The callback also receives
    /// the step index and count so the caller can interpolate
    /// depth along the vector (midpoint inserts report the same
    /// step as the sample that triggered them).
    static func rasterizeLine(x0: Int, y0: Int, x1: Int, y1: Int,
                              emit: (Int, Int, Int, Int) -> Void) {
        let sub = 1 << ScreenSpace.fracBits   // grid units per pixel (64)
        let dx = x1 - x0, dy = y1 - y0
        let gridSteps = max(abs(dx), abs(dy))
        if gridSteps == 0 {
            emit(x0, y0, 0, 0)
            return
        }
        let steps = (gridSteps + sub - 1) / sub
        let fs = Float(steps)
        var lastPx = Int.min, lastPy = Int.min
        for s in 0...steps {
            let t = Float(s) / fs
            let gx = Float(x0) + Float(dx) * t
            let gy = Float(y0) + Float(dy) * t
            let ex = Int(gx + 0.5), ey = Int(gy + 0.5)
            let px = ex >> ScreenSpace.fracBits, py = ey >> ScreenSpace.fracBits
            if lastPx != Int.min
                && (abs(px - lastPx) > 1 || abs(py - lastPy) > 1) {
                // Gap insurance: fill the midpoint pixel so the
                // chain stays 8-connected.
                emit(((lastPx + px) / 2) << ScreenSpace.fracBits,
                     ((lastPy + py) / 2) << ScreenSpace.fracBits, s, steps)
            }
            emit(ex, ey, s, steps)
            lastPx = px; lastPy = py
        }
    }
}
