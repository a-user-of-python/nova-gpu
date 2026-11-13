// NovaGPUStandard.swift — standard shader programs, matrix helpers and
// procedural geometry shared by the host tests and the test app.
//
// Uniform layout for the lit shaders:
//   c0..c3  model-view-projection matrix rows (vec4 each)
//   c4..c7  component masks (1,0,0,0) (0,1,0,0) (0,0,1,0) (0,0,0,1)
//   c8      light direction xyz (normalised), w unused
//   c9      (1,1,1,1)

import Foundation

enum NovaGPUStandard {

    /// Vertex shader: clip = MVP * pos (assembled from DOT4 splats with
    /// mask SELs), normal rotated by the MVP upper 3x3, colour and uv
    /// passed through. OUT 0 = clip, OUT 1 = colour, OUT 2 = normal,
    /// OUT 3 = uv.
    static func vertexLitProgram() -> [UInt32] {
        var b = ShaderBuilder()
        b.ldattr(0, 0)            // r0 = position (x, y, z, 1)
        b.ldunif(1, 0); b.dot4(4, 0, 1)
        b.ldunif(1, 1); b.dot4(5, 0, 1)
        b.ldunif(1, 2); b.dot4(6, 0, 1)
        b.ldunif(1, 3); b.dot4(7, 0, 1)
        b.movi(8, 0)              // r8 = clip accumulator (0,0,0,0)
        b.ldunif(1, 4); b.sel(8, 1, 4)
        b.ldunif(1, 5); b.sel(8, 1, 5)
        b.ldunif(1, 6); b.sel(8, 1, 6)
        b.ldunif(1, 7); b.sel(8, 1, 7)
        b.out(0, 8)
        // Normal: rotate by rows 0..2 with DOT3.
        b.ldattr(2, 1)            // r2 = normal
        b.ldunif(1, 0); b.dot3(4, 2, 1)
        b.ldunif(1, 1); b.dot3(5, 2, 1)
        b.ldunif(1, 2); b.dot3(6, 2, 1)
        b.movi(9, 0)
        b.ldunif(1, 4); b.sel(9, 1, 4)
        b.ldunif(1, 5); b.sel(9, 1, 5)
        b.ldunif(1, 6); b.sel(9, 1, 6)
        b.out(2, 9)
        b.ldattr(3, 3); b.out(1, 3)   // colour
        b.ldattr(3, 2); b.out(3, 3)   // uv
        b.end()
        return b.build()
    }

    /// Fragment shader: diffuse-lit colour. shade = 0.25 + 0.75 *
    /// max(dot(n, light), 0); alpha forced to the interpolated alpha.
    /// v4 lit vertex program: identical outputs to
    /// vertexLitProgram(), with the clip transform computed by one
    /// VMAT4 (the vector unit's matrix op) instead of four DOT4s
    /// plus the SEL component assembly. Bit-identical by
    /// construction — VMAT4's components ARE the DOT4 rows — and
    /// proven so at frame level in the host tests (the same scene
    /// rendered through both programs must produce identical
    /// framebuffer bytes).
    static func vertexLitProgramV4() -> [UInt32] {
        var b = ShaderBuilder()
        b.ldattr(0, 0)            // r0 = position (x, y, z, 1)
        b.vmat4(8, 0, rows: 0)    // r8 = MVP * position
        b.out(0, 8)
        // Normal: rotate by rows 0..2 with DOT3 (as v1).
        b.ldattr(2, 1)            // r2 = normal
        b.ldunif(1, 0); b.dot3(4, 2, 1)
        b.ldunif(1, 1); b.dot3(5, 2, 1)
        b.ldunif(1, 2); b.dot3(6, 2, 1)
        b.movi(9, 0)
        b.ldunif(1, 4); b.sel(9, 1, 4)
        b.ldunif(1, 5); b.sel(9, 1, 5)
        b.ldunif(1, 6); b.sel(9, 1, 6)
        b.out(2, 9)
        b.ldattr(3, 3); b.out(1, 3)   // colour
        b.ldattr(3, 2); b.out(3, 3)   // uv
        b.end()
        return b.build()
    }

    static func fragmentLitProgram() -> [UInt32] {
        var b = ShaderBuilder()
        b.ldattr(0, 1)            // colour
        b.ldattr(1, 2)            // normal
        b.ldunif(2, 8)            // light dir
        b.dot3(3, 1, 2)           // diffuse (splat)
        b.movf(4, 0); b.max(3, 3, 4)
        b.movf(5, 0.75); b.mul(3, 3, 5)
        b.movf(5, 0.25); b.add(3, 3, 5)
        b.ldunif(6, 7)            // mask w
        b.ldunif(7, 9)            // ones
        b.sel(3, 6, 7)            // shade.w = 1 so alpha passes through
        b.mul(0, 0, 3)
        b.out(0, 0)
        b.end()
        return b.build()
    }

    /// Textured fragment variant: colour = texture(uv) * diffuse shade.
    static func fragmentTexProgram() -> [UInt32] {
        var b = ShaderBuilder()
        b.ldattr(0, 3)            // uv
        b.ldtex(0, 0, unit: 0)    // r0 = texel
        b.ldattr(1, 2)
        b.ldunif(2, 8)
        b.dot3(3, 1, 2)
        b.movf(4, 0); b.max(3, 3, 4)
        b.movf(5, 0.75); b.mul(3, 3, 5)
        b.movf(5, 0.25); b.add(3, 3, 5)
        b.ldunif(6, 7); b.ldunif(7, 9); b.sel(3, 6, 7)
        b.mul(0, 0, 3)
        b.out(0, 0)
        b.end()
        return b.build()
    }

    /// Compute kernel for the self-test: in-place y[i] = alpha * x[i] + y[i]
    /// over a buffer laid out [x: N floats][y: N floats].
    /// c0.x bits = byte offset of the y region (N * 4). c1 = (alpha x4).
    static func saxpyProgram() -> [UInt32] {
        var b = ShaderBuilder()
        b.ldattr(0, 0)            // gid (float)
        b.f2i(0, 0)               // gid as int bits
        b.movi(1, 4)
        b.imul(0, 0, 1)           // r0.x = byte offset of x[gid]
        b.ldbuf(2, 0)             // x
        b.ldunif(4, 0)
        b.iadd(5, 0, 4)           // r5.x = byte offset of y[gid]
        b.ldbuf(6, 5)             // y
        b.ldunif(7, 1)            // alpha
        b.mad(6, 2, 7)            // y = x * alpha + y
        b.stbuf(5, 6)
        b.end()
        return b.build()
    }

    // MARK: - Matrices ([Float] 16, row-major; applied as row dot vec)

    static func matIdentity() -> [Float] {
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }
    static func matMul(_ a: [Float], _ b: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: 16)
        for r in 0..<4 { for c in 0..<4 {
            var s: Float = 0
            for k in 0..<4 { s += a[r * 4 + k] * b[k * 4 + c] }
            out[r * 4 + c] = s
        } }
        return out
    }
    static func matScale(_ s: Float) -> [Float] {
        [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]
    }
    static func matTranslate(_ x: Float, _ y: Float, _ z: Float) -> [Float] {
        [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1]
    }
    static func matRotY(_ a: Float) -> [Float] {
        let c = cosf(a), s = sinf(a)
        return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]
    }
    static func matRotX(_ a: Float) -> [Float] {
        let c = cosf(a), s = sinf(a)
        return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1]
    }
    /// Simple perspective: w = 1 + z * k (keeps the shader honest —
    /// the divide happens in the rasteriser), z mapped toward 0..1.
    static func matPerspective(k: Float) -> [Float] {
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0, 0, k, 1]
    }

    static func setLitUniforms(_ client: NovaBusClient, mvp: [Float],
                               light: (Float, Float, Float) = (0.577, 0.577, 0.577)) throws {
        for r in 0..<4 {
            try client.setUniform(r, (mvp[r * 4], mvp[r * 4 + 1], mvp[r * 4 + 2], mvp[r * 4 + 3]))
        }
        try client.setUniform(4, (1, 0, 0, 0))
        try client.setUniform(5, (0, 1, 0, 0))
        try client.setUniform(6, (0, 0, 1, 0))
        try client.setUniform(7, (0, 0, 0, 1))
        try client.setUniform(8, (light.0, light.1, light.2, 0))
        try client.setUniform(9, (1, 1, 1, 1))
    }

    // MARK: - Geometry (NVF1: 12 floats per vertex)

    struct Mesh { var vertices: [Float]; var indices: [UInt32]
                  var vertexCount: Int { vertices.count / 12 }
                  var triangleCount: Int { indices.count / 3 } }

    private static func pushVert(_ v: inout [Float], p: (Float, Float, Float),
                                 n: (Float, Float, Float), uv: (Float, Float),
                                 c: (Float, Float, Float, Float)) {
        v.append(contentsOf: [p.0, p.1, p.2, n.0, n.1, n.2, uv.0, uv.1, c.0, c.1, c.2, c.3])
    }

    /// Icosphere: positions = normals, colour from normal direction.
    static func icosphere(subdivisions: Int, radius: Float = 1) -> Mesh {
        let t = (1 + sqrtf(5)) / 2
        var verts: [(Float, Float, Float)] = [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ]
        var faces: [(Int, Int, Int)] = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]
        func norm(_ p: (Float, Float, Float)) -> (Float, Float, Float) {
            let l = sqrtf(p.0 * p.0 + p.1 * p.1 + p.2 * p.2)
            return (p.0 / l, p.1 / l, p.2 / l)
        }
        verts = verts.map(norm)
        var midCache: [Int: Int] = [:]
        func midpoint(_ a: Int, _ b: Int) -> Int {
            let key = min(a, b) * 100000 + max(a, b)
            if let hit = midCache[key] { return hit }
            let pa = verts[a], pb = verts[b]
            let m = norm(((pa.0 + pb.0) / 2, (pa.1 + pb.1) / 2, (pa.2 + pb.2) / 2))
            verts.append(m)
            midCache[key] = verts.count - 1
            return verts.count - 1
        }
        for _ in 0..<subdivisions {
            var next: [(Int, Int, Int)] = []
            for f in faces {
                let ab = midpoint(f.0, f.1), bc = midpoint(f.1, f.2), ca = midpoint(f.2, f.0)
                next.append((f.0, ab, ca)); next.append((f.1, bc, ab))
                next.append((f.2, ca, bc)); next.append((ab, bc, ca))
            }
            faces = next
        }
        var out: [Float] = []
        for p in verts {
            let n = p
            let u = atan2f(p.2, p.0) / (2 * Float.pi) + 0.5
            let v = 0.5 - asinf(min(max(p.1, -1), 1)) / Float.pi
            pushVert(&out, p: (p.0 * radius, p.1 * radius, p.2 * radius), n: n,
                     uv: (u, v), c: (n.0 * 0.5 + 0.5, n.1 * 0.5 + 0.5, n.2 * 0.5 + 0.5, 1))
        }
        return Mesh(vertices: out, indices: faces.flatMap { [UInt32($0.0), UInt32($0.1), UInt32($0.2)] })
    }

    /// Torus in the XZ plane.
    static func torus(rings: Int, slices: Int, radius: Float = 0.9, tube: Float = 0.34) -> Mesh {
        var out: [Float] = []
        var idx: [UInt32] = []
        for i in 0...rings {
            let a = Float(i) / Float(rings) * 2 * Float.pi
            for j in 0...slices {
                let b = Float(j) / Float(slices) * 2 * Float.pi
                let cx = cosf(a), sx = sinf(a)
                let center: (Float, Float, Float) = (radius * cx, 0, radius * sx)
                let n = (cosf(b) * cx, sinf(b), cosf(b) * sx)
                let p = (center.0 + tube * n.0, center.1 + tube * n.1, center.2 + tube * n.2)
                pushVert(&out, p: p, n: n, uv: (Float(i) / Float(rings), Float(j) / Float(slices)),
                         c: (0.9, 0.55 + 0.4 * n.1, 0.25, 1))
            }
        }
        let cols = slices + 1
        for i in 0..<rings { for j in 0..<slices {
            let a = UInt32(i * cols + j), bq = UInt32((i + 1) * cols + j)
            idx.append(contentsOf: [a, bq, a + 1, bq, bq + 1, a + 1])
        } }
        return Mesh(vertices: out, indices: idx)
    }

    /// The 24-vertex cube used by the Nova-assembly bus proof. The
    /// driver embeds exactly this vertex data (as .word bit patterns)
    /// and produces it on the GPU's ASSET request; the host test
    /// renders the same cube over the bus and compares checksums.
    static func proofCube() -> Mesh {
        let s: Float = 0.62
        // face: normal, 4 corners (counter-clockwise seen from outside)
        let faces: [((Float, Float, Float), [(Float, Float, Float)], (Float, Float, Float, Float))] = [
            ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)], (1, 0.25, 0.2, 1)),
            ((0, 0, -1), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)], (0.2, 1, 0.35, 1)),
            ((1, 0, 0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)], (0.25, 0.45, 1, 1)),
            ((-1, 0, 0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)], (1, 0.85, 0.25, 1)),
            ((0, 1, 0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)], (0.9, 0.35, 1, 1)),
            ((0, -1, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)], (0.3, 0.95, 0.95, 1)),
        ]
        var out: [Float] = []
        var idx: [UInt32] = []
        for (fi, f) in faces.enumerated() {
            let base = UInt32(fi * 4)
            for (ci, corner) in f.1.enumerated() {
                pushVert(&out, p: corner, n: f.0,
                         uv: (Float(ci % 2), Float(ci / 2)), c: f.2)
            }
            idx.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
        return Mesh(vertices: out, indices: idx)
    }

    /// Bytes for a mesh's vertex buffer (NVF1 floats, little-endian).
    static func vertexBytes(_ m: Mesh) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: m.vertices.count * 4)
        for i in 0..<m.vertices.count {
            let bits = m.vertices[i].bitPattern
            out[i * 4] = UInt8(bits & 0xFF); out[i * 4 + 1] = UInt8((bits >> 8) & 0xFF)
            out[i * 4 + 2] = UInt8((bits >> 16) & 0xFF); out[i * 4 + 3] = UInt8((bits >> 24) & 0xFF)
        }
        return out
    }
    static func indexBytes(_ m: Mesh) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: m.indices.count * 4)
        for i in 0..<m.indices.count {
            let bits = m.indices[i]
            out[i * 4] = UInt8(bits & 0xFF); out[i * 4 + 1] = UInt8((bits >> 8) & 0xFF)
            out[i * 4 + 2] = UInt8((bits >> 16) & 0xFF); out[i * 4 + 3] = UInt8((bits >> 24) & 0xFF)
        }
        return out
    }
    static func programBytes(_ p: [UInt32]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: p.count * 4)
        for i in 0..<p.count {
            out[i * 4] = UInt8(p[i] & 0xFF); out[i * 4 + 1] = UInt8((p[i] >> 8) & 0xFF)
            out[i * 4 + 2] = UInt8((p[i] >> 16) & 0xFF); out[i * 4 + 3] = UInt8((p[i] >> 24) & 0xFF)
        }
        return out
    }

    /// FNV-1a checksum over bytes (used to compare framebuffers).
    static func fnv1a(_ bytes: [UInt8]) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for b in bytes { h = (h ^ UInt64(b)) &* 0x100000001b3 }
        return h
    }
}
