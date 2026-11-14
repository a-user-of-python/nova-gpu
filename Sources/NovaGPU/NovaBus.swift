// NovaBus.swift — the two-way bus between NovaCPU and NovaGPU.
//
// The bus is fitted to NovaCPU's existing conventions, exactly like the
// console / file / network / timer devices that already live in
// NovaMemory:
//
//   * An MMIO register block at 0xFFFF_FFFF_FFFF_0100 (16 x 8-byte
//     registers) inside the existing MMIO page. NovaCPU.swift needs no
//     change: it already routes every address >= 0xFFFF_FFFF_FFFF_0000
//     to NovaMemory.mmioRead/mmioWrite generically. (In this test app
//     NovaMemory carries a small delimited section forwarding the GPU
//     range to this device; the Nova v0.1 merge lands the same section
//     in NovaVM's NovaMemory.)
//   * A command ring in GUEST RAM: the CPU writes 64-byte descriptors
//     and rings the doorbell; the GPU drains the ring, executes each
//     command, and writes the completion status back INTO the
//     descriptor (shared memory both sides can see).
//   * DMA: the GPU reads/writes guest RAM directly (through
//     NovaMemory's bounds-clamped bulk helpers) for uploads,
//     downloads and the command ring itself.
//   * GPU -> CPU direction: when the GPU needs something only the CPU
//     side can provide (the ASSET service), it suspends the command,
//     writes a request record into a guest-RAM request page, and raises
//     the service IRQ bit. The CPU driver services the request and
//     writes REQ_ACK; the GPU then finishes the suspended command and
//     raises the completion IRQ. Completion and error IRQs work the
//     same way: bits in IRQ_STATUS that the CPU polls and acknowledges
//     (the current NovaCPU has no GPU trap vector; the register model
//     is the contract, and a vector can be attached in the merge the
//     same way the timer's was).
//
// Execution model of THIS implementation: the device NEVER touches
// guest RAM from inside an MMIO write. While NovaCPU executes, its
// hot loop holds NovaMemory's RAM buffer under a modify access
// (withHotBuffer), and any device DMA in that window would trip
// Swift's exclusivity enforcement — the same trap NovaVM's own
// file device is known to hit. So the doorbell and REQ_ACK writes
// only record pending work; draining happens at host service points:
// immediately inside the bus client's submit path (no CPU running
// there), or between CPU run slices when a real NovaCPU drives the
// bus (the host calls servicePendingWork() whenever the CPU yields
// its budget). The protocol itself is asynchronous — descriptors,
// fences and IRQ bits are the synchronisation — so this costs the
// design nothing: a merged version may equally drain on its own
// thread behind the same registers.

import Foundation

// MARK: - Register map (offsets from NovaMemory.gpuBaseAddr)

enum GPURegister: UInt64 {
    case id         = 0x00   // RO: 0x4E475055_00010000 ("NGPU", v1.0)
    case status     = 0x08   // RO: bit0 busy, bit1 ring nonempty, bit2 error,
                             //     bit3 completion IRQ pending, bit4 service
                             //     request pending, bit5 present ready
    case ctrl       = 0x10   // WO: 1 = reset device, 2 = clear error
    case ringBase   = 0x18   // RW: guest address of the command ring
    case ringSlots  = 0x20   // RW: ring size in descriptors (power of 2)
    case doorbell   = 0x28   // WO: write the new ring tail (monotonic)
    case irqStatus  = 0x30   // reads: pending IRQ bits; writes: ack (clear bits)
    case reqPending = 0x38   // RO: sequence number of pending service request, 0 = none
    case reqPage    = 0x40   // RW: guest address of the request page (64 bytes)
    case reqAck     = 0x48   // WO: write the serviced request's sequence number
    case error      = 0x50   // RO: last error code (GPUErrorCode)
    case fence      = 0x58   // RO: last completed fence value
    case scratch    = 0x60   // RW: general scratch (protocol debugging)
}

enum GPUCommand: UInt64 {
    case nop        = 0
    case alloc      = 1    // a0=size -> result handle written to word2
    case free       = 2    // a0=handle
    case upload     = 3    // a0=dst handle, a1=dst off, a2=guest src, a3=len
    case download   = 4    // a0=src handle, a1=src off, a2=guest dst, a3=len
    case fill       = 5    // a0=handle, a1=off, a2=len, a3=pattern u32
    case setVShader = 6    // a0=handle, a1=off, a2=len (bytecode in VRAM)
    case setFShader = 7
    case setCompute = 8
    case setUniform = 9    // a0=index, a1..a4 = vec4 float bits
    case setVBuf    = 10   // a0=handle
    case setIBuf    = 11   // a0=handle
    case setFBuf    = 12   // a0=handle, a1=width, a2=height
    case setTexture = 13   // a0=unit, a1=handle, a2=width, a3=height
    case clear      = 14   // a0=RGBA8888
    case draw       = 15   // a0=index count, a1=flags (bit0 = needsAsset),
                           // a2=asset staging guest addr, a3=asset length,
                           // a4=asset target VRAM handle
    case launch     = 16   // a0=lane count, a1=target buffer handle
    case fenceCmd   = 17   // a0=fence value to publish on completion
    case present    = 18   // a0=framebuffer handle (marks present-ready)
    case plotVectors = 19  // v4 plot unit: a0=index count, a1=mode
                           // (0 = lines from index pairs, 1 = points);
                           // transform = uniforms c0..c3, colour = c10
    case vpuTest    = 20   // v5 in-scene VPU test: a0 != 0 arms the
                           // in-draw VPU test (and resets its result
                           // window); a0 == 0 disarms it
    case governorCmd = 21  // v6: forwarded to the attached
                           // NovaGovernorDevice — a0: 1 = enable,
                           // 2 = disable, 3 = tick (a1 = frame-work
                           // microseconds). The chip also ticks on
                           // every PRESENT by itself; this is the
                           // explicit channel.
}

enum GPUErrorCode: UInt64 {
    case none = 0, unknownCommand = 1, badHandle = 2, outOfMemory = 3
    case outOfBounds = 4, badProgram = 5, badState = 6, dmaFault = 7
}

enum GPUDescriptorStatus: UInt64 {
    case pending = 0, done = 1, waitingService = 2, error = 0xFFFF_FFFF_FFFF_FFFF
}

/// One 64-byte ring descriptor, as laid out in guest RAM:
/// word0 opcode, word1 fence tag, word2..word6 args (word2 doubles as
/// the ALLOC result), word7 status (written by the device).
struct GPUDescriptor {
    var opcode: UInt64 = 0
    var fence: UInt64 = 0
    var args: [UInt64] = [0, 0, 0, 0, 0]
    var status: UInt64 = 0
    static let byteSize = 64
    static func unpack(_ bytes: [UInt8]) -> GPUDescriptor {
        func w(_ i: Int) -> UInt64 {
            var v: UInt64 = 0
            for k in 0..<8 { v |= UInt64(bytes[i * 8 + k]) << UInt64(k * 8) }
            return v
        }
        return GPUDescriptor(opcode: w(0), fence: w(1),
                             args: [w(2), w(3), w(4), w(5), w(6)], status: w(7))
    }
    func packed() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: 64)
        let words = [opcode, fence, args[0], args[1], args[2], args[3], args[4], status]
        for i in 0..<8 {
            for k in 0..<8 { out[i * 8 + k] = UInt8((words[i] >> UInt64(k * 8)) & 0xFF) }
        }
        return out
    }
}

// MARK: - The device

final class NovaGPUDevice {
    let gpu = NovaGPU()
    weak var memory: NovaMemory?

    /// Host display hook (like NovaMemory.consoleOutput): called when a
    /// PRESENT command completes, with the GPU and framebuffer handle.
    var onPresent: ((NovaGPU, UInt64) -> Void)?

    /// v6: the attached supervisor chip, if any. Ring opcode 21
    /// forwards to it, and it is ticked on every PRESENT.
    var governor: NovaGovernorDevice?

    private var ringBase: UInt64 = 0
    private var ringSlots: UInt64 = 0
    private var tail: UInt64 = 0          // last doorbelled tail
    private var head: UInt64 = 0          // next descriptor to execute
    private var irq: UInt64 = 0           // bit0 completion, bit1 error, bit2 service
    private var errorCode: UInt64 = 0
    private var fenceValue: UInt64 = 0
    private var scratch: UInt64 = 0
    private var reqPageAddr: UInt64 = 0
    private var reqSeq: UInt64 = 0        // sequence of the pending request (0 = none)
    private var nextReqSeq: UInt64 = 1
    private var drainPending = false      // doorbell recorded; drain at service point
    private var pendingAck: UInt64 = 0    // REQ_ACK recorded; service at service point
    private var suspended: (index: UInt64, desc: GPUDescriptor)? = nil
    private var presentReady = false
    private(set) var descriptorsExecuted: UInt64 = 0

    /// v7: host wall time the device has spent executing work
    /// since the last consumeBusyMs() — the governor's pure WORK
    /// sensor. It accumulates only inside servicePendingWork(),
    /// so it excludes every wait (pacing, timers, UI) by
    /// construction.
    private var busyAccumMs: Double = 0

    /// Reads and clears the accumulated device busy time.
    func consumeBusyMs() -> Double {
        defer { busyAccumMs = 0 }
        return busyAccumMs
    }

    static let deviceID: UInt64 = 0x4E47_5055_0001_0000

    func attach(to memory: NovaMemory) {
        self.memory = memory
        memory.gpuDevice = self
    }

    func reset() {
        gpu.reset()
        tail = 0; head = 0; irq = 0; errorCode = 0; fenceValue = 0
        reqSeq = 0; suspended = nil; presentReady = false
        drainPending = false; pendingAck = 0
        descriptorsExecuted = 0
        busyAccumMs = 0
    }

    // MARK: MMIO

    func mmioRead(offset: UInt64) -> UInt64 {
        guard let reg = GPURegister(rawValue: offset) else { return 0 }
        switch reg {
        case .id: return Self.deviceID
        case .status:
            var s: UInt64 = 0
            if drainPending || suspended != nil || head < tail { s |= 1 }  // busy
            if head < tail { s |= 2 }                            // ring nonempty
            if errorCode != 0 { s |= 4 }                         // error
            if irq & 1 != 0 { s |= 8 }                           // completion IRQ
            if reqSeq != 0 { s |= 16 }                           // service request
            if presentReady { s |= 32 }                         // present ready
            return s
        case .irqStatus: return irq
        case .reqPending: return reqSeq
        case .ringBase: return ringBase
        case .ringSlots: return ringSlots
        case .reqPage: return reqPageAddr
        case .error: return errorCode
        case .fence: return fenceValue
        case .scratch: return scratch
        default: return 0
        }
    }

    func mmioWrite(offset: UInt64, value: UInt64) {
        guard let reg = GPURegister(rawValue: offset) else { return }
        switch reg {
        case .ctrl:
            if value == 1 { reset() }
            if value == 2 { errorCode = 0; irq &= ~2 }
        case .ringBase: ringBase = value
        case .ringSlots: ringSlots = value
        case .doorbell:
            if value > tail { tail = value }
            drainPending = true   // drained at the next service point
        case .irqStatus: irq &= ~value   // write to 0x30 = acknowledge
        case .reqPage: reqPageAddr = value
        case .reqAck:
            if value != 0 { pendingAck = value }
        case .scratch: scratch = value
        default: break
        }
    }

    // MARK: Ring draining

    /// Host service point: performs any work recorded by doorbell /
    /// REQ_ACK writes. Called by the bus client after each submit, and
    /// by a CPU host between run slices. This is the only place the
    /// device touches guest RAM.
    func servicePendingWork() {
        let t0 = NovaGPU.nowMs()
        if pendingAck != 0 {
            let seq = pendingAck
            pendingAck = 0
            if reqSeq != 0 && seq == reqSeq { serviceRequest() }
        }
        if drainPending {
            drainPending = false
            drain()
        }
        busyAccumMs += NovaGPU.nowMs() - t0
    }

    private func descriptorAddress(_ index: UInt64) -> UInt64? {
        guard let mem = memory, ringSlots > 0, ringBase != 0 else { return nil }
        _ = mem
        return ringBase + (index % ringSlots) * UInt64(GPUDescriptor.byteSize)
    }

    private func readDescriptor(_ index: UInt64) -> GPUDescriptor? {
        guard let mem = memory, let addr = descriptorAddress(index) else { return nil }
        let bytes = mem.loadHostBytes(from: addr, count: GPUDescriptor.byteSize)
        guard bytes.count == GPUDescriptor.byteSize else { return nil }
        return GPUDescriptor.unpack(bytes)
    }

    private func writeDescriptor(_ index: UInt64, _ desc: GPUDescriptor) {
        guard let mem = memory, let addr = descriptorAddress(index) else { return }
        mem.storeHostBytes(desc.packed(), at: addr)
    }

    private func drain() {
        guard suspended == nil else { return }
        while head < tail {
            guard var desc = readDescriptor(head) else {
                fail(code: .dmaFault)
                return
            }
            let outcome = execute(&desc)
            switch outcome {
            case .done:
                desc.status = GPUDescriptorStatus.done.rawValue
                writeDescriptor(head, desc)
                descriptorsExecuted += 1
                if desc.fence != 0 { fenceValue = desc.fence }
                irq |= 1
                head += 1
            case .error(let code):
                desc.status = GPUDescriptorStatus.error.rawValue
                writeDescriptor(head, desc)
                descriptorsExecuted += 1
                fail(code: code)
                head += 1
            case .suspended:
                // Status + request record are already written; the
                // descriptor completes when REQ_ACK resumes it.
                suspended = (index: head, desc: desc)
                return
            }
        }
    }

    private func fail(code: GPUErrorCode) {
        errorCode = code.rawValue
        irq |= 2
    }

    /// The CPU has serviced the pending request: for the ASSET service,
    /// DMA the staged data from guest RAM into the target VRAM buffer,
    /// then re-run the suspended command to completion.
    private func serviceRequest() {
        guard let s = suspended else { reqSeq = 0; return }
        var desc = s.desc
        reqSeq = 0
        irq &= ~4
        suspended = nil
        // ASSET: args were (staging guest addr, length, target handle).
        let staging = desc.args[2], len = desc.args[3], target = desc.args[4]
        if let mem = memory, len > 0 {
            let data = mem.loadHostBytes(from: staging, count: Int(len))
            gpu.vramWrite(at: target, bytes: data)
        }
        // Re-execute with the asset flag cleared.
        desc.args[1] = 0
        let outcome = execute(&desc)
        switch outcome {
        case .done:
            desc.status = GPUDescriptorStatus.done.rawValue
            descriptorsExecuted += 1
            if desc.fence != 0 { fenceValue = desc.fence }
            irq |= 1
        case .error(let code):
            desc.status = GPUDescriptorStatus.error.rawValue
            descriptorsExecuted += 1
            fail(code: code)
        case .suspended:
            desc.status = GPUDescriptorStatus.error.rawValue
            fail(code: .badState)
        }
        writeDescriptor(s.index, desc)
        head += 1
        drain()
    }

    private enum Outcome { case done, error(GPUErrorCode), suspended }

    private func execute(_ desc: inout GPUDescriptor) -> Outcome {
        guard let mem = memory else { return .error(.dmaFault) }
        let cmd = GPUCommand(rawValue: desc.opcode)
        do {
            switch cmd {
            case .nop, .none:
                if cmd == nil { return .error(.unknownCommand) }
            case .alloc:
                let h = try gpu.alloc(size: desc.args[0])
                desc.args[0] = h
            case .free:
                try gpu.free(desc.args[0])
            case .upload:
                guard gpu.allocationSize(desc.args[0]) != nil else { return .error(.badHandle) }
                let data = mem.loadHostBytes(from: desc.args[2], count: Int(desc.args[3]))
                guard data.count == Int(desc.args[3]) else { return .error(.dmaFault) }
                gpu.vramWrite(at: desc.args[0] + desc.args[1], bytes: data)
            case .download:
                guard let size = gpu.allocationSize(desc.args[0]) else { return .error(.badHandle) }
                guard desc.args[1] + desc.args[3] <= size else { return .error(.outOfBounds) }
                let data = gpu.vramRead(at: desc.args[0] + desc.args[1], count: Int(desc.args[3]))
                mem.storeHostBytes(data, at: desc.args[2])
            case .fill:
                guard let size = gpu.allocationSize(desc.args[0]) else { return .error(.badHandle) }
                guard desc.args[1] + desc.args[2] <= size else { return .error(.outOfBounds) }
                gpu.vramFill(at: desc.args[0] + desc.args[1], count: Int(desc.args[2]),
                             pattern: UInt32(desc.args[3] & 0xFFFF_FFFF))
            case .setVShader, .setFShader, .setCompute:
                guard let size = gpu.allocationSize(desc.args[0]) else { return .error(.badHandle) }
                guard desc.args[1] + desc.args[2] <= size, desc.args[2] % 4 == 0 else { return .error(.outOfBounds) }
                let bytes = gpu.vramRead(at: desc.args[0] + desc.args[1], count: Int(desc.args[2]))
                var prog = [UInt32](repeating: 0, count: bytes.count / 4)
                for i in 0..<prog.count {
                    prog[i] = UInt32(bytes[i * 4]) | (UInt32(bytes[i * 4 + 1]) << 8)
                        | (UInt32(bytes[i * 4 + 2]) << 16) | (UInt32(bytes[i * 4 + 3]) << 24)
                }
                if cmd == .setVShader { gpu.vertexProgram = prog }
                else if cmd == .setFShader { gpu.fragmentProgram = prog }
                else { gpu.computeProgram = prog }
            case .setUniform:
                gpu.setUniform(index: Int(desc.args[0]),
                               (Float(bitPattern: UInt32(desc.args[1] & 0xFFFF_FFFF)),
                                Float(bitPattern: UInt32(desc.args[2] & 0xFFFF_FFFF)),
                                Float(bitPattern: UInt32(desc.args[3] & 0xFFFF_FFFF)),
                                Float(bitPattern: UInt32(desc.args[4] & 0xFFFF_FFFF))))
            case .setVBuf:
                guard gpu.allocationSize(desc.args[0]) != nil else { return .error(.badHandle) }
                gpu.vertexBuffer = desc.args[0]
            case .setIBuf:
                guard gpu.allocationSize(desc.args[0]) != nil else { return .error(.badHandle) }
                gpu.indexBuffer = desc.args[0]
            case .setFBuf:
                guard gpu.allocationSize(desc.args[0]) != nil else { return .error(.badHandle) }
                gpu.framebuffer = desc.args[0]
                gpu.fbWidth = Int(desc.args[1]); gpu.fbHeight = Int(desc.args[2])
            case .setTexture:
                guard gpu.allocationSize(desc.args[1]) != nil else { return .error(.badHandle) }
                gpu.textures[Int(desc.args[0])] = (desc.args[1], Int(desc.args[2]), Int(desc.args[3]))
            case .clear:
                try gpu.clearFramebuffer(rgba: UInt32(desc.args[0] & 0xFFFF_FFFF))
            case .draw:
                if desc.args[1] & 1 != 0 {
                    // Needs the ASSET service: ask the CPU, suspend.
                    return suspendForAsset(&desc)
                }
                _ = try gpu.draw(indexCount: Int(desc.args[0]))
            case .plotVectors:
                _ = try gpu.plotVectors(indexCount: Int(desc.args[0]), mode: Int(desc.args[1]))
            case .vpuTest:
                gpu.armVPUTest(desc.args[0] != 0)
            case .launch:
                try gpu.launchCompute(lanes: Int(desc.args[0]), buffer: desc.args[1])
            case .fenceCmd:
                fenceValue = desc.args[0]
            case .governorCmd:
                if let gov = governor {
                    switch desc.args[0] {
                    case 1: gov.chip.setEnabled(true)
                    case 2: gov.chip.setEnabled(false)
                    case 3: gov.chip.tick(GovernorTelemetry(
                                frameWorkMs: Double(desc.args[1]) / 1000))
                    default: break
                    }
                }
            case .present:
                presentReady = true
                onPresent?(gpu, desc.args[0])
                governor?.notePresent()
            }
        } catch let e as GPUError {
            switch e {
            case .outOfMemory: return .error(.outOfMemory)
            case .badHandle: return .error(.badHandle)
            case .outOfBounds: return .error(.outOfBounds)
            case .badProgram: return .error(.badProgram)
            case .badState: return .error(.badState)
            }
        } catch {
            return .error(.badState)
        }
        return .done
    }

    /// Writes the ASSET request record into the guest request page and
    /// raises the service IRQ. Request page layout (64 bytes):
    /// word0 sequence, word1 type (1 = ASSET), word2 staging guest addr,
    /// word3 length, word4 target VRAM handle, word5 fence tag.
    private func suspendForAsset(_ desc: inout GPUDescriptor) -> Outcome {
        guard let mem = memory, reqPageAddr != 0 else { return .error(.badState) }
        reqSeq = nextReqSeq; nextReqSeq += 1
        var bytes = [UInt8](repeating: 0, count: 64)
        let words: [UInt64] = [reqSeq, 1, desc.args[2], desc.args[3], desc.args[4], desc.fence, 0, 0]
        for i in 0..<8 { for k in 0..<8 { bytes[i * 8 + k] = UInt8((words[i] >> UInt64(k * 8)) & 0xFF) } }
        mem.storeHostBytes(bytes, at: reqPageAddr)
        desc.status = GPUDescriptorStatus.waitingService.rawValue
        if let addr = descriptorAddress(head) { mem.storeHostBytes(desc.packed(), at: addr) }
        irq |= 4
        return .suspended
    }
}
