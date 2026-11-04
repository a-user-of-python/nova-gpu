// NovaMemory.swift — Guest physical memory for the NOVA VM.
// Uses a single ContiguousArray<UInt8> for cache-friendly access.
// Stack regions grow downward from their bases; program loaded at low addresses.
//
// SECURITY: All accessors are bounds-checked. Out-of-bounds access throws
// NovaTrap.loadAccessFault / .storeAccessFault instead of crashing the host.

import Foundation

// ============================================================================
// HARNESS VARIANT MEMORY — identical to vm-app/NovaVM/VM/NovaMemory.swift
// (V1) except for ONE added method, withHotBuffer (see its comment), which
// the variant CPU's hot loop uses for direct buffer access. Nothing else
// differs; trap kinds, ranges and MMIO behaviour are V1's, byte for byte.
// ============================================================================

// MARK: - Trap Causes (shared by CPU and memory)
enum NovaTrap: UInt64, Error {
    case instrAddrMisaligned = 0
    case instrAccessFault    = 1
    case illegalInstruction  = 2
    case breakpoint          = 3
    case loadAddrMisaligned  = 4
    case loadAccessFault     = 5
    case storeAddrMisaligned = 6
    case storeAccessFault    = 7
    case ecallU              = 8
    case ecallS              = 9
    case ecallH              = 10
    case ecallM              = 11
    case instrPageFault      = 12
    case loadPageFault       = 13
    case storePageFault      = 14
    case stackOverflow       = 15
    case stackUnderflow      = 16
    case timerInterrupt      = 17  // Hardware timer fired (for firmware scheduler)
}

final class NovaMemory {

    // MARK: - Layout
    // [0x0            ] program + data (grows up)
    // [dataStackBase  ] data stack (grows down from base)
    // [callStackBase  ] call stack (grows down)
    // [vectorStackBase] vector stack (grows down, 16-byte entries)

    let totalSize: Int
    private let sizeU64: UInt64
    /// Precomputed last valid start address for multi-byte accesses
    /// (sizeU64 - 2 / - 4 / - 8). Lets each multi-byte accessor range-check
    /// with ONE comparison (`addr <= maxStartN`) instead of the old
    /// three-part guard; the two are exactly equivalent for these sizes
    /// (proof in the accessor section comment below). totalSize is clamped
    /// to >= 64 MB in init, so these subtractions can never underflow.
    private let maxStart16: UInt64
    private let maxStart32: UInt64
    private let maxStart64: UInt64
    private var bytes: ContiguousArray<UInt8>

    let dataStackBase: UInt64
    let callStackBase: UInt64
    let vectorStackBase: UInt64

    static let pageSize = 4096

    /// Hard cap: the VM can never allocate more than 1 GB of device RAM.
    static let maxTotalSize = 1024 * 1024 * 1024

    init(totalSize: Int = 64 * 1024 * 1024) {
        // Clamp into [64MB, 1GB]. 64MB minimum ensures stack regions fit.
        let clamped = min(max(totalSize, 64 * 1024 * 1024), Self.maxTotalSize)
        self.totalSize = clamped
        self.sizeU64 = UInt64(clamped)
        self.maxStart16 = UInt64(clamped) - 2
        self.maxStart32 = UInt64(clamped) - 4
        self.maxStart64 = UInt64(clamped) - 8
        self.bytes = ContiguousArray<UInt8>(repeating: 0, count: clamped)

        let top = UInt64(clamped)
        // Reserve stack space proportionally, capped at 8MB/16MB.
        // For 64MB: reserve 4MB/8MB. For >=128MB: use full 8MB/16MB.
        let reserveUnit = min(UInt64(8 * 1024 * 1024), top / 16)
        self.vectorStackBase = top
        self.callStackBase   = top &- reserveUnit
        self.dataStackBase   = top &- (reserveUnit * 2)
    }

    // MARK: - Bounds-checked access (little-endian)
    //
    // v0.0.70 fast path. Each accessor does ONE range check, then (for
    // multi-byte accesses) a single constant-size memcpy of the value's
    // bytes — under -O that lowers to one unaligned load/store — through a
    // buffer pointer fetched fresh for this access (never stored across
    // calls, so ContiguousArray growth/reallocation can never stale it).
    //
    // Why guest semantics are identical:
    //  * Range check: for access size n in {2,4,8} and RAM size S
    //    (S >= 64 MB by the init clamp, so S - n never underflows),
    //      addr <= S - n   <=>   addr < S && addr + n <= S,
    //    because addr + n cannot overflow in the passing case
    //    (addr <= S <= 1 GB). That is exactly the old three-part guard
    //    (addr < S, n <= S, addr &+ n <= S) for these sizes. read8/write8
    //    keep the equivalent single comparison addr < S. An access faults
    //    iff any of its bytes is out of bounds, exactly as before — this
    //    covers accesses spanning the RAM/MMIO boundary and accesses
    //    partially out of RAM. MMIO-page addresses (>= the console
    //    address) fail this check just as they failed the old guard;
    //    MMIO routing itself stays in NovaCPU, untouched.
    //  * Atomicity: the check still runs BEFORE any byte is touched, so a
    //    faulting access modifies nothing and a passing access transfers
    //    all its bytes — same as the checked byte-wise code.
    //  * Endianness: the original loops assemble/disassemble LSB-first
    //    (byte at addr+0 is the least significant byte) — guest memory is
    //    little-endian. Copying the host representation of a UInt16/32/64
    //    preserves that order because every supported host (arm64 and
    //    x86_64) is little-endian as well.

    @inline(__always) func read8(_ addr: UInt64) throws -> UInt8 {
        guard addr < sizeU64 else { throw NovaTrap.loadAccessFault }
        return bytes[Int(addr)]
    }

    @inline(__always) func read16(_ addr: UInt64) throws -> UInt16 {
        guard addr <= maxStart16 else { throw NovaTrap.loadAccessFault }
        let i = Int(addr)
        var v: UInt16 = 0
        bytes.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memcpy(&v, base + i, 2)
        }
        return v
    }

    @inline(__always) func read32(_ addr: UInt64) throws -> UInt32 {
        guard addr <= maxStart32 else { throw NovaTrap.loadAccessFault }
        let i = Int(addr)
        var v: UInt32 = 0
        bytes.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memcpy(&v, base + i, 4)
        }
        return v
    }

    @inline(__always) func read64(_ addr: UInt64) throws -> UInt64 {
        guard addr <= maxStart64 else { throw NovaTrap.loadAccessFault }
        let i = Int(addr)
        var v: UInt64 = 0
        bytes.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memcpy(&v, base + i, 8)
        }
        return v
    }

    @inline(__always) func write8(_ addr: UInt64, _ v: UInt8) throws {
        guard addr < sizeU64 else { throw NovaTrap.storeAccessFault }
        bytes[Int(addr)] = v
    }

    @inline(__always) func write16(_ addr: UInt64, _ v: UInt16) throws {
        guard addr <= maxStart16 else { throw NovaTrap.storeAccessFault }
        let i = Int(addr)
        var tmp = v
        bytes.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memcpy(base + i, &tmp, 2)
        }
    }

    @inline(__always) func write32(_ addr: UInt64, _ v: UInt32) throws {
        guard addr <= maxStart32 else { throw NovaTrap.storeAccessFault }
        let i = Int(addr)
        var tmp = v
        bytes.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memcpy(base + i, &tmp, 4)
        }
    }

    @inline(__always) func write64(_ addr: UInt64, _ v: UInt64) throws {
        guard addr <= maxStart64 else { throw NovaTrap.storeAccessFault }
        let i = Int(addr)
        var tmp = v
        bytes.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memcpy(base + i, &tmp, 8)
        }
    }

    func copyBytes(from src: UInt64, to dst: UInt64, count: UInt64) throws {
        guard count <= sizeU64,
              src < sizeU64, src &+ count <= sizeU64,
              dst < sizeU64, dst &+ count <= sizeU64 else {
            throw NovaTrap.loadAccessFault
        }
        let n = Int(count)
        bytes.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            memmove(base + Int(dst), base + Int(src), n)
        }
    }

    // MARK: - Program loading

    func loadBytes(_ data: Data, at address: UInt64) throws {
        guard address < sizeU64 else { throw NovaTrap.loadAccessFault }
        let n = min(data.count, Int(sizeU64 - address))
        guard n > 0 else { return }
        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let sp = src.baseAddress else { return }
            bytes.withUnsafeMutableBufferPointer { dst in
                guard let dp = dst.baseAddress else { return }
                memcpy(dp + Int(address), sp, n)
            }
        }
    }

    func loadProgram(_ data: Data) throws {
        try loadBytes(data, at: 0)
    }

    // MARK: - Host bulk transfers (for the app lane's MMIO handlers)
    //
    // The file/net MMIO payload handlers live in NovaVMEngine.swift (and
    // are mirrored in tools/harness/main.swift) — NOT in this file. They
    // currently copy payloads byte-at-a-time via read8/write8 with try?.
    // These helpers give them a single-memcpy path with IDENTICAL
    // observable semantics to those byte loops:
    //  * store: the byte loop writes each in-bounds byte and silently
    //    ignores faults (try?), so the net effect is "the prefix of the
    //    payload lying inside RAM is written, the rest dropped". Plain
    //    write8 faults on MMIO-page addresses, so such bytes are dropped
    //    here too — never routed to mmioWrite.
    //  * load: the byte loop stops at the first faulting byte, so the
    //    result is the in-RAM prefix starting at `addr` (empty when
    //    `addr` is out of RAM) — a contiguous range, so clamping the
    //    count to the end of RAM reproduces it exactly.
    // Handlers that report the full REQUESTED byte count (as the current
    // file/net handlers do) keep doing so from their own bookkeeping;
    // the store helpers merely also return the count actually written.

    /// Copies host bytes into guest RAM at `addr` with a single memcpy.
    /// Returns the number of bytes actually written (the in-RAM prefix).
    @discardableResult
    func storeHostBytes(_ hostBytes: [UInt8], at addr: UInt64) -> Int {
        guard addr < sizeU64, !hostBytes.isEmpty else { return 0 }
        let n = min(hostBytes.count, Int(sizeU64 - addr))
        hostBytes.withUnsafeBufferPointer { src in
            guard let sp = src.baseAddress else { return }
            bytes.withUnsafeMutableBufferPointer { dst in
                guard let dp = dst.baseAddress else { return }
                memcpy(dp + Int(addr), sp, n)
            }
        }
        return n
    }

    /// Data overload of storeHostBytes(_:at:) — same semantics.
    @discardableResult
    func storeHostBytes(_ data: Data, at addr: UInt64) -> Int {
        guard addr < sizeU64, !data.isEmpty else { return 0 }
        let n = min(data.count, Int(sizeU64 - addr))
        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let sp = src.baseAddress else { return }
            bytes.withUnsafeMutableBufferPointer { dst in
                guard let dp = dst.baseAddress else { return }
                memcpy(dp + Int(addr), sp, n)
            }
        }
        return n
    }

    /// Copies up to `count` guest bytes out of RAM at `addr` with a single
    /// memcpy. Returns the in-RAM prefix (empty if `addr` is out of RAM
    /// or `count` <= 0), matching the stop-at-first-fault byte loop.
    func loadHostBytes(from addr: UInt64, count: Int) -> [UInt8] {
        guard addr < sizeU64, count > 0 else { return [] }
        let n = min(count, Int(sizeU64 - addr))
        var out = [UInt8](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { dst in
            guard let dp = dst.baseAddress else { return }
            bytes.withUnsafeBufferPointer { src in
                guard let sp = src.baseAddress else { return }
                memcpy(dp, sp + Int(addr), n)
            }
        }
        return out
    }

    // MARK: - Hot-loop buffer access (harness-variant support)
    //
    // Added for the tools/harness variants (hotloop/dispatch); NOT part
    // of the V1 shipping core. Runs `body` with a mutable pointer to
    // the guest RAM buffer. The pointer is valid ONLY for the duration
    // of the call and must not escape it.
    //
    // Why handing out the buffer for a whole interpreter run is safe
    // here: `bytes` is assigned exactly once, in init, and NO code path
    // in this file ever resizes, appends to, removes from, or
    // reassigns it afterwards — every mutation is an in-place subscript
    // store or a memcpy/memmove through a buffer pointer (grep this
    // file for `bytes` to re-verify). The storage therefore cannot
    // move while `body` runs. The CPU side is single-threaded (the
    // engine runs the VM on one thread). Host MMIO handlers invoked
    // from inside `body` use the ordinary checked accessors; they
    // mutate the same fixed storage in place, which is compatible
    // with the outstanding pointer for the same no-resize reason.
    //
    // This accessor performs NO bounds checking: the caller (the
    // variant CPU's hot loop) reproduces each checked accessor's range
    // comparison itself before touching the buffer, which is what
    // keeps its fault behaviour identical to the checked accessors.
    func withHotBuffer<T>(_ body: (UnsafeMutablePointer<UInt8>) -> T) -> T {
        return bytes.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else {
                preconditionFailure("NovaMemory: guest RAM buffer is empty")
            }
            return body(base)
        }
    }

    // MARK: - Console I/O (memory-mapped)

    static let consoleAddr: UInt64 = 0xFFFF_FFFF_FFFF_0000
    static let keyboardAddr: UInt64 = 0xFFFF_FFFF_FFFF_0008
    static let fileCmdAddr: UInt64 = 0xFFFF_FFFF_FFFF_0010
    static let fileArg0Addr: UInt64 = 0xFFFF_FFFF_FFFF_0018
    static let fileArg1Addr: UInt64 = 0xFFFF_FFFF_FFFF_0020
    static let fileArg2Addr: UInt64 = 0xFFFF_FFFF_FFFF_0028
    static let fileResultAddr: UInt64 = 0xFFFF_FFFF_FFFF_0030
    // Hardware timer (for firmware scheduler). Firmware writes interval in cycles.
    // Write 0 to disable. When the timer fires, the CPU takes a Machine trap
    // and vectors to the firmware's trap handler (hardware behavior, not app control).
    static let timerIntervalAddr: UInt64 = 0xFFFF_FFFF_FFFF_0040
    static let timerCountAddr: UInt64 = 0xFFFF_FFFF_FFFF_0048
    // Timer vector: firmware writes its handler address here (like mtvec).
    // On timer interrupt, the CPU jumps to this address.
    static let timerVectorAddr: UInt64 = 0xFFFF_FFFF_FFFF_0050
    // Network device (HTTP GET). App implements the hardware; firmware
    // drives it purely in Nova machine code via these MMIO registers.
    static let netCmdAddr: UInt64 = 0xFFFF_FFFF_FFFF_0058      // write 1 = GET
    static let netUrlAddr: UInt64 = 0xFFFF_FFFF_FFFF_0060      // guest addr of NUL-terminated URL
    static let netBufAddr: UInt64 = 0xFFFF_FFFF_FFFF_0068      // guest addr of response buffer
    static let netBufSizeAddr: UInt64 = 0xFFFF_FFFF_FFFF_0070  // max response bytes
    static let netResultAddr: UInt64 = 0xFFFF_FFFF_FFFF_0078   // bytes written, or -1 on error
    // NovaGPU (NovaBus) device block. Added for the NovaGPU test app in
    // exactly the style of the devices above; in the Nova v0.1 merge this
    // same block lands in NovaVM's NovaMemory. NovaCPU.swift needs NO
    // change: it already routes every address >= consoleAddr to
    // mmioRead/mmioWrite generically. 16 registers x 8 bytes.
    static let gpuBaseAddr: UInt64 = 0xFFFF_FFFF_FFFF_0100
    static let gpuEndAddr: UInt64  = 0xFFFF_FFFF_FFFF_0180

    var consoleOutput: ((String) -> Void)?

    /// Keyboard input queue (host pushes keys, guest pops via MMIO read)
    var keyboardQueue: [UInt8] = []
    /// Guards keyboardQueue: pushKey runs on the main (UI) thread,
    /// mmioRead runs on the VM thread. Without this, rapid input
    /// corrupts the array and duplicates/drops bytes.
    private let keyboardLock = NSLock()

    /// File operation handler (set by host)
    var fileHandler: ((UInt64, UInt64, UInt64, UInt64) -> Int64)?

    // File command args
    private var fileArg0: UInt64 = 0
    private var fileArg1: UInt64 = 0
    private var fileArg2: UInt64 = 0
    private var fileResult: Int64 = 0

    /// Hardware timer interval in CPU cycles (0 = disabled). Set by firmware via MMIO.
    /// The CPU checks this and fires a timer interrupt when the cycle count reaches it.
    var timerInterval: UInt64 = 0
    /// Timer countdown (cycles remaining). Firmware can read via MMIO.
    var timerCount: UInt64 = 0
    /// Timer vector: firmware's handler address (like mtvec). Set via MMIO.
    var timerVector: UInt64 = 0
    // Network device state
    var netUrl: UInt64 = 0
    var netBuf: UInt64 = 0
    var netBufSize: UInt64 = 0
    var netResult: Int64 = 0
    /// Host-side network handler: (urlAddr, bufAddr, bufSize) -> bytes written or -1.
    /// Set by the app; this is the "hardware" behind the MMIO network device.
    var netHandler: ((UInt64, UInt64, UInt64, UInt64) -> Int64)?

    /// Host-attached NovaGPU device: the "hardware" behind the NovaBus
    /// MMIO block (see NovaBus.swift). Set by the app / test harness.
    var gpuDevice: NovaGPUDevice?

    /// Buffer for assembling multi-byte UTF-8 sequences from MMIO writes.
    private var utf8Buffer = [UInt8]()

    func mmioWrite(_ addr: UInt64, _ v: UInt64) {
        if addr == Self.consoleAddr {
            let byte = UInt8(v & 0xFF)
            utf8Buffer.append(byte)
            if isCompleteUTF8(utf8Buffer), let s = String(bytes: utf8Buffer, encoding: .utf8) {
                utf8Buffer.removeAll()
                consoleOutput?(s)
            } else if utf8Buffer.count > 8 {
                let s = String(bytes: utf8Buffer, encoding: .utf8) ?? ""
                utf8Buffer.removeAll()
                consoleOutput?(s)
            }
        } else if addr == Self.timerIntervalAddr {
            timerInterval = v
            timerCount = v  // Reset countdown
        } else if addr == Self.timerVectorAddr {
            timerVector = v
        } else if addr == Self.fileCmdAddr {
            // Execute file command synchronously
            if let handler = fileHandler {
                fileResult = handler(v, fileArg0, fileArg1, fileArg2)
            } else {
                fileResult = -1
            }
        } else if addr == Self.fileArg0Addr {
            fileArg0 = v
        } else if addr == Self.fileArg1Addr {
            fileArg1 = v
        } else if addr == Self.fileArg2Addr {
            fileArg2 = v
        } else if addr == Self.netCmdAddr {
            // Execute network command synchronously (blocks VM thread like a syscall)
            // v=1: GET (20s timeout), v=2: quick GET (5s timeout, for boot sync)
            if (v == 1 || v == 2), let handler = netHandler {
                netResult = handler(netUrl, netBuf, netBufSize, v)
            } else {
                netResult = -1
            }
        } else if addr == Self.netUrlAddr {
            netUrl = v
        } else if addr == Self.netBufAddr {
            netBuf = v
        } else if addr == Self.netBufSizeAddr {
            netBufSize = v
        } else if addr >= Self.gpuBaseAddr && addr < Self.gpuEndAddr {
            gpuDevice?.mmioWrite(offset: addr - Self.gpuBaseAddr, value: v)
        }
    }

    func mmioRead(_ addr: UInt64) -> UInt64 {
        if addr == Self.keyboardAddr {
            keyboardLock.lock()
            defer { keyboardLock.unlock() }
            return keyboardQueue.isEmpty ? 0 : UInt64(keyboardQueue.removeFirst())
        } else if addr == Self.fileResultAddr {
            return UInt64(bitPattern: fileResult)
        } else if addr == Self.timerCountAddr {
            return timerCount
        } else if addr == Self.timerIntervalAddr {
            return timerInterval
        } else if addr == Self.timerVectorAddr {
            return timerVector
        } else if addr == Self.netResultAddr {
            return UInt64(bitPattern: netResult)
        } else if addr >= Self.gpuBaseAddr && addr < Self.gpuEndAddr {
            return gpuDevice?.mmioRead(offset: addr - Self.gpuBaseAddr) ?? 0
        }
        return 0
    }

    /// Host pushes a key for the guest to read
    func pushKey(_ byte: UInt8) {
        keyboardLock.lock()
        keyboardQueue.append(byte)
        keyboardLock.unlock()
    }

    /// Clears pending keyboard input (used on reboot).
    func clearKeyboardQueue() {
        keyboardLock.lock()
        keyboardQueue.removeAll()
        keyboardLock.unlock()
    }

    /// Returns true if bytes form complete UTF-8 sequences (not cut mid-character).
    private func isCompleteUTF8(_ bytes: [UInt8]) -> Bool {
        var i = 0
        while i < bytes.count {
            let b = bytes[i]
            let len: Int
            if b < 0x80 { len = 1 }
            else if b < 0xC2 { return false } // stray continuation
            else if b < 0xE0 { len = 2 }
            else if b < 0xF0 { len = 3 }
            else if b < 0xF5 { len = 4 }
            else { return false }
            if i + len > bytes.count { return false } // incomplete
            // Validate continuation bytes
            for j in 1..<len {
                if bytes[i+j] < 0x80 || bytes[i+j] > 0xBF { return false }
            }
            i += len
        }
        return true
    }
}
