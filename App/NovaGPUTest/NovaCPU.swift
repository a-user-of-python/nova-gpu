// NovaCPU.swift — NOVA stack-machine CPU emulator
// Optimized for Apple Silicon (M3+) withContiguousArray storage,
// inlinable hot paths, and minimal ARC traffic in the fetch loop.
//
// NOTE: NovaTrap is defined in NovaMemory.swift (shared).

import Foundation

// ============================================================================
// HARNESS VARIANT "hotloop" — tools/harness/variants/hotloop/NovaCPU.swift
// Built on the V1 core: everything above the hot-loop MARK (including the
// throwing step()) is verbatim vm-app/NovaVM/VM/NovaCPU.swift. This file is
// NOT the shipping core; the coordinator promotes only measured,
// gate-passing winners. Design: agent_notes/v070-agg-a.md.
// ============================================================================

// MARK: - Privilege
enum NovaPrivilege: UInt64 {
    case user = 0, supervisor = 1, hypervisor = 2, machine = 3
}

// MARK: - CPU State
final class NovaCPU {

    // Registers
    var dsp: UInt64 = 0       // Data stack pointer (byte address, grows down)
    var csp: UInt64 = 0       // Call stack pointer
    var esp: UInt64 = 0       // Exception stack pointer
    var vsp: UInt64 = 0       // Vector stack pointer
    var pc: UInt64 = 0        // Program counter
    var fp: UInt64 = 0        // Frame pointer
    var status: UInt64 = 0    // STATUS register
    var cause: UInt64 = 0
    var epc: UInt64 = 0
    var tval: UInt64 = 0
    var satp: UInt64 = 0
    var savedStatusForMret: UInt64 = 0  // Saved status for MRET (timer interrupt)
    var dslim: UInt64 = 0
    var cslim: UInt64 = 0
    var eslim: UInt64 = 0
    var vslim: UInt64 = 0
    var cycle: UInt64 = 0

    // NZCV flags (decoded from status for speed)
    var flagN = false, flagZ = false, flagC = false, flagV = false

    // Memory backing (owned by NovaMemory)
    var memory: NovaMemory!

    // Execution control
    var running = false
    var trapHandler: ((NovaTrap, UInt64) -> Void)?

    // [JIT J1 — ADDITION 1] Engine indicator surface (design §2.3).
    // requestedEngine is set by the host (harness --engine flag; the
    // app in lane J5). activeEngine is rewritten by
    // selectEngineForRun at every run() entry with what that run
    // actually used — this is the value an indicator shows.
    // jitStats is non-nil only for runs where .jit was requested.
    var requestedEngine: NovaEnginePreference = .interpreter
    private(set) var activeEngine: NovaActiveEngine = .interpreterHotloop
    private(set) var jitStats: JitStats? = nil
    // [JIT J4 — ADDITION 5] D3 console callout switch (design §8
    // D3): true = compiled STORE8/STORE64 to the console address
    // call NovaMemory.mmioWrite directly; false = the J3 peek-exit
    // for every MMIO store. Only the JIT compiler reads this.
    var jitD3Enabled: Bool = true

    // General-purpose registers R0–R15 (programmer-visible)
    var regs: [UInt64] = Array(repeating: 0, count: 16)

    // Scratch
    private var t0: UInt64 = 0
    private var t1: UInt64 = 0

    // MARK: - Stack Primitives (inlinable hot paths)

    @inline(__always) private func checkPush(_ newSP: UInt64, limit: UInt64) throws {
        if newSP < limit { throw NovaTrap.stackOverflow }
    }

    @inline(__always) func dpush(_ v: UInt64) throws {
        let nsp = dsp &- 8
        if nsp < dslim { throw NovaTrap.stackOverflow }
        try memory.write64(nsp, v)
        dsp = nsp
    }

    @inline(__always) func dpop() throws -> UInt64 {
        // Underflow check: DSP must be below stack base.
        // Stack base tracked by memory object.
        if dsp >= memory.dataStackBase { throw NovaTrap.stackUnderflow }
        let v = try memory.read64(dsp)
        dsp = dsp &+ 8
        return v
    }

    @inline(__always) func cpush(_ v: UInt64) throws {
        let nsp = csp &- 8
        if nsp < cslim { throw NovaTrap.stackOverflow }
        try memory.write64(nsp, v)
        csp = nsp
    }

    @inline(__always) func cpop() throws -> UInt64 {
        if csp >= memory.callStackBase { throw NovaTrap.stackUnderflow }
        let v = try memory.read64(csp)
        csp = csp &+ 8
        return v
    }

    // MARK: - Flag helpers

    @inline(__always) private func setFlagsSub(_ a: UInt64, _ b: UInt64) {
        // flags for (a - b)
        let r = a &- b
        flagN = (r & 0x8000_0000_0000_0000) != 0
        flagZ = r == 0
        flagC = a >= b  // NOT borrow
        flagV = ((a ^ b) & (a ^ r) & 0x8000_0000_0000_0000) != 0
    }

    // MARK: - Main step

    func step() throws {
        // v0.0.70: resolve the implicitly-unwrapped memory reference ONCE
        // per step; the timer check, opcode fetch and every operand access
        // below go through this single reference instead of re-unwrapping
        // `memory` per access. The memory object cannot be swapped
        // mid-instruction (run() and the MMIO handlers all operate on this
        // same instance), so this changes nothing observable — faults,
        // trap vectors/tval, pc and cycle all behave exactly as before.
        let mem: NovaMemory = memory
        // Hardware timer check (for firmware scheduler).
        // MUST be before opcode fetch: when the timer fires, epc = pc must
        // point AT the interrupted instruction (not past it), otherwise MRET
        // resumes past a fetched-but-unexecuted opcode, corrupting the stack.
        // The firmware's trap handler (at the timer vector) is responsible for
        // context switching. This is hardware behavior, not app control.
        // Only fire if interrupts are enabled (bit 0 of status).
        if mem.timerInterval > 0 && (status & 1) != 0 {
            if mem.timerCount > 0 {
                mem.timerCount &-= 1
            }
            if mem.timerCount == 0 {
                // Timer fired. Reset for periodic mode.
                mem.timerCount = mem.timerInterval
                throw NovaTrap.timerInterrupt
            }
        }

        // Opcode fetch uses NovaMemory's fast path (read8 = one range
        // comparison + one byte load). It is deliberately NOT fused with
        // the operand fetches below into one wider checked read: in the
        // original code a faulting operand still leaves pc/cycle advanced
        // past the opcode byte, and trap observers can see that state —
        // fusing would change fault-visible pc/cycle. Each operand fetch
        // instead gets the same single-comparison fast path via
        // read16/read32/read64.
        let opcode = try mem.read8(pc)
        pc &+= 1
        cycle &+= 1

        switch opcode {
        // ---- Stack manipulation ----
        case 0x00: break // NOP
        case 0x01: // PUSH8
            let imm = Int64(Int8(bitPattern: try mem.read8(pc))); pc &+= 1
            try dpush(UInt64(bitPattern: imm))
        case 0x02: // PUSH16
            let imm = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            try dpush(UInt64(bitPattern: imm))
        case 0x03: // PUSH32
            let imm = Int64(Int32(bitPattern: try mem.read32(pc))); pc &+= 4
            try dpush(UInt64(bitPattern: imm))
        case 0x04: // PUSH64
            let imm = try mem.read64(pc); pc &+= 8
            try dpush(imm)
        case 0x05: // POP
            _ = try dpop()
        case 0x06: // POPN
            let n = try mem.read8(pc); pc &+= 1
            for _ in 0..<n { _ = try dpop() }
        case 0x07: // DUP
            let a = try dpop(); try dpush(a); try dpush(a)
        case 0x08: // DUP2
            let a = try dpop(); let b = try dpop()
            try dpush(b); try dpush(a); try dpush(b); try dpush(a)
        case 0x09: // SWAP
            let a = try dpop(); let b = try dpop()
            try dpush(a); try dpush(b)
        case 0x0A: // ROT: a b c → b c a (a=top)
            let a = try dpop(); let b = try dpop(); let c = try dpop()
            try dpush(b); try dpush(a); try dpush(c)
        case 0x0B: // ROTR: a b c → c a b
            let a = try dpop(); let b = try dpop(); let c = try dpop()
            try dpush(a); try dpush(c); try dpush(b)
        case 0x0C: // OVER: a b → a b a
            let a = try dpop(); let b = try dpop()
            try dpush(b); try dpush(a); try dpush(b)
        case 0x0D: // TUCK: a b → b a b
            let a = try dpop(); let b = try dpop()
            try dpush(a); try dpush(b); try dpush(a)
        case 0x0E: // PICK n
            let n = try mem.read8(pc); pc &+= 1
            let addr = dsp &+ UInt64(n) &* 8
            try dpush(try mem.read64(addr))
        case 0x0F: // ROLL n — rotate top n items (top goes to bottom of the n)
            let n = Int(try mem.read8(pc)); pc &+= 1
            if n > 1 {
                var tmp = [UInt64]()
                for _ in 0..<n { tmp.append(try dpop()) }
                // tmp[0] was top; move it to bottom: push tmp[0] first, then rest reversed
                let top = tmp[0]
                for i in (1..<n).reversed() { try dpush(tmp[i]) }
                try dpush(top)
            }
        case 0x10: // DEPTH
            let depth = (mem.dataStackBase &- dsp) / 8
            try dpush(depth)
        case 0x11: // DSPGET
            try dpush(dsp)
        case 0x12: // DSPSET (privileged: firmware use only)
            dsp = try dpop()
        case 0x13: // FPGET
            try dpush(fp)
        case 0x14: // ENTER n
            let n = try mem.read8(pc); pc &+= 1
            try cpush(fp)
            fp = dsp
            for _ in 0..<n { try dpush(0) }
        case 0x15: // LEAVE
            dsp = fp
            fp = try cpop()
        case 0x16: // LOADL n — local n lives at FP-8*(n+1)
            let n = try mem.read8(pc); pc &+= 1
            try dpush(try mem.read64(fp &- (UInt64(n) &+ 1) &* 8))
        case 0x17: // STORL n
            let n = try mem.read8(pc); pc &+= 1
            let v = try dpop()
            try mem.write64(fp &- (UInt64(n) &+ 1) &* 8, v)
        case 0x18: // LOADARG n — arg n above frame: FP+8*n (0 = last pushed)
            let n = try mem.read8(pc); pc &+= 1
            try dpush(try mem.read64(fp &+ UInt64(n) &* 8))
        case 0x19: // PUSHR n — push R[n]
            let n = try mem.read8(pc); pc &+= 1
            guard n < 16 else { tval = UInt64(n); throw NovaTrap.illegalInstruction }
            try dpush(regs[Int(n)])
        case 0x1A: // POPR n — pop into R[n]
            let n = try mem.read8(pc); pc &+= 1
            guard n < 16 else { tval = UInt64(n); throw NovaTrap.illegalInstruction }
            regs[Int(n)] = try dpop()
        case 0x1B: // PUSHALL — push R0..R15 (R0 first, R15 on top)
            for i in 0..<16 { try dpush(regs[i]) }
        case 0x1C: // POPALL — pop into R15..R0 (top of stack → R15)
            for i in (0..<16).reversed() { regs[i] = try dpop() }
        case 0x1D: // CSPGET — push call stack pointer
            try dpush(csp)
        case 0x1E: // CSPSET — pop into call stack pointer (privileged)
            csp = try dpop()

        // ---- Integer ALU ----
        case 0x40: // ADD
            let a = try dpop(); let b = try dpop()
            try dpush(b &+ a)
        case 0x41: // SUB
            let a = try dpop(); let b = try dpop()
            try dpush(b &- a)
        case 0x42: // MUL
            let a = try dpop(); let b = try dpop()
            try dpush(b &* a)
        case 0x43: // MULH (signed high)
            let a = Int64(bitPattern: try dpop())
            let b = Int64(bitPattern: try dpop())
            let (hi, _) = b.multipliedFullWidth(by: a)
            try dpush(UInt64(bitPattern: hi))
        case 0x44: // DIV (signed)
            let a = Int64(bitPattern: try dpop())
            let b = Int64(bitPattern: try dpop())
            guard a != 0 else { throw NovaTrap.illegalInstruction }
            try dpush(UInt64(bitPattern: b / a))
        case 0x45: // DIVU
            let a = try dpop(); let b = try dpop()
            guard a != 0 else { throw NovaTrap.illegalInstruction }
            try dpush(b / a)
        case 0x46: // REM (signed)
            let a = Int64(bitPattern: try dpop())
            let b = Int64(bitPattern: try dpop())
            guard a != 0 else { throw NovaTrap.illegalInstruction }
            try dpush(UInt64(bitPattern: b % a))
        case 0x47: // REMU
            let a = try dpop(); let b = try dpop()
            guard a != 0 else { throw NovaTrap.illegalInstruction }
            try dpush(b % a)
        case 0x48: // AND
            let a = try dpop(); let b = try dpop()
            try dpush(b & a)
        case 0x49: // OR
            let a = try dpop(); let b = try dpop()
            try dpush(b | a)
        case 0x4A: // XOR
            let a = try dpop(); let b = try dpop()
            try dpush(b ^ a)
        case 0x4B: // NOT
            try dpush(~(try dpop()))
        case 0x4C: // NEG
            let a = try dpop()
            try dpush(0 &- a)
        case 0x4D: // SHL
            let a = try dpop(); let b = try dpop()
            try dpush(b &<< (a & 63))
        case 0x4E: // SHR (logical)
            let a = try dpop(); let b = try dpop()
            try dpush(b &>> (a & 63))
        case 0x4F: // SAR (arithmetic)
            let a = try dpop()
            let b = Int64(bitPattern: try dpop())
            try dpush(UInt64(bitPattern: b >> (a & 63)))
        case 0x50: // INC
            try dpush((try dpop()) &+ 1)
        case 0x51: // DEC
            try dpush((try dpop()) &- 1)
        case 0x52: // CMP
            let a = try dpop(); let b = try dpop()
            setFlagsSub(b, a)
        case 0x53: // TST
            let a = try dpop()
            setFlagsSub(a, 0)
            try dpush(a)
        case 0x54: // EQ
            let a = try dpop(); let b = try dpop()
            try dpush(b == a ? 1 : 0)
        case 0x55: // NE
            let a = try dpop(); let b = try dpop()
            try dpush(b != a ? 1 : 0)
        case 0x56: // LT (signed)
            let a = Int64(bitPattern: try dpop())
            let b = Int64(bitPattern: try dpop())
            try dpush(b < a ? 1 : 0)
        case 0x57: // LTU
            let a = try dpop(); let b = try dpop()
            try dpush(b < a ? 1 : 0)
        case 0x5C: // POPCNT
            try dpush(UInt64((try dpop()).nonzeroBitCount))
        case 0x5D: // REV8
            try dpush((try dpop()).byteSwapped)
        case 0x5E: // EXT8 — sign-extend low 8 bits
            try dpush(UInt64(bitPattern: Int64(Int8(bitPattern: UInt8(try dpop() & 0xFF)))))
        case 0x5F: // EXT16
            try dpush(UInt64(bitPattern: Int64(Int16(bitPattern: UInt16(try dpop() & 0xFFFF)))))
        case 0x60: // EXT32
            try dpush(UInt64(bitPattern: Int64(Int32(bitPattern: UInt32(try dpop() & 0xFFFFFFFF)))))

        // ---- Memory ----
        case 0x80: // LOAD8
            let addr = try dpop()
            try dpush(UInt64(bitPattern: Int64(Int8(bitPattern: try mem.read8(addr)))))
        case 0x83: // LOAD64
            let addr = try dpop()
            if addr >= 0xFFFF_FFFF_FFFF_0000 {
                try dpush(mem.mmioRead(addr))
            } else {
                try dpush(try mem.read64(addr))
            }
        case 0x84: // LOADU8
            let addr84 = try dpop()
            if addr84 >= 0xFFFF_FFFF_FFFF_0000 {
                try dpush(mem.mmioRead(addr84) & 0xFF)
            } else {
                try dpush(UInt64(try mem.read8(addr84)))
            }
        case 0x85: // LOADU16
            let addr85 = try dpop()
            try dpush(UInt64(try mem.read16(addr85)))
        case 0x86: // LOADU32
            let addr86 = try dpop()
            try dpush(UInt64(try mem.read32(addr86)))
        case 0x87: // STORE8
            let addr87 = try dpop(); let v87 = try dpop()
            if addr87 >= 0xFFFF_FFFF_FFFF_0000 {
                mem.mmioWrite(addr87, v87 & 0xFF)
            } else {
                try mem.write8(addr87, UInt8(v87 & 0xFF))
            }
        case 0x88: // STORE16
            let addr88 = try dpop(); let v88 = try dpop()
            try mem.write16(addr88, UInt16(v88 & 0xFFFF))
        case 0x89: // STORE32
            let addr89 = try dpop(); let v89 = try dpop()
            try mem.write32(addr89, UInt32(v89 & 0xFFFFFFFF))
        case 0x8A: // STORE64
            let addr = try dpop(); let v = try dpop()
            if addr >= 0xFFFF_FFFF_FFFF_0000 {
                mem.mmioWrite(addr, v)
            } else {
                try mem.write64(addr, v)
            }
        case 0x8F: // MEMCPY: n dst src →  (src top)
            let src = try dpop(); let dst = try dpop(); let n = try dpop()
            try mem.copyBytes(from: src, to: dst, count: n)

        // ---- Control flow ----
        case 0xA0: // JMP off16
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            pc = UInt64(Int64(bitPattern: pc) &+ off)
        case 0xA1: // JMP32
            let off = Int64(Int32(bitPattern: try mem.read32(pc))); pc &+= 4
            pc = UInt64(Int64(bitPattern: pc) &+ off)
        case 0xA2: // BZ
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            if (try dpop()) == 0 { pc = UInt64(Int64(bitPattern: pc) &+ off) }
        case 0xA3: // BNZ
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            if (try dpop()) != 0 { pc = UInt64(Int64(bitPattern: pc) &+ off) }
        case 0xA6: // BEQ (uses Z flag)
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            if flagZ { pc = UInt64(Int64(bitPattern: pc) &+ off) }
        case 0xA7: // BNE
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            if !flagZ { pc = UInt64(Int64(bitPattern: pc) &+ off) }
        case 0xA8: // CALL off16
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            try cpush(pc)
            pc = UInt64(Int64(bitPattern: pc) &+ off)
        case 0xA9: // CALL32 off32
            let off32 = Int64(Int32(bitPattern: try mem.read32(pc))); pc &+= 4
            try cpush(pc)
            pc = UInt64(Int64(bitPattern: pc) &+ off32)
        case 0xAB: // RET
            pc = try cpop()
        case 0xAC: // JMPI
            pc = try dpop()
        case 0xAA: // CALLI
            let addr = try dpop()
            try cpush(pc)
            pc = addr
        case 0xAD: // LOOP off16 — decrement top; if != 0 branch (keeps counter)
            let off = Int64(Int16(bitPattern: try mem.read16(pc))); pc &+= 2
            let n = try dpop() &- 1
            try dpush(n)
            if n != 0 { pc = UInt64(Int64(bitPattern: pc) &+ off) }
        case 0xAE: // TRAP code
            let code = try mem.read8(pc); pc &+= 1
            tval = UInt64(code)
            throw NovaTrap.breakpoint // generic software trap

        // ---- System ----
        case 0xC0: // ECALL
            throw NovaTrap.ecallU
        case 0xC1: // EBREAK
            throw NovaTrap.breakpoint
        case 0xC2: // MRET — return from Machine trap
            // Restore PC from EPC, restore status (re-enables interrupts)
            pc = epc
            status = savedStatusForMret
        case 0xC6: // CSRGET — push CSR value (0=epc, 1=cause, 2=status)
            let cget = try mem.read8(pc); pc &+= 1
            switch cget {
            case 0: try dpush(epc)
            case 1: try dpush(cause)
            case 2: try dpush(status)
            default: throw NovaTrap.illegalInstruction
            }
        case 0xC7: // CSRSET — pop value into CSR (0=epc, 1=cause, 2=status)
            let cset = try mem.read8(pc); pc &+= 1
            let cv = try dpop()
            switch cset {
            case 0: epc = cv
            case 1: cause = cv
            case 2: status = cv
            default: throw NovaTrap.illegalInstruction
            }
        case 0xC5: // WFI — yield in VM
            running = false
        case 0xCB: // RDTIME
            try dpush(UInt64(Date().timeIntervalSince1970 * 1_000_000))
        case 0xCC: // RDCYCLE
            try dpush(cycle)

        // ---- Crypto ----
        case 0xF4: // RDRAND
            try dpush(UInt64.random(in: 0...UInt64.max))

        case 0xF5: // CPUID
            let leaf = try dpop()
            switch leaf {
            case 0: try dpush(0x4E564F41) // "NOVA" vendor
            case 1: try dpush(0x0200)     // version 2.0
            case 2: try dpush(0b0111)     // FP|SIMD|crypto
            case 3: try dpush(0x00312D61766F4E) // "Nova-1" model name (packed ASCII)
            default: try dpush(0)
            }

        default:
            tval = UInt64(opcode)
            throw NovaTrap.illegalInstruction
        }
    }

    // MARK: - Hot loop (variant "hotloop")
    //
    // run() executes through hotRun() below: a NON-THROWING loop that
    // holds all hot CPU state in locals for the whole run and returns
    // traps as values. The public throwing step() above is UNCHANGED
    // (verbatim V1) and remains the implementation for:
    //   * any external caller of step(), and
    //   * the one "bail" opcode (ROLL, 0x0F — see hotRun) whose
    //     heap-allocating, multi-pop/push body is not mirrored here;
    //     run() executes it via one real step() call, state already
    //     synced, so its results are identical by construction.
    //
    // Trap-state contract (the crux): on EVERY exit from hotRun the
    // object state is written back in full (syncOut), and for every
    // mirrored instruction the state at the moment of a trap is
    // byte-identical to what V1's throwing step() leaves:
    //   * timer trap: fires BEFORE fetch — pc/cycle untouched for the
    //     interrupted instruction, timerCount already reset to the
    //     interval (mirrors NovaCPU.swift step() lines 118-127);
    //   * fetch fault: pc/cycle NOT advanced (read8 throws first);
    //   * operand fault: pc still points AT the operand (the original
    //     `pc &+= n` runs only after a successful read), cycle already
    //     incremented, dsp/csp already reflect any completed pops;
    //   * stack overflow on push: the value is NOT written and the
    //     stack pointer is NOT moved (dpush checks before writing and
    //     assigns dsp only on success) — hpush mirrors that order;
    //   * underflow/load fault on pop: the stack pointer is NOT moved
    //     (dpop assigns dsp only after a successful read) — hpop
    //     mirrors that order;
    //   * div-by-zero / illegal CSR / bad register index: all pops and
    //     operand advances that the original performs before the
    //     guard have already happened; tval matches (set only where
    //     the original sets it).
    // Each helper below is a line-level mirror of the V1 primitive it
    // replaces, with the generic checked accessor replaced by a raw
    // buffer access guarded by EXACTLY the accessor's own comparison
    // (read64/write64 succeed iff addr <= totalSize-8, read8 iff
    // addr < totalSize, etc.), so a mirrored access succeeds iff the
    // generic one would, and transfers the same little-endian bytes.
    //
    // Timer state is cached in locals (ltInterval/ltCount) with two
    // synchronisation points that keep MMIO-observed behaviour exact:
    // ltCount is flushed to memory before every mmioRead (the guest
    // can read the live countdown at timerCountAddr), and both values
    // are reloaded after every mmioWrite (a write to
    // timerIntervalAddr mutates both fields inside NovaMemory).
    // Host handlers behind file/net MMIO never touch the timer fields.
    //
    // Object state is additionally synced every 8192 instructions so
    // the app engine's 4 Hz register snapshot and its stop()
    // (cpu.running = false from another thread) observe live state
    // with bounded lag, as they do with the per-step loop.

    private enum HotOutcome {
        case budgetOrHalt            // budget exhausted, WFI, or external stop
        case trap(NovaTrap)          // one instruction trapped; steps include it
        case bail                    // next instruction is ROLL; 0 steps used for it
    }

    /// The trap handling of V1 run()'s catch block, factored out
    /// verbatim (comments included) so the hot loop's trap exits and
    /// the bail path's real step() share it exactly. Precondition:
    /// object state is fully synced (hotRun syncs on exit; step()
    /// maintains object state itself).
    private func handleTrapAfterStep(_ trap: NovaTrap) {
        // Hardware timer interrupt vectoring (for firmware scheduler).
        // If firmware has set timerVector, vector to it instead of halting.
        // This is hardware behavior, not app control.
        if (trap == .timerInterrupt || trap == .ecallU) && memory.timerVector != 0 {
            // Save trap state
            epc = pc
            cause = (trap == .ecallU) ? 8 : 0x8000000000000007
            // Push epc and cause onto data stack for firmware handler
            // (firmware has no CSRGET, so we pass via stack)
            try? dpush(epc)
            try? dpush(cause)
            // Save status and disable interrupts (clear bit 0)
            // Firmware MRET will restore.
            // Note: we save status in a CPU field, firmware doesn't need it
            // because MRET restores from the saved value.
            let savedStatus = status
            status &= ~UInt64(1)  // Disable interrupts
            // Store saved status where MRET can find it
            // (use a dedicated field)
            savedStatusForMret = savedStatus
            // Jump to firmware handler
            pc = memory.timerVector
            // Keep running — do NOT call trapHandler or halt
        } else {
            trapHandler?(trap, tval)
            // Default: halt on unhandled trap
            if trapHandler == nil { running = false }
        }
    }

    /// Non-throwing interpreter loop. Runs up to `budget` instructions
    /// starting from the current object state; on return the object
    /// state is fully synced. Returns (stepsUsed, outcome).
    private func hotRun(budget: UInt64) -> (UInt64, HotOutcome) {
        let mem: NovaMemory = memory
        return mem.withHotBuffer { (buf: UnsafeMutablePointer<UInt8>) -> (UInt64, HotOutcome) in
            // ---- Load object state into locals ----
            var lpc = pc
            var lcyc = cycle
            var ldsp = dsp
            var lcsp = csp
            var lfp = fp
            var lstatus = status
            var ltval = tval
            var lepc = epc
            var lcause = cause
            var lsaved = savedStatusForMret
            var lfN = flagN
            var lfZ = flagZ
            var lfC = flagC
            var lfV = flagV
            var lregs = regs
            let ldslim = dslim
            let lcslim = cslim
            let dataBase = mem.dataStackBase
            let callBase = mem.callStackBase
            let ramSize = UInt64(mem.totalSize)
            let max16 = ramSize - 2
            let max32 = ramSize - 4
            let max64 = ramSize - 8
            var ltInterval = mem.timerInterval
            var ltCount = mem.timerCount
            var used: UInt64 = 0
            var pendingTrap: NovaTrap? = nil
            var bailRequested = false
            var halted = false
            var sinceSync: UInt64 = 0

            // Write every local back to the object. Called on every
            // exit and periodically during long runs (see header).
            // timerInterval is NOT written back: NovaMemory's copy is
            // authoritative (mmioWrite mutates it; ltInterval is a
            // cache reloaded after every mmioWrite). timerCount IS
            // written back: during the loop the local is authoritative.
            func syncOut() {
                self.pc = lpc
                self.cycle = lcyc
                self.dsp = ldsp
                self.csp = lcsp
                self.fp = lfp
                self.status = lstatus
                self.tval = ltval
                self.epc = lepc
                self.cause = lcause
                self.savedStatusForMret = lsaved
                self.flagN = lfN
                self.flagZ = lfZ
                self.flagC = lfC
                self.flagV = lfV
                self.regs = lregs
                mem.timerCount = ltCount
            }

            // ---- Stack helpers: exact mirrors of dpush/dpop/cpush/cpop.
            // On trap they set pendingTrap and change NO state, exactly
            // like the throwing originals (pointer assigned only after
            // the memory access succeeds). The raw store/load is
            // guarded by the generic accessor's own range comparison
            // (addr <= ramSize-8), so it succeeds iff write64/read64
            // would — this is the data-stack fast path: dsp lives in a
            // local and the slot is reached directly in the buffer.
            func hpush(_ v: UInt64) -> Bool {
                let nsp = ldsp &- 8
                if nsp < ldslim { pendingTrap = .stackOverflow; return false }
                if nsp > max64 { pendingTrap = .storeAccessFault; return false }
                var tmp = v
                memcpy(buf + Int(nsp), &tmp, 8)
                ldsp = nsp
                return true
            }
            func hpop() -> UInt64? {
                if ldsp >= dataBase { pendingTrap = .stackUnderflow; return nil }
                if ldsp > max64 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt64 = 0
                memcpy(&v, buf + Int(ldsp), 8)
                ldsp = ldsp &+ 8
                return v
            }
            func hcpush(_ v: UInt64) -> Bool {
                let nsp = lcsp &- 8
                if nsp < lcslim { pendingTrap = .stackOverflow; return false }
                if nsp > max64 { pendingTrap = .storeAccessFault; return false }
                var tmp = v
                memcpy(buf + Int(nsp), &tmp, 8)
                lcsp = nsp
                return true
            }
            func hcpop() -> UInt64? {
                if lcsp >= callBase { pendingTrap = .stackUnderflow; return nil }
                if lcsp > max64 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt64 = 0
                memcpy(&v, buf + Int(lcsp), 8)
                lcsp = lcsp &+ 8
                return v
            }

            // ---- Operand fetches at lpc. These NEVER advance lpc:
            // callers advance explicitly after success, mirroring the
            // original `let x = try mem.readN(pc); pc &+= n` order, so
            // a fault leaves lpc pointing at the operand, as V1 does.
            func frd8() -> UInt8? {
                if lpc >= ramSize { pendingTrap = .loadAccessFault; return nil }
                return buf[Int(lpc)]
            }
            func frd16() -> UInt16? {
                if lpc > max16 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt16 = 0
                memcpy(&v, buf + Int(lpc), 2)
                return v
            }
            func frd32() -> UInt32? {
                if lpc > max32 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt32 = 0
                memcpy(&v, buf + Int(lpc), 4)
                return v
            }
            func frd64() -> UInt64? {
                if lpc > max64 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt64 = 0
                memcpy(&v, buf + Int(lpc), 8)
                return v
            }

            // ---- General loads/stores (RAM addresses only; callers
            // route MMIO addresses to mmioRead/mmioWrite first, exactly
            // where V1's switch routes them). Guards are the accessors'
            // own comparisons; failure kind matches the accessor.
            func hload8(_ addr: UInt64) -> UInt8? {
                if addr >= ramSize { pendingTrap = .loadAccessFault; return nil }
                return buf[Int(addr)]
            }
            func hload16(_ addr: UInt64) -> UInt16? {
                if addr > max16 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt16 = 0
                memcpy(&v, buf + Int(addr), 2)
                return v
            }
            func hload32(_ addr: UInt64) -> UInt32? {
                if addr > max32 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt32 = 0
                memcpy(&v, buf + Int(addr), 4)
                return v
            }
            func hload64(_ addr: UInt64) -> UInt64? {
                if addr > max64 { pendingTrap = .loadAccessFault; return nil }
                var v: UInt64 = 0
                memcpy(&v, buf + Int(addr), 8)
                return v
            }
            func hstore8(_ addr: UInt64, _ v: UInt8) -> Bool {
                if addr >= ramSize { pendingTrap = .storeAccessFault; return false }
                buf[Int(addr)] = v
                return true
            }
            func hstore16(_ addr: UInt64, _ v: UInt16) -> Bool {
                if addr > max16 { pendingTrap = .storeAccessFault; return false }
                var tmp = v
                memcpy(buf + Int(addr), &tmp, 2)
                return true
            }
            func hstore32(_ addr: UInt64, _ v: UInt32) -> Bool {
                if addr > max32 { pendingTrap = .storeAccessFault; return false }
                var tmp = v
                memcpy(buf + Int(addr), &tmp, 4)
                return true
            }
            func hstore64(_ addr: UInt64, _ v: UInt64) -> Bool {
                if addr > max64 { pendingTrap = .storeAccessFault; return false }
                var tmp = v
                memcpy(buf + Int(addr), &tmp, 8)
                return true
            }

            // Mirror of setFlagsSub, writing the local flag copies.
            func setFlagsSubH(_ a: UInt64, _ b: UInt64) {
                let r = a &- b
                lfN = (r & 0x8000_0000_0000_0000) != 0
                lfZ = r == 0
                lfC = a >= b  // NOT borrow
                lfV = ((a ^ b) & (a ^ r) & 0x8000_0000_0000_0000) != 0
            }

            if !self.running {
                syncOut()
                return (0, .budgetOrHalt)
            }

            // <<DISPATCH-TABLE-SLOT>>

            mainLoop: while used < budget {
                // Periodic sync: bounds the lag the engine's live
                // register snapshot and stop() can observe, and is the
                // external-stop check (V1 checks running per step; the
                // hot loop checks it at least every 8192 instructions).
                if sinceSync >= 8192 {
                    sinceSync = 0
                    syncOut()
                    if !self.running { break mainLoop }
                }
                // Bail peek: classify the next opcode WITHOUT any
                // timer/fetch side effects. If it is ROLL, exit before
                // the timer check so run()'s real step() re-does the
                // timer check + fetch + execute exactly as V1 would
                // (no double countdown, no skipped timer trap). A
                // fetch-fault pc is not a bail: fall through to the
                // normal path, which faults after the timer check,
                // in V1's order.
                if lpc < ramSize && buf[Int(lpc)] == 0x0F {
                    bailRequested = true
                    break mainLoop
                }
                // Hardware timer check — mirrors step() lines 118-127
                // exactly: before fetch, only when interrupts are
                // enabled; on firing, count is reset BEFORE the trap
                // and pc/cycle are untouched for this instruction.
                if ltInterval > 0 && (lstatus & 1) != 0 {
                    if ltCount > 0 {
                        ltCount &-= 1
                    }
                    if ltCount == 0 {
                        ltCount = ltInterval
                        used &+= 1
                        sinceSync &+= 1
                        pendingTrap = .timerInterrupt
                        break mainLoop
                    }
                }
                // Opcode fetch (NOT fused with operand fetches, for
                // the same fault-visible pc/cycle reason as V1).
                if lpc >= ramSize {
                    used &+= 1
                    pendingTrap = .loadAccessFault
                    break mainLoop
                }
                let opcode = buf[Int(lpc)]
                lpc &+= 1
                lcyc &+= 1
                used &+= 1
                sinceSync &+= 1

                // <<DISPATCH-SWITCH-BEGIN>>
                switch opcode {
                // ---- Stack manipulation ----
                case 0x00: break // NOP
                case 0x01: // PUSH8
                    guard let raw = frd8() else { break }
                    lpc &+= 1
                    _ = hpush(UInt64(bitPattern: Int64(Int8(bitPattern: raw))))
                case 0x02: // PUSH16
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    _ = hpush(UInt64(bitPattern: Int64(Int16(bitPattern: raw))))
                case 0x03: // PUSH32
                    guard let raw = frd32() else { break }
                    lpc &+= 4
                    _ = hpush(UInt64(bitPattern: Int64(Int32(bitPattern: raw))))
                case 0x04: // PUSH64
                    guard let imm = frd64() else { break }
                    lpc &+= 8
                    _ = hpush(imm)
                case 0x05: // POP
                    _ = hpop()
                case 0x06: // POPN
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    for _ in 0..<n { if hpop() == nil { break } }
                case 0x07: // DUP
                    guard let a = hpop(), hpush(a) else { break }
                    _ = hpush(a)
                case 0x08: // DUP2
                    guard let a = hpop(), let b = hpop(), hpush(b), hpush(a), hpush(b) else { break }
                    _ = hpush(a)
                case 0x09: // SWAP
                    guard let a = hpop(), let b = hpop(), hpush(a) else { break }
                    _ = hpush(b)
                case 0x0A: // ROT: a b c → b c a (a=top)
                    guard let a = hpop(), let b = hpop(), let c = hpop(), hpush(b), hpush(a) else { break }
                    _ = hpush(c)
                case 0x0B: // ROTR: a b c → c a b
                    guard let a = hpop(), let b = hpop(), let c = hpop(), hpush(a), hpush(c) else { break }
                    _ = hpush(b)
                case 0x0C: // OVER: a b → a b a
                    guard let a = hpop(), let b = hpop(), hpush(b), hpush(a) else { break }
                    _ = hpush(b)
                case 0x0D: // TUCK: a b → b a b
                    guard let a = hpop(), let b = hpop(), hpush(a), hpush(b) else { break }
                    _ = hpush(a)
                case 0x0E: // PICK n
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    let addr = ldsp &+ UInt64(n) &* 8
                    guard let v = hload64(addr) else { break }
                    _ = hpush(v)
                case 0x0F: // ROLL n — BAIL opcode: run() executes it via
                    // step(). Unreachable in practice (the peek above
                    // bails first); kept so the switch is total.
                    bailRequested = true
                case 0x10: // DEPTH
                    let depth = (dataBase &- ldsp) / 8
                    _ = hpush(depth)
                case 0x11: // DSPGET
                    _ = hpush(ldsp)
                case 0x12: // DSPSET (privileged: firmware use only)
                    if let v = hpop() { ldsp = v }
                case 0x13: // FPGET
                    _ = hpush(lfp)
                case 0x14: // ENTER n
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    guard hcpush(lfp) else { break }
                    lfp = ldsp
                    for _ in 0..<n { if !hpush(0) { break } }
                case 0x15: // LEAVE
                    ldsp = lfp
                    if let v = hcpop() { lfp = v }
                case 0x16: // LOADL n — local n lives at FP-8*(n+1)
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    guard let v = hload64(lfp &- (UInt64(n) &+ 1) &* 8) else { break }
                    _ = hpush(v)
                case 0x17: // STORL n
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    guard let v = hpop() else { break }
                    _ = hstore64(lfp &- (UInt64(n) &+ 1) &* 8, v)
                case 0x18: // LOADARG n — arg n above frame: FP+8*n (0 = last pushed)
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    guard let v = hload64(lfp &+ UInt64(n) &* 8) else { break }
                    _ = hpush(v)
                case 0x19: // PUSHR n — push R[n]
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    guard n < 16 else { ltval = UInt64(n); pendingTrap = .illegalInstruction; break }
                    _ = hpush(lregs[Int(n)])
                case 0x1A: // POPR n — pop into R[n]
                    guard let n = frd8() else { break }
                    lpc &+= 1
                    guard n < 16 else { ltval = UInt64(n); pendingTrap = .illegalInstruction; break }
                    if let v = hpop() { lregs[Int(n)] = v }
                case 0x1B: // PUSHALL — push R0..R15 (R0 first, R15 on top)
                    for i in 0..<16 { if !hpush(lregs[i]) { break } }
                case 0x1C: // POPALL — pop into R15..R0 (top of stack → R15)
                    for i in (0..<16).reversed() {
                        guard let v = hpop() else { break }
                        lregs[i] = v
                    }
                case 0x1D: // CSPGET — push call stack pointer
                    _ = hpush(lcsp)
                case 0x1E: // CSPSET — pop into call stack pointer (privileged)
                    if let v = hpop() { lcsp = v }

                // ---- Integer ALU ----
                case 0x40: // ADD
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b &+ a)
                case 0x41: // SUB
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b &- a)
                case 0x42: // MUL
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b &* a)
                case 0x43: // MULH (signed high)
                    guard let ar = hpop(), let br = hpop() else { break }
                    let a = Int64(bitPattern: ar)
                    let b = Int64(bitPattern: br)
                    let (hi, _) = b.multipliedFullWidth(by: a)
                    _ = hpush(UInt64(bitPattern: hi))
                case 0x44: // DIV (signed)
                    guard let ar = hpop(), let br = hpop() else { break }
                    let a = Int64(bitPattern: ar)
                    let b = Int64(bitPattern: br)
                    guard a != 0 else { pendingTrap = .illegalInstruction; break }
                    _ = hpush(UInt64(bitPattern: b / a))
                case 0x45: // DIVU
                    guard let a = hpop(), let b = hpop() else { break }
                    guard a != 0 else { pendingTrap = .illegalInstruction; break }
                    _ = hpush(b / a)
                case 0x46: // REM (signed)
                    guard let ar = hpop(), let br = hpop() else { break }
                    let a = Int64(bitPattern: ar)
                    let b = Int64(bitPattern: br)
                    guard a != 0 else { pendingTrap = .illegalInstruction; break }
                    _ = hpush(UInt64(bitPattern: b % a))
                case 0x47: // REMU
                    guard let a = hpop(), let b = hpop() else { break }
                    guard a != 0 else { pendingTrap = .illegalInstruction; break }
                    _ = hpush(b % a)
                case 0x48: // AND
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b & a)
                case 0x49: // OR
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b | a)
                case 0x4A: // XOR
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b ^ a)
                case 0x4B: // NOT
                    if let a = hpop() { _ = hpush(~a) }
                case 0x4C: // NEG
                    if let a = hpop() { _ = hpush(0 &- a) }
                case 0x4D: // SHL
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b &<< (a & 63))
                case 0x4E: // SHR (logical)
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b &>> (a & 63))
                case 0x4F: // SAR (arithmetic)
                    guard let a = hpop(), let br = hpop() else { break }
                    let b = Int64(bitPattern: br)
                    _ = hpush(UInt64(bitPattern: b >> (a & 63)))
                case 0x50: // INC
                    if let a = hpop() { _ = hpush(a &+ 1) }
                case 0x51: // DEC
                    if let a = hpop() { _ = hpush(a &- 1) }
                case 0x52: // CMP
                    guard let a = hpop(), let b = hpop() else { break }
                    setFlagsSubH(b, a)
                case 0x53: // TST
                    if let a = hpop() {
                        setFlagsSubH(a, 0)
                        _ = hpush(a)
                    }
                case 0x54: // EQ
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b == a ? 1 : 0)
                case 0x55: // NE
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b != a ? 1 : 0)
                case 0x56: // LT (signed)
                    guard let ar = hpop(), let br = hpop() else { break }
                    let a = Int64(bitPattern: ar)
                    let b = Int64(bitPattern: br)
                    _ = hpush(b < a ? 1 : 0)
                case 0x57: // LTU
                    guard let a = hpop(), let b = hpop() else { break }
                    _ = hpush(b < a ? 1 : 0)
                case 0x5C: // POPCNT
                    if let a = hpop() { _ = hpush(UInt64(a.nonzeroBitCount)) }
                case 0x5D: // REV8
                    if let a = hpop() { _ = hpush(a.byteSwapped) }
                case 0x5E: // EXT8 — sign-extend low 8 bits
                    if let a = hpop() { _ = hpush(UInt64(bitPattern: Int64(Int8(bitPattern: UInt8(a & 0xFF))))) }
                case 0x5F: // EXT16
                    if let a = hpop() { _ = hpush(UInt64(bitPattern: Int64(Int16(bitPattern: UInt16(a & 0xFFFF))))) }
                case 0x60: // EXT32
                    if let a = hpop() { _ = hpush(UInt64(bitPattern: Int64(Int32(bitPattern: UInt32(a & 0xFFFFFFFF))))) }

                // ---- Memory ----
                case 0x80: // LOAD8 (sign-extended)
                    guard let addr = hpop() else { break }
                    guard let v = hload8(addr) else { break }
                    _ = hpush(UInt64(bitPattern: Int64(Int8(bitPattern: v))))
                case 0x83: // LOAD64
                    guard let addr = hpop() else { break }
                    if addr >= 0xFFFF_FFFF_FFFF_0000 {
                        mem.timerCount = ltCount  // flush: guest may read timerCountAddr
                        _ = hpush(mem.mmioRead(addr))
                    } else {
                        guard let v = hload64(addr) else { break }
                        _ = hpush(v)
                    }
                case 0x84: // LOADU8
                    guard let addr = hpop() else { break }
                    if addr >= 0xFFFF_FFFF_FFFF_0000 {
                        mem.timerCount = ltCount  // flush (see LOAD64)
                        _ = hpush(mem.mmioRead(addr) & 0xFF)
                    } else {
                        guard let v = hload8(addr) else { break }
                        _ = hpush(UInt64(v))
                    }
                case 0x85: // LOADU16
                    guard let addr = hpop() else { break }
                    guard let v = hload16(addr) else { break }
                    _ = hpush(UInt64(v))
                case 0x86: // LOADU32
                    guard let addr = hpop() else { break }
                    guard let v = hload32(addr) else { break }
                    _ = hpush(UInt64(v))
                case 0x87: // STORE8
                    guard let addr = hpop(), let v = hpop() else { break }
                    if addr >= 0xFFFF_FFFF_FFFF_0000 {
                        mem.mmioWrite(addr, v & 0xFF)
                        ltInterval = mem.timerInterval  // reload (see header)
                        ltCount = mem.timerCount
                    } else {
                        _ = hstore8(addr, UInt8(v & 0xFF))
                    }
                case 0x88: // STORE16
                    guard let addr = hpop(), let v = hpop() else { break }
                    _ = hstore16(addr, UInt16(v & 0xFFFF))
                case 0x89: // STORE32
                    guard let addr = hpop(), let v = hpop() else { break }
                    _ = hstore32(addr, UInt32(v & 0xFFFFFFFF))
                case 0x8A: // STORE64
                    guard let addr = hpop(), let v = hpop() else { break }
                    if addr >= 0xFFFF_FFFF_FFFF_0000 {
                        mem.mmioWrite(addr, v)
                        ltInterval = mem.timerInterval  // reload (see header)
                        ltCount = mem.timerCount
                    } else {
                        _ = hstore64(addr, v)
                    }
                case 0x8F: // MEMCPY: n dst src →  (src top)
                    guard let src = hpop(), let dst = hpop(), let n = hpop() else { break }
                    // copyBytes' guard reproduced verbatim (NovaMemory):
                    // same comparisons, same wrapping adds, same trap.
                    guard n <= ramSize,
                          src < ramSize, src &+ n <= ramSize,
                          dst < ramSize, dst &+ n <= ramSize else {
                        pendingTrap = .loadAccessFault
                        break
                    }
                    memmove(buf + Int(dst), buf + Int(src), Int(n))

                // ---- Control flow ----
                case 0xA0: // JMP off16
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    let off = Int64(Int16(bitPattern: raw))
                    lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                case 0xA1: // JMP32
                    guard let raw = frd32() else { break }
                    lpc &+= 4
                    let off = Int64(Int32(bitPattern: raw))
                    lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                case 0xA2: // BZ
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    guard let v = hpop() else { break }
                    if v == 0 {
                        let off = Int64(Int16(bitPattern: raw))
                        lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                    }
                case 0xA3: // BNZ
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    guard let v = hpop() else { break }
                    if v != 0 {
                        let off = Int64(Int16(bitPattern: raw))
                        lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                    }
                case 0xA6: // BEQ (uses Z flag)
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    if lfZ {
                        let off = Int64(Int16(bitPattern: raw))
                        lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                    }
                case 0xA7: // BNE
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    if !lfZ {
                        let off = Int64(Int16(bitPattern: raw))
                        lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                    }
                case 0xA8: // CALL off16
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    guard hcpush(lpc) else { break }
                    let off = Int64(Int16(bitPattern: raw))
                    lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                case 0xA9: // CALL32 off32
                    guard let raw = frd32() else { break }
                    lpc &+= 4
                    guard hcpush(lpc) else { break }
                    let off = Int64(Int32(bitPattern: raw))
                    lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                case 0xAB: // RET
                    if let v = hcpop() { lpc = v }
                case 0xAC: // JMPI
                    if let v = hpop() { lpc = v }
                case 0xAA: // CALLI
                    guard let addr = hpop() else { break }
                    guard hcpush(lpc) else { break }
                    lpc = addr
                case 0xAD: // LOOP off16 — decrement top; if != 0 branch (keeps counter)
                    guard let raw = frd16() else { break }
                    lpc &+= 2
                    guard let top = hpop() else { break }
                    let n = top &- 1
                    guard hpush(n) else { break }
                    if n != 0 {
                        let off = Int64(Int16(bitPattern: raw))
                        lpc = UInt64(Int64(bitPattern: lpc) &+ off)
                    }
                case 0xAE: // TRAP code
                    guard let code = frd8() else { break }
                    lpc &+= 1
                    ltval = UInt64(code)
                    pendingTrap = .breakpoint // generic software trap

                // ---- System ----
                case 0xC0: // ECALL
                    pendingTrap = .ecallU
                case 0xC1: // EBREAK
                    pendingTrap = .breakpoint
                case 0xC2: // MRET — return from Machine trap
                    lpc = lepc
                    lstatus = lsaved
                case 0xC6: // CSRGET — push CSR value (0=epc, 1=cause, 2=status)
                    guard let cget = frd8() else { break }
                    lpc &+= 1
                    switch cget {
                    case 0: _ = hpush(lepc)
                    case 1: _ = hpush(lcause)
                    case 2: _ = hpush(lstatus)
                    default: pendingTrap = .illegalInstruction
                    }
                case 0xC7: // CSRSET — pop value into CSR (0=epc, 1=cause, 2=status)
                    guard let cset = frd8() else { break }
                    lpc &+= 1
                    guard let cv = hpop() else { break }
                    switch cset {
                    case 0: lepc = cv
                    case 1: lcause = cv
                    case 2: lstatus = cv
                    default: pendingTrap = .illegalInstruction
                    }
                case 0xC5: // WFI — yield in VM
                    halted = true
                case 0xCB: // RDTIME
                    _ = hpush(UInt64(Date().timeIntervalSince1970 * 1_000_000))
                case 0xCC: // RDCYCLE
                    _ = hpush(lcyc)

                // ---- Crypto ----
                case 0xF4: // RDRAND
                    _ = hpush(UInt64.random(in: 0...UInt64.max))
                case 0xF5: // CPUID
                    guard let leaf = hpop() else { break }
                    switch leaf {
                    case 0: _ = hpush(0x4E564F41) // "NOVA" vendor
                    case 1: _ = hpush(0x0200)     // version 2.0
                    case 2: _ = hpush(0b0111)     // FP|SIMD|crypto
                    case 3: _ = hpush(0x00312D61766F4E) // "Nova-1" model name (packed ASCII)
                    default: _ = hpush(0)
                    }

                default:
                    ltval = UInt64(opcode)
                    pendingTrap = .illegalInstruction
                }
                // <<DISPATCH-SWITCH-END>>

                if pendingTrap != nil { break mainLoop }
                if halted { break mainLoop }
                if bailRequested { break mainLoop }
            }

            syncOut()
            if halted { self.running = false }
            if let t = pendingTrap { return (used, .trap(t)) }
            if bailRequested { return (used, .bail) }
            return (used, .budgetOrHalt)
        }
    }

    // MARK: - Run loop

    // [JIT J1 — ADDITION 3] Engine selection + runtime gate
    // (design §2.1/§2.2). Called once at every run() entry.
    // Returns true only if the JIT trampoline drove the whole run
    // (and has already synced object state, as hotRun does).
    //
    // J1 behaviour: when .jit is requested, the probe runs and its
    // result is recorded (NovaJIT.lastProbe), a fresh zeroed
    // JitStats is installed, and the trampoline is offered the run —
    // but it compiles zero blocks and always declines, so this
    // returns false and activeEngine truthfully stays
    // .interpreterHotloop for a run the interpreter drove. When the
    // probe fails, same result, silently. When .jit is not
    // requested, nothing but the activeEngine/jitStats resets runs.
    // [JIT J2 — ADDITION 4] Trampoline bridges. handleTrapAfterStep
    // is `private` (file-scoped) and jitStats' setter is
    // `private(set)`, so NovaJIT.swift cannot reach either directly;
    // these two internal wrappers expose exactly those operations
    // to the trampoline and nothing else. Used only by
    // NovaJIT.runTrampoline.
    func jitHandleTrapAfterStep(_ trap: NovaTrap) {
        handleTrapAfterStep(trap)
    }
    func jitSetStats(_ s: JitStats) {
        jitStats = s
    }

    private func selectEngineForRun(maxSteps: UInt64) -> Bool {
        activeEngine = .interpreterHotloop
        guard requestedEngine == .jit else {
            jitStats = nil
            return false
        }
        jitStats = JitStats()
        guard case .available(let path) = NovaJIT.probe() else {
            return false
        }
        if NovaJIT.runTrampoline(cpu: self, maxSteps: maxSteps) {
            activeEngine = .jit(path: path)
            return true
        }
        return false
    }

    func run(maxSteps: UInt64 = .max) {
        // [JIT J1 — ADDITION 2] Probe + dispatch hook (see banner).
        // In J1 this never takes the run over; execution falls
        // through to the shipped body below, unchanged.
        if selectEngineForRun(maxSteps: maxSteps) { return }
        running = true
        var steps: UInt64 = 0
        while running && steps < maxSteps {
            // Run the non-throwing hot loop for the remaining budget.
            // It returns on budget exhaustion, WFI/external stop, a
            // trap (state already synced), or a bail opcode — the
            // outcomes are disjoint and each is handled exactly as
            // V1's per-step loop handled the same event.
            let (used, outcome) = hotRun(budget: maxSteps - steps)
            steps &+= used
            switch outcome {
            case .budgetOrHalt:
                // Budget exhausted (harness sees running == true), or
                // WFI / external stop cleared running; the while
                // condition re-checks both, exactly as V1 did after
                // each step. hotRun guarantees progress (used >= 1,
                // a trap, a bail, or running == false) whenever it is
                // entered with budget >= 1 and running == true, so
                // this case cannot spin.
                break
            case .trap(let trap):
                // Identical to V1's catch block (NovaTrap is the only
                // error type step() can throw, so V1's generic catch
                // was unreachable; hotRun produces NovaTrap values
                // only, by construction).
                handleTrapAfterStep(trap)
            case .bail:
                // One real throwing step() for the bail opcode (ROLL),
                // with the same trap handling and step accounting as
                // V1 (the trapping step counts, exactly as in V1's
                // do/catch + steps increment).
                do {
                    try step()
                } catch let trap as NovaTrap {
                    handleTrapAfterStep(trap)
                } catch {
                    running = false
                }
                steps &+= 1
            }
        }
    }

}
