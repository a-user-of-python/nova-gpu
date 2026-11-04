#!/usr/bin/env python3
"""NOVA assembler — stack-machine assembly to binary.
Usage: python3 novas.py program.nova -o program.bin

Assembly syntax:
    ; comment
    label:
    PUSH8 42 / PUSH16 / PUSH32 / PUSH64 <imm>
    ADD SUB MUL DIV AND OR XOR NOT NEG ...   (zero-operand)
    JMP label / JMP32 label
    BZ label / BNZ label / BEQ label / BNE label / BLT label / BGE label
    CALL label / CALL32 label / RET
    LOAD64 / STORE64 / LOAD8 / STORE8 ...
    ENTER n / LEAVE / LOADL n / STORL n
    ECALL / EBREAK / WFI / RDTIME / RDCYCLE / RDRAND / CPUID
    FADD.D / FSUB.D / FMUL.D / FDIV.D ...
    .word <value>  — emit raw 64-bit word
"""

import sys, struct

# opcode table: mnemonic -> (opcode_byte, imm_size)
# Covers the full ISA_REFERENCE.md opcode map.
OPS = {
    'NOP': (0x00, 0),
    'PUSH8': (0x01, 1), 'PUSH16': (0x02, 2), 'PUSH32': (0x03, 4), 'PUSH64': (0x04, 8),
    'POP': (0x05, 0), 'POPN': (0x06, 1),
    'DUP': (0x07, 0), 'DUP2': (0x08, 0),
    'SWAP': (0x09, 0), 'ROT': (0x0A, 0), 'ROTR': (0x0B, 0),
    'OVER': (0x0C, 0), 'TUCK': (0x0D, 0),
    'PICK': (0x0E, 1), 'ROLL': (0x0F, 1),
    'DEPTH': (0x10, 0), 'DSPGET': (0x11, 0), 'DSPSET': (0x12, 0), 'FPGET': (0x13, 0),
    'ENTER': (0x14, 1), 'LEAVE': (0x15, 0),
    'LOADL': (0x16, 1), 'STORL': (0x17, 1), 'LOADARG': (0x18, 1),
    'PUSHR': (0x19, 1), 'POPR': (0x1A, 1),
    'PUSHALL': (0x1B, 0), 'POPALL': (0x1C, 0), 'CSPGET': (0x1D, 0), 'CSPSET': (0x1E, 0),
    'ADD': (0x40, 0), 'SUB': (0x41, 0), 'MUL': (0x42, 0), 'MULH': (0x43, 0),
    'DIV': (0x44, 0), 'DIVU': (0x45, 0), 'REM': (0x46, 0), 'REMU': (0x47, 0),
    'AND': (0x48, 0), 'OR': (0x49, 0), 'XOR': (0x4A, 0),
    'NOT': (0x4B, 0), 'NEG': (0x4C, 0),
    'SHL': (0x4D, 0), 'SHR': (0x4E, 0), 'SAR': (0x4F, 0),
    'INC': (0x50, 0), 'DEC': (0x51, 0),
    'CMP': (0x52, 0), 'TST': (0x53, 0),
    'EQ': (0x54, 0), 'NE': (0x55, 0), 'LT': (0x56, 0), 'LTU': (0x57, 0),
    'MIN': (0x58, 0), 'MAX': (0x59, 0),
    'CLZ': (0x5A, 0), 'CTZ': (0x5B, 0), 'POPCNT': (0x5C, 0), 'REV8': (0x5D, 0),
    'EXT8': (0x5E, 0), 'EXT16': (0x5F, 0), 'EXT32': (0x60, 0),
    'LOAD8': (0x80, 0), 'LOAD16': (0x81, 0), 'LOAD32': (0x82, 0), 'LOAD64': (0x83, 0),
    'LOADU8': (0x84, 0), 'LOADU16': (0x85, 0), 'LOADU32': (0x86, 0),
    'STORE8': (0x87, 0), 'STORE16': (0x88, 0),
    'STORE32': (0x89, 0), 'STORE64': (0x8A, 0),
    'LOADO': (0x8B, 1), 'STOREO': (0x8C, 1),
    'LOADX': (0x8D, 0), 'STOREX': (0x8E, 0),
    'MEMCPY': (0x8F, 0), 'MEMSET': (0x90, 0), 'MEMCMP': (0x91, 0),
    'LR': (0x98, 0), 'SC': (0x99, 0),
    'AMOADD': (0x9A, 0), 'AMOSWAP': (0x9B, 0),
    'AMOAND': (0x9C, 0), 'AMOOR': (0x9D, 0),
    'JMP': (0xA0, 2), 'JMP32': (0xA1, 4),
    'BZ': (0xA2, 2), 'BNZ': (0xA3, 2),
    'BLT': (0xA4, 2), 'BGE': (0xA5, 2), 'BEQ': (0xA6, 2), 'BNE': (0xA7, 2),
    'CALL': (0xA8, 2), 'CALL32': (0xA9, 4),
    'CALLI': (0xAA, 0), 'RET': (0xAB, 0), 'JMPI': (0xAC, 0),
    'LOOP': (0xAD, 2), 'TRAP': (0xAE, 1),
    'ECALL': (0xC0, 0), 'EBREAK': (0xC1, 0),
    'MRET': (0xC2, 0), 'SRET': (0xC3, 0), 'HRET': (0xC4, 0),
    'WFI': (0xC5, 0),
    'CSRGET': (0xC6, 1), 'CSRSET': (0xC7, 1),
    'FENCE': (0xC8, 0), 'FENCEI': (0xC9, 0),
    'RDPMC': (0xCA, 1),
    'RDTIME': (0xCB, 0), 'RDCYCLE': (0xCC, 0),
    'TLBFLUSH': (0xCD, 0),
    'FADD.S': (0xD0, 0), 'FADD.D': (0xD1, 0),
    'FSUB.S': (0xD2, 0), 'FSUB.D': (0xD3, 0),
    'FMUL.S': (0xD4, 0), 'FMUL.D': (0xD5, 0),
    'FDIV.S': (0xD6, 0), 'FDIV.D': (0xD7, 0),
    'FSQRT.D': (0xD8, 0),
    'F2I': (0xD9, 0), 'I2F': (0xDA, 0),
    'F2S': (0xDB, 0), 'S2F': (0xDC, 0),
    'FCMP.D': (0xDD, 0), 'FMIN.D': (0xDE, 0), 'FMAX.D': (0xDF, 0),
    'VPUSH': (0xE0, 0), 'VPOP': (0xE1, 0),
    'VADD.4S': (0xE2, 0), 'VADD.2D': (0xE3, 0),
    'VSUB.4S': (0xE4, 0), 'VMUL.4S': (0xE5, 0),
    'VAND': (0xE6, 0), 'VOR': (0xE7, 0), 'VXOR': (0xE8, 0),
    'VSHL.4S': (0xE9, 0),
    'VLOAD': (0xEA, 0), 'VSTORE': (0xEB, 0),
    'VCMP.4S': (0xEC, 0), 'VSEL': (0xED, 0),
    'AESENC': (0xF0, 0), 'AESDEC': (0xF1, 0), 'AESKEY': (0xF2, 0),
    'SHA256C': (0xF3, 0),
    'RDRAND': (0xF4, 0), 'CPUID': (0xF5, 0),
}

# Signed range for each immediate size (PUSH8/16/32 and branch offsets)
SIGNED_RANGES = {1: (-128, 127), 2: (-32768, 32767), 4: (-2**31, 2**31 - 1)}

def parse_int(s):
    s = s.strip()
    if s.startswith('0x'): return int(s, 16)
    if s.startswith('0b'): return int(s, 2)
    return int(s)

def assemble(src):
    lines = []
    for raw in src.split('\n'):
        line = raw.split(';')[0].strip()
        if line: lines.append(line)
    # Pass 1: labels
    labels = {}
    addr = 0
    items = []  # (addr, kind, payload)
    for line in lines:
        if line.endswith(':'):
            labels[line[:-1].strip()] = addr
            continue
        parts = line.split()
        mn = parts[0].upper()
        if mn == '.WORD':
            items.append((addr, 'word', parse_int(parts[1])))
            addr += 8
            continue
        if mn not in OPS:
            raise SyntaxError(f"Unknown mnemonic: {mn}")
        opc, isz = OPS[mn]
        items.append((addr, 'insn', (mn, opc, isz, parts[1:])))
        addr += 1 + isz
    # Pass 2: emit
    out = bytearray()
    for (a, kind, payload) in items:
        if kind == 'word':
            out += struct.pack('<Q', payload & 0xFFFFFFFFFFFFFFFF)
            continue
        mn, opc, isz, args = payload
        out.append(opc)
        if isz == 0:
            if args:
                raise SyntaxError(f"{mn} takes no operands")
            continue
        if not args:
            raise SyntaxError(f"{mn} requires an operand")
        arg = args[0]
        if len(args) > 1:
            raise SyntaxError(f"{mn} takes exactly one operand")
        # Branch/call → label or numeric offset
        if mn in ('JMP','JMP32','BZ','BNZ','BLT','BGE','BEQ','BNE','CALL','CALL32','LOOP'):
            insn_len = 1 + isz
            if arg in labels:
                target = labels[arg]
                off = target - (a + insn_len)
            else:
                off = parse_int(arg)
            lo, hi = SIGNED_RANGES[isz]
            if not (lo <= off <= hi):
                raise ValueError(f"{mn} offset {off} out of range for {isz}-byte immediate")
            out += off.to_bytes(isz, 'little', signed=True)
        elif mn == 'PUSH64':
            if arg in labels:
                v = labels[arg]
            else:
                v = parse_int(arg)
            if not (0 <= v <= 0xFFFFFFFFFFFFFFFF):
                raise ValueError(f"PUSH64 value out of range: {arg}")
            out += v.to_bytes(8, 'little', signed=False)
        else:
            # All other immediates are signed (PUSH8/16/32, ENTER, LOADL, etc.)
            v = parse_int(arg)
            lo, hi = SIGNED_RANGES[isz]
            if not (lo <= v <= hi):
                raise ValueError(
                    f"{mn} immediate {v} out of range [{lo}, {hi}] for {isz}-byte field; "
                    f"use a wider PUSH")
            out += v.to_bytes(isz, 'little', signed=True)
    return bytes(out)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    argv = sys.argv[1:]
    out_path = 'a.out.bin'
    src_path = None
    i = 0
    while i < len(argv):
        if argv[i] == '-o':
            if i + 1 >= len(argv):
                print("error: -o requires a filename", file=sys.stderr); sys.exit(1)
            out_path = argv[i + 1]
            i += 2
        elif argv[i].startswith('-'):
            print(f"error: unknown flag {argv[i]}", file=sys.stderr); sys.exit(1)
        else:
            if src_path is not None:
                print("error: multiple input files", file=sys.stderr); sys.exit(1)
            src_path = argv[i]
            i += 1
    if src_path is None:
        print(__doc__); sys.exit(1)
    try:
        src = open(src_path).read()
    except OSError as e:
        print(f"error: cannot read {src_path}: {e}", file=sys.stderr); sys.exit(1)
    try:
        binary = assemble(src)
    except (SyntaxError, ValueError) as e:
        print(f"assembly error: {e}", file=sys.stderr); sys.exit(1)
    with open(out_path, 'wb') as f:
        f.write(binary)
    print(f"Assembled {len(binary)} bytes → {out_path}")

if __name__ == '__main__':
    main()
