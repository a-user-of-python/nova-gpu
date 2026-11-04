// NovaBusClient.swift — host-side NovaBus master.
//
// This is how the test app (and the host tests) talk to the GPU: by
// writing command descriptors into guest RAM and ringing the MMIO
// doorbell — byte for byte the same protocol the NovaCPU driver uses.
// UI code never calls NovaGPU internals; everything goes over the bus.
// (The one exception is reading PRESENTed framebuffers out of guest
// RAM after a DOWNLOAD, which is what the bus is for.)
//
// The client also plays the CPU's half of the service protocol: if the
// device raises a service request while the client waits, the client's
// `serviceHandler` is invoked with the request record, and the client
// writes REQ_ACK when the handler returns.

import Foundation

enum BusClientError: Error {
    case deviceError(UInt64)
    case timeout
}

final class NovaBusClient {
    let memory: NovaMemory
    let device: NovaGPUDevice
    let ringBase: UInt64
    let ringSlots: UInt64
    let reqPage: UInt64
    private var tail: UInt64 = 0

    /// Called when the GPU requests service (request record words).
    /// Return to acknowledge. Default: acknowledge immediately.
    var serviceHandler: (([UInt64]) -> Void)?

    init(memory: NovaMemory, device: NovaGPUDevice,
         ringBase: UInt64 = 0x20000, ringSlots: UInt64 = 64, reqPage: UInt64 = 0x21000) {
        self.memory = memory
        self.device = device
        self.ringBase = ringBase
        self.ringSlots = ringSlots
        self.reqPage = reqPage
        device.attach(to: memory)
        writeReg(.ringBase, ringBase)
        writeReg(.ringSlots, ringSlots)
        writeReg(.reqPage, reqPage)
    }

    private func writeReg(_ reg: GPURegister, _ v: UInt64) {
        memory.mmioWrite(NovaMemory.gpuBaseAddr + reg.rawValue, v)
    }
    private func readReg(_ reg: GPURegister) -> UInt64 {
        memory.mmioRead(NovaMemory.gpuBaseAddr + reg.rawValue)
    }

    var irqStatus: UInt64 { readReg(.irqStatus) }
    var errorCode: UInt64 { readReg(.error) }
    var statusWord: UInt64 { readReg(.status) }

    /// Submits one descriptor and waits for completion, servicing GPU
    /// requests while waiting. Returns the completed descriptor.
    @discardableResult
    func submit(opcode: GPUCommand, fence: UInt64 = 0,
                args: [UInt64] = [0, 0, 0, 0, 0]) throws -> GPUDescriptor {
        var desc = GPUDescriptor(opcode: opcode.rawValue, fence: fence, args: args, status: 0)
        let slot = tail % ringSlots
        let addr = ringBase + slot * UInt64(GPUDescriptor.byteSize)
        memory.storeHostBytes(desc.packed(), at: addr)
        tail += 1
        writeReg(.doorbell, tail)
        device.servicePendingWork()
        // Poll the descriptor's status word in guest RAM.
        let deadline = Date().addingTimeInterval(120)
        while Date() < deadline {
            let bytes = memory.loadHostBytes(from: addr, count: GPUDescriptor.byteSize)
            desc = GPUDescriptor.unpack(bytes)
            if desc.status == GPUDescriptorStatus.done.rawValue {
                return desc
            }
            if desc.status == GPUDescriptorStatus.error.rawValue {
                throw BusClientError.deviceError(readReg(.error))
            }
            if desc.status == GPUDescriptorStatus.waitingService.rawValue {
                try servicePendingRequest()
                // After the ACK the device completes the descriptor
                // synchronously in this implementation; loop re-reads.
                continue
            }
            Thread.sleep(forTimeInterval: 0.0002)
        }
        throw BusClientError.timeout
    }

    private func servicePendingRequest() throws {
        let seq = readReg(.reqPending)
        guard seq != 0 else { return }
        let bytes = memory.loadHostBytes(from: reqPage, count: 64)
        var words = [UInt64](repeating: 0, count: 8)
        for i in 0..<8 {
            var v: UInt64 = 0
            for k in 0..<8 { v |= UInt64(bytes[i * 8 + k]) << UInt64(k * 8) }
            words[i] = v
        }
        serviceHandler?(words)
        writeReg(.reqAck, seq)
        device.servicePendingWork()
    }

    // MARK: - Convenience wrappers (one descriptor each)

    func alloc(_ size: UInt64) throws -> UInt64 {
        let d = try submit(opcode: .alloc, args: [size, 0, 0, 0, 0])
        return d.args[0]
    }
    func free(_ h: UInt64) throws { _ = try submit(opcode: .free, args: [h, 0, 0, 0, 0]) }
    func upload(_ h: UInt64, off: UInt64 = 0, guestSrc: UInt64, len: Int) throws {
        _ = try submit(opcode: .upload, args: [h, off, guestSrc, UInt64(len), 0])
    }
    func uploadData(_ h: UInt64, off: UInt64 = 0, data: [UInt8], guestStaging: UInt64) throws {
        memory.storeHostBytes(data, at: guestStaging)
        try upload(h, off: off, guestSrc: guestStaging, len: data.count)
    }
    func download(_ h: UInt64, off: UInt64 = 0, guestDst: UInt64, len: Int) throws {
        _ = try submit(opcode: .download, args: [h, off, guestDst, UInt64(len), 0])
    }
    func downloadData(_ h: UInt64, off: UInt64 = 0, len: Int, guestStaging: UInt64) throws -> [UInt8] {
        try download(h, off: off, guestDst: guestStaging, len: len)
        return memory.loadHostBytes(from: guestStaging, count: len)
    }
    func fill(_ h: UInt64, off: UInt64, len: UInt64, pattern: UInt32) throws {
        _ = try submit(opcode: .fill, args: [h, off, len, UInt64(pattern), 0])
    }
    func setShader(_ which: GPUCommand, programHandle: UInt64, off: UInt64, len: Int) throws {
        _ = try submit(opcode: which, args: [programHandle, off, UInt64(len), 0, 0])
    }
    func setUniform(_ index: Int, _ v: (Float, Float, Float, Float)) throws {
        _ = try submit(opcode: .setUniform,
                       args: [UInt64(index), UInt64(v.0.bitPattern), UInt64(v.1.bitPattern),
                              UInt64(v.2.bitPattern), UInt64(v.3.bitPattern)])
    }
    func setVBuf(_ h: UInt64) throws { _ = try submit(opcode: .setVBuf, args: [h, 0, 0, 0, 0]) }
    func setIBuf(_ h: UInt64) throws { _ = try submit(opcode: .setIBuf, args: [h, 0, 0, 0, 0]) }
    func setFBuf(_ h: UInt64, _ w: Int, _ hgt: Int) throws {
        _ = try submit(opcode: .setFBuf, args: [h, UInt64(w), UInt64(hgt), 0, 0])
    }
    func setTexture(unit: Int, _ h: UInt64, _ w: Int, _ hgt: Int) throws {
        _ = try submit(opcode: .setTexture, args: [UInt64(unit), h, UInt64(w), UInt64(hgt), 0])
    }
    func clear(_ rgba: UInt32) throws { _ = try submit(opcode: .clear, args: [UInt64(rgba), 0, 0, 0, 0]) }
    func draw(indexCount: Int, flags: UInt64 = 0, assetStaging: UInt64 = 0,
              assetLen: UInt64 = 0, assetTarget: UInt64 = 0) throws {
        _ = try submit(opcode: .draw, args: [UInt64(indexCount), flags, assetStaging, assetLen, assetTarget])
    }
    func launch(lanes: Int, buffer: UInt64) throws {
        _ = try submit(opcode: .launch, args: [UInt64(lanes), buffer, 0, 0, 0])
    }
    func present(_ fb: UInt64) throws { _ = try submit(opcode: .present, args: [fb, 0, 0, 0, 0]) }
}
