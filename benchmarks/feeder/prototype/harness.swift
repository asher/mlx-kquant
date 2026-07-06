// CPU<->GPU sync primitive latency + coherence harness for Apple Silicon.
// Measures the cost of supported synchronization primitives (MTLSharedEvent
// encoded waits/signals, released/observed by the host mid-command-buffer)
// vs full command-buffer splits, and stress-tests data coherence at encoder
// boundaries (the thing MLX_METAL_FAST_SYNCH's spin-kernel tries to skip).
import Metal
import Foundation

func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue(maxCommandBufferCount: 128)!

let kernelSrc = """
#include <metal_stdlib>
using namespace metal;
kernel void csum(device const uint* data [[buffer(0)]],
                 device uint* out [[buffer(1)]],
                 constant uint& nwords [[buffer(2)]],
                 constant uint& outIdx [[buffer(3)]],
                 uint tid [[thread_position_in_threadgroup]],
                 uint tgsize [[threads_per_threadgroup]]) {
  threadgroup uint partial[256];
  uint sum = 0;
  for (uint i = tid; i < nwords; i += tgsize) sum += data[i];
  partial[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    uint total = 0;
    for (uint i = 0; i < tgsize; i++) total += partial[i];
    out[outIdx] = total;
  }
}
"""
let lib = try! device.makeLibrary(source: kernelSrc, options: nil)
let pso = try! device.makeComputePipelineState(function: lib.makeFunction(name: "csum")!)

func encodeSum(_ cb: MTLCommandBuffer, slot: MTLBuffer, nwords: UInt32, out: MTLBuffer, outIdx: UInt32) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pso)
    e.setBuffer(slot, offset: 0, index: 0)
    e.setBuffer(out, offset: 0, index: 1)
    var n = nwords
    var oi = outIdx
    e.setBytes(&n, length: 4, index: 2)
    e.setBytes(&oi, length: 4, index: 3)
    e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding()
}

// Fill slot with a k-dependent pattern; return wrapping uint32 sum.
func fill(_ slot: MTLBuffer, nwords: Int, seed: UInt32) -> UInt32 {
    let p = slot.contents().bindMemory(to: UInt32.self, capacity: nwords)
    var sum: UInt32 = 0
    var v: UInt32 = seed &* 2654435761 &+ 12345
    for i in 0..<nwords {
        v = v &* 1664525 &+ 1013904223
        p[i] = v
        sum = sum &+ v
        _ = i
    }
    return sum
}

func stats(_ label: String, _ ns: [UInt64], mismatches: Int) {
    let s = ns.sorted()
    let us = { (x: UInt64) in Double(x) / 1000.0 }
    let p = { (q: Double) in us(s[min(s.count - 1, Int(Double(s.count) * q))]) }
    print(String(format: "%@  n=%d  min=%.1fus p50=%.1fus p90=%.1fus p99=%.1fus max=%.1fus  mismatches=%d",
                 label, s.count, us(s[0]), p(0.5), p(0.9), p(0.99), us(s[s.count-1]), mismatches))
}

let POISON_DEADLINE_NS: UInt64 = 15_000_000_000

// Poll a shared event value with deadline; poison-release `poisonTarget` on timeout.
// Returns false on timeout. This is the "un-wedge" path: encoded waits are always
// releasable from the host, unlike an in-kernel spin.
func pollEvent(_ ev: MTLSharedEvent, _ value: UInt64, poisonTarget: MTLSharedEvent, label: String) -> Bool {
    let t0 = now()
    while ev.signaledValue < value {
        if now() - t0 > POISON_DEADLINE_NS {
            print("TIMEOUT in \(label) waiting for \(value); poison-releasing GPU waits and exiting")
            poisonTarget.signaledValue = UInt64.max
            usleep(300_000)
            exit(2)
        }
    }
    return true
}

// T1: full command-buffer split per handoff (commit + waitUntilCompleted).
func t1_splitBuffer(iters: Int, slotWords: Int) {
    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: 4, options: .storageModeShared)!
    _ = fill(slot, nwords: slotWords, seed: 7)
    var ns: [UInt64] = []
    for _ in 0..<iters {
        let t0 = now()
        let cb = queue.makeCommandBuffer()!
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: 0)
        cb.commit()
        cb.waitUntilCompleted()
        ns.append(now() - t0)
    }
    stats("T1 cb-split          ", ns, mismatches: 0)
}

// T2: single command buffer; per segment: encodeWait(cpuReady) -> csum -> encodeSignal(gpuDone).
// CPU: fill slot, setSignaledValue (host release), spin-poll gpuDone, verify checksum.
// verifyData=false variant encodes no kernel (pure sync ping-pong floor).
func t2_pingpong(iters: Int, slotWords: Int, withKernel: Bool, label: String) {
    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: max(4, iters * 4), options: .storageModeShared)!
    let cpuReady = device.makeSharedEvent()!
    let gpuDone = device.makeSharedEvent()!
    let cb = queue.makeCommandBuffer()!
    for k in 0..<iters {
        cb.encodeWaitForEvent(cpuReady, value: UInt64(k + 1))
        if withKernel {
            encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: UInt32(k))
        }
        cb.encodeSignalEvent(gpuDone, value: UInt64(k + 1))
    }
    cb.addCompletedHandler { c in
        if c.status == .error { print("T2 cb error: \(String(describing: c.error))") }
    }
    cb.commit()
    var ns: [UInt64] = []
    var mism = 0
    let outP = out.contents().bindMemory(to: UInt32.self, capacity: max(1, iters))
    for k in 0..<iters {
        let expected = withKernel ? fill(slot, nwords: slotWords, seed: UInt32(k)) : 0
        let t0 = now()
        cpuReady.signaledValue = UInt64(k + 1)
        guard pollEvent(gpuDone, UInt64(k + 1), poisonTarget: cpuReady, label: label) else { return }
        ns.append(now() - t0)
        if withKernel && outP[k] != expected { mism += 1 }
    }
    stats(label, ns, mismatches: mism)
}

// T2p: waits already satisfied when GPU arrives (CPU pre-signals everything) —
// the steady state of a prefetch-ahead feeder. Measures per-segment cost sans stall.
func t2p_presignaled(iters: Int, slotWords: Int, encodeWaits: Bool, label: String) {
    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: iters * 4, options: .storageModeShared)!
    let cpuReady = device.makeSharedEvent()!
    let gpuDone = device.makeSharedEvent()!
    _ = fill(slot, nwords: slotWords, seed: 42)
    cpuReady.signaledValue = UInt64(iters + 1)   // all waits pre-satisfied
    let cb = queue.makeCommandBuffer()!
    for k in 0..<iters {
        if encodeWaits { cb.encodeWaitForEvent(cpuReady, value: UInt64(k + 1)) }
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: UInt32(k))
        cb.encodeSignalEvent(gpuDone, value: UInt64(k + 1))
    }
    let t0 = now()
    cb.commit()
    guard pollEvent(gpuDone, UInt64(iters), poisonTarget: cpuReady, label: label) else { return }
    let per = Double(now() - t0) / Double(iters) / 1000.0
    print(String(format: "%@  n=%d  avg per-segment=%.1fus (throughput mode)", label, iters, per))
}

// T2w: like T2 but with a second queue continuously running tiny kernels to keep
// the GPU clocked up. Does stall-wake latency drop when the GPU never idles?
func t2w_keepWarm(iters: Int, slotWords: Int) {
    let warmQueue = device.makeCommandQueue()!
    let warmSlot = device.makeBuffer(length: 4096 * 4, options: .storageModeShared)!
    let warmOut = device.makeBuffer(length: 4, options: .storageModeShared)!
    var stopWarm = false
    let warmDone = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
        while !stopWarm {
            let cb = warmQueue.makeCommandBuffer()!
            for _ in 0..<8 { encodeSum(cb, slot: warmSlot, nwords: 4096, out: warmOut, outIdx: 0) }
            cb.commit()
            cb.waitUntilCompleted()
        }
        warmDone.signal()
    }
    usleep(50_000)

    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: iters * 4, options: .storageModeShared)!
    let cpuReady = device.makeSharedEvent()!
    let gpuDone = device.makeSharedEvent()!
    let cb = queue.makeCommandBuffer()!
    for k in 0..<iters {
        cb.encodeWaitForEvent(cpuReady, value: UInt64(k + 1))
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: UInt32(k))
        cb.encodeSignalEvent(gpuDone, value: UInt64(k + 1))
    }
    cb.commit()
    var ns: [UInt64] = []
    var mism = 0
    let outP = out.contents().bindMemory(to: UInt32.self, capacity: iters)
    for k in 0..<iters {
        let expected = fill(slot, nwords: slotWords, seed: UInt32(k) &+ 31337)
        let t0 = now()
        cpuReady.signaledValue = UInt64(k + 1)
        guard pollEvent(gpuDone, UInt64(k + 1), poisonTarget: cpuReady, label: "T2w") else { return }
        ns.append(now() - t0)
        if outP[k] != expected { mism += 1 }
    }
    stopWarm = true
    warmDone.wait()
    stats("T2w warm-gpu pingpong", ns, mismatches: mism)
}

// T3: like T2 but CPU blocks in waitUntilSignaledValue (what MLX's default Event::wait does).
func t3_listener(iters: Int, slotWords: Int) {
    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: iters * 4, options: .storageModeShared)!
    let cpuReady = device.makeSharedEvent()!
    let gpuDone = device.makeSharedEvent()!
    let cb = queue.makeCommandBuffer()!
    for k in 0..<iters {
        cb.encodeWaitForEvent(cpuReady, value: UInt64(k + 1))
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: UInt32(k))
        cb.encodeSignalEvent(gpuDone, value: UInt64(k + 1))
    }
    cb.commit()
    var ns: [UInt64] = []
    var mism = 0
    let outP = out.contents().bindMemory(to: UInt32.self, capacity: iters)
    for k in 0..<iters {
        let expected = fill(slot, nwords: slotWords, seed: UInt32(k) &+ 9000)
        let t0 = now()
        cpuReady.signaledValue = UInt64(k + 1)
        if !gpuDone.wait(untilSignaledValue: UInt64(k + 1), timeoutMS: 15000) {
            print("TIMEOUT in T3; poisoning"); cpuReady.signaledValue = .max; exit(2)
        }
        ns.append(now() - t0)
        if outP[k] != expected { mism += 1 }
    }
    stats("T3 waitUntilSignaled ", ns, mismatches: mism)
}

// T4: adversarial coherence. Per segment, an UNGATED primer encoder reads the slot
// first (warming GPU caches with stale data, and proving it read the OLD pattern),
// then CPU rewrites the slot and host-signals, then the gated encoder re-reads.
// Any stale line surviving the encoder boundary => checksum mismatch (detected, not hung).
func t4_primedCoherence(iters: Int, slotWords: Int, label: String) {
    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: iters * 4, options: .storageModeShared)!
    let primeOut = device.makeBuffer(length: iters * 4, options: .storageModeShared)!
    let cpuReady = device.makeSharedEvent()!
    let primed = device.makeSharedEvent()!
    let gpuDone = device.makeSharedEvent()!
    var expectedPrev = fill(slot, nwords: slotWords, seed: 0xDEAD)
    let cb = queue.makeCommandBuffer()!
    for k in 0..<iters {
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: primeOut, outIdx: UInt32(k))
        cb.encodeSignalEvent(primed, value: UInt64(k + 1))
        cb.encodeWaitForEvent(cpuReady, value: UInt64(k + 1))
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: UInt32(k))
        cb.encodeSignalEvent(gpuDone, value: UInt64(k + 1))
    }
    cb.commit()
    var ns: [UInt64] = []
    var mism = 0, primeMism = 0
    let outP = out.contents().bindMemory(to: UInt32.self, capacity: iters)
    let primeP = primeOut.contents().bindMemory(to: UInt32.self, capacity: iters)
    for k in 0..<iters {
        guard pollEvent(primed, UInt64(k + 1), poisonTarget: cpuReady, label: label) else { return }
        let expected = fill(slot, nwords: slotWords, seed: UInt32(k) &+ 5555)
        let t0 = now()
        cpuReady.signaledValue = UInt64(k + 1)
        guard pollEvent(gpuDone, UInt64(k + 1), poisonTarget: cpuReady, label: label) else { return }
        ns.append(now() - t0)
        if outP[k] != expected { mism += 1 }
        if primeP[k] != expectedPrev { primeMism += 1 }
        expectedPrev = expected
    }
    stats(label, ns, mismatches: mism)
    print("   (primer-read-old-data check mismatches: \(primeMism) — should be 0; nonzero means test harness bug, not coherence result)")
}

// T5: Tier-A style — each segment is its own pre-committed command buffer gated by a
// leading encoded wait. Measures per-handoff cost when everything is at documented
// command-buffer granularity, with commit cost paid up front.
func t5_gatedBuffers(iters: Int, slotWords: Int) {
    let slot = device.makeBuffer(length: slotWords * 4, options: .storageModeShared)!
    let out = device.makeBuffer(length: iters * 4, options: .storageModeShared)!
    let cpuReady = device.makeSharedEvent()!
    let gpuDone = device.makeSharedEvent()!
    for k in 0..<iters {
        let cb = queue.makeCommandBuffer()!
        cb.encodeWaitForEvent(cpuReady, value: UInt64(k + 1))
        encodeSum(cb, slot: slot, nwords: UInt32(slotWords), out: out, outIdx: UInt32(k))
        cb.encodeSignalEvent(gpuDone, value: UInt64(k + 1))
        cb.commit()
    }
    var ns: [UInt64] = []
    var mism = 0
    let outP = out.contents().bindMemory(to: UInt32.self, capacity: iters)
    for k in 0..<iters {
        let expected = fill(slot, nwords: slotWords, seed: UInt32(k) &+ 777)
        let t0 = now()
        cpuReady.signaledValue = UInt64(k + 1)
        guard pollEvent(gpuDone, UInt64(k + 1), poisonTarget: cpuReady, label: "T5") else { return }
        ns.append(now() - t0)
        if outP[k] != expected { mism += 1 }
    }
    stats("T5 gated-buffers     ", ns, mismatches: mism)
}

// T6: streaming data plane — MTLIOCommandQueue loads from the GGUF vs pread, random offsets.
func t6_io(path: String, chunkMB: Int, nchunks: Int) {
    let chunk = chunkMB * 1 << 20
    let fsize: Int
    do {
        let attrs = try FileManager.default.attributesOfItem(atPath: path)
        fsize = (attrs[.size] as! NSNumber).intValue
    } catch { print("T6 skipped: \(error)"); return }
    var v: UInt64 = 0x9E3779B97F4A7C15
    func freshOffsets() -> [Int] {
        var offs: [Int] = []
        for _ in 0..<nchunks {
            v = v &* 6364136223846793005 &+ 1442695040888963407
            offs.append(Int(v % UInt64((fsize - chunk) / 16384)) * 16384)
        }
        return offs
    }
    let totalGB = Double(chunk * nchunks) / 1e9

    // pread with F_NOCACHE (true cold-SSD numbers), fresh offsets per test
    let fd = open(path, O_RDONLY)
    _ = fcntl(fd, F_NOCACHE, 1)
    let offs1 = freshOffsets()
    let buf = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16384)
    var t0 = now()
    for o in offs1 { _ = pread(fd, buf, chunk, off_t(o)) }
    var dt = Double(now() - t0) / 1e9
    print(String(format: "T6 pread qd1 nocache %.2f GB in %.2fs = %.2f GB/s", totalGB, dt, totalGB / dt))
    buf.deallocate()

    let offs8 = freshOffsets()
    t0 = now()
    DispatchQueue.concurrentPerform(iterations: 8) { t in
        let b = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16384)
        var i = t
        while i < nchunks { _ = pread(fd, b, chunk, off_t(offs8[i])); i += 8 }
        b.deallocate()
    }
    dt = Double(now() - t0) / 1e9
    print(String(format: "T6 pread qd8 nocache %.2f GB in %.2fs = %.2f GB/s", totalGB, dt, totalGB / dt))
    close(fd)

    // MTLIO
    if #available(macOS 13.0, *) {
        do {
            let desc = MTLIOCommandQueueDescriptor()
            desc.type = .concurrent
            let ioq = try device.makeIOCommandQueue(descriptor: desc)
            let fh = try device.makeIOFileHandle(url: URL(fileURLWithPath: path))
            let big = device.makeBuffer(length: chunk, options: .storageModeShared)!
            // fresh offsets (avoid page cache from pread runs)
            var offs2: [Int] = []
            for _ in 0..<nchunks {
                v = v &* 6364136223846793005 &+ 1442695040888963407
                offs2.append(Int(v % UInt64((fsize - chunk) / 16384)) * 16384)
            }
            t0 = now()
            let iocb = ioq.makeCommandBuffer()
            for o in offs2 { iocb.load(big, offset: 0, size: chunk, sourceHandle: fh, sourceHandleOffset: o) }
            iocb.commit()
            iocb.waitUntilCompleted()
            dt = Double(now() - t0) / 1e9
            print(String(format: "T6 MTLIO concurrent  %.2f GB in %.2fs = %.2f GB/s (status=%d)", totalGB, dt, totalGB / dt, iocb.status.rawValue))
        } catch { print("T6 MTLIO failed: \(error)") }
    }
}

let args = CommandLine.arguments
let which = args.count > 1 ? args[1] : "all"
let SMALL = 4096          // 16KB slot
let BIG = 2 * 1 << 20     // 8MB slot (in words: 2M words)

print("device=\(device.name)  os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
if which == "all" || which == "sync" {
    t1_splitBuffer(iters: 200, slotWords: SMALL)
    t2_pingpong(iters: 400, slotWords: SMALL, withKernel: false, label: "T2e empty pingpong   ")
    t2_pingpong(iters: 400, slotWords: SMALL, withKernel: true,  label: "T2 pingpong 16KB     ")
    t2p_presignaled(iters: 400, slotWords: SMALL, encodeWaits: true,  label: "T2p presignaled waits")
    t2p_presignaled(iters: 400, slotWords: SMALL, encodeWaits: false, label: "T2b no waits (floor) ")
    t2w_keepWarm(iters: 300, slotWords: SMALL)
    t3_listener(iters: 200, slotWords: SMALL)
    t4_primedCoherence(iters: 300, slotWords: SMALL, label: "T4s primed 16KB      ")
    t4_primedCoherence(iters: 60,  slotWords: BIG,   label: "T4l primed 8MB       ")
    t5_gatedBuffers(iters: 48, slotWords: SMALL)
}
if which == "all" || which == "io" {
    t6_io(path: "/Users/asher/llm/gguf/MiniMax-M2.7-ultra-uncensored-heretic-GGUF/MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q5_K_M.gguf",
          chunkMB: 16, nchunks: 96)
}
print("done")
