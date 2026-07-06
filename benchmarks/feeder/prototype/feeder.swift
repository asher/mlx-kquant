// Feeder-loop prototype: GPU-resident MoE decode skeleton with a CPU control/IO plane.
//
// Simulates MiniMax-M2.7-class decode (62 layers, 256 experts, top-8, ~9.28MiB/expert
// at Q5_K_M) with bandwidth-faithful kernels:
//   per layer, pre-encoded in ONE command buffer per token:
//     [spine+router kernel] -> encodeSignal(routerDone) -> encodeWait(weightsReady)
//     -> [expert kernel: runtime data-driven addressing resident-arena vs staging-ring]
// The CPU feeder polls routerDone, preads missing experts from the real GGUF
// (F_NOCACHE) into the staging ring, and host-signals weightsReady. Residency and
// routing are deterministic hashes computed identically on CPU and GPU, so no
// CPU-written control state crosses mid-buffer — only weight bytes and event values.
//
// Usage: feeder <residentPct 0-100> <mode: demand|prefetch> <tokens> [--kill-feeder]
import Metal
import Foundation

let NLAYERS = 62
let NEXPERTS_USED = 8
let EXPERT_BYTES = 9_732_096            // 594 * 16384 (3 matrices of 3072x1536 @ 5.5bpw)
let EXPERT_W4 = EXPERT_BYTES / 16       // uint4 words
let SPINE_BYTES = 12 * 1 << 20          // per-layer resident spine reads
let SPINE_W4 = SPINE_BYTES / 16
let ARENA_BYTES = 4 << 30
let ARENA_W4 = ARENA_BYTES / 16
let TG_PER_EXPERT = 32
let GGUF = "/Users/asher/llm/gguf/MiniMax-M2.7-ultra-uncensored-heretic-GGUF/MiniMax-M2.7-BF16-ultra-uncensored-heretic-Q5_K_M.gguf"

func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
func hash3(_ a: UInt32, _ b: UInt32, _ c: UInt32) -> UInt32 {
    var h = a &* 2654435761 ^ b &* 2246822519 ^ c &* 3266489917
    h ^= h >> 15; h = h &* 2246822519; h ^= h >> 13
    return h
}
func expertIds(token: Int, layer: Int) -> [UInt32] {
    let base = hash3(UInt32(token), UInt32(layer), 0xABCD1234)
    return (0..<NEXPERTS_USED).map { (base &+ UInt32($0) &* 37) & 255 }
}
func isResident(layer: Int, id: UInt32, residentPct: Int) -> Bool {
    hash3(UInt32(layer), id, 0x51EDA127) % 100 < UInt32(residentPct)
}

let kernelSrc = """
#include <metal_stdlib>
using namespace metal;
inline uint hash3(uint a, uint b, uint c) {
  uint h = a * 2654435761u ^ b * 2246822519u ^ c * 3266489917u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return h;
}
kernel void spine_router(
    device const uint4* spine [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device atomic_uint* out [[buffer(2)]],
    constant uint2& p [[buffer(3)]],          // (layer, token)
    constant uint& spineW4 [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint nthreads [[threads_per_grid]]) {
  uint L = p.x, T = p.y;
  uint4 acc = uint4(0);
  for (uint i = tid; i < spineW4; i += nthreads) acc += spine[i];
  atomic_fetch_add_explicit(&out[L], acc.x + acc.y + acc.z + acc.w, memory_order_relaxed);
  if (tid < 8) {
    uint base = hash3(T, L, 0xABCD1234u);
    ids[L * 8 + tid] = (base + tid * 37u) & 255u;
  }
}
kernel void experts(
    device const uint* idsAll [[buffer(0)]],
    device const uint4* arena [[buffer(1)]],
    device const uint4* staging [[buffer(2)]],
    device atomic_uint* out [[buffer(3)]],
    constant uint4& p [[buffer(4)]],          // (layer, token, residentPct, tgPerExpert)
    constant uint2& sizes [[buffer(5)]],      // (expertW4, arenaW4)
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
  uint L = p.x, residentPct = p.z, tgPer = p.w;
  uint e = tgpig / tgPer;
  uint part = tgpig % tgPer;
  uint id = idsAll[L * 8 + e];
  uint expertW4 = sizes.x;
  bool resident = (hash3(L, id, 0x51EDA127u) % 100u) < residentPct;
  device const uint4* src;
  if (resident) {
    uint off = hash3(L, id, 0x00C0FFEEu) % (sizes.y - expertW4);
    src = arena + off;
  } else {
    ulong slot = (ulong)(L * 8u + e);
    src = staging + slot * (ulong)expertW4;
  }
  uint per = (expertW4 + tgPer - 1) / tgPer;
  uint start = part * per;
  uint end = min(start + per, expertW4);
  uint4 acc = uint4(0);
  for (uint i = start + tid; i < end; i += tgsize) acc += src[i];
  uint s = acc.x + acc.y + acc.z + acc.w;
  if ((s & 0xFFu) == 0x7Fu) s += 1;   // keep the compiler honest, ~never taken
  atomic_fetch_add_explicit(&out[64 + L], s, memory_order_relaxed);
}
"""

// --- Setup ---------------------------------------------------------------
let args = CommandLine.arguments
guard args.count >= 4, let residentPct = Int(args[1]), let nTokens = Int(args[3]) else {
    print("usage: feeder <residentPct> <demand|prefetch> <tokens> [--kill-feeder]"); exit(1)
}
let prefetch = args[2] == "prefetch"
let killFeeder = args.contains("--kill-feeder")

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue(maxCommandBufferCount: 32)!
let lib = try! device.makeLibrary(source: kernelSrc, options: nil)
let psoSpine = try! device.makeComputePipelineState(function: lib.makeFunction(name: "spine_router")!)
let psoExperts = try! device.makeComputePipelineState(function: lib.makeFunction(name: "experts")!)

let arena = device.makeBuffer(length: ARENA_BYTES, options: .storageModeShared)!
let spineArena = device.makeBuffer(length: SPINE_BYTES, options: .storageModeShared)!  // shared across layers (bandwidth model)
let staging = device.makeBuffer(length: NLAYERS * NEXPERTS_USED * EXPERT_BYTES, options: .storageModeShared)!
let idsBuf = device.makeBuffer(length: NLAYERS * 8 * 4, options: .storageModeShared)!
let outBuf = device.makeBuffer(length: 4 * 256, options: .storageModeShared)!
let routerDone = device.makeSharedEvent()!
let weightsReady = device.makeSharedEvent()!

// Touch arena/spine once so first-token timing isn't page-fault noise.
memset(arena.contents(), 0x5A, ARENA_BYTES)
memset(spineArena.contents(), 0xA5, SPINE_BYTES)

let fd = open(GGUF, O_RDONLY)
guard fd >= 0 else { print("cannot open gguf"); exit(1) }
_ = fcntl(fd, F_NOCACHE, 1)
let fsize = lseek(fd, 0, SEEK_END)

let idsPtr = idsBuf.contents().bindMemory(to: UInt32.self, capacity: NLAYERS * 8)

func vVal(_ token: Int, _ layer: Int) -> UInt64 { UInt64(token * NLAYERS + layer + 1) }

func encodeToken(_ token: Int) -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    for L in 0..<NLAYERS {
        let e1 = cb.makeComputeCommandEncoder()!
        e1.setComputePipelineState(psoSpine)
        e1.setBuffer(spineArena, offset: 0, index: 0)
        e1.setBuffer(idsBuf, offset: 0, index: 1)
        e1.setBuffer(outBuf, offset: 0, index: 2)
        var p = SIMD2<UInt32>(UInt32(L), UInt32(token))
        var sw = UInt32(SPINE_W4)
        e1.setBytes(&p, length: 8, index: 3)
        e1.setBytes(&sw, length: 4, index: 4)
        e1.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e1.endEncoding()
        cb.encodeSignalEvent(routerDone, value: vVal(token, L))
        cb.encodeWaitForEvent(weightsReady, value: vVal(token, L))
        let e2 = cb.makeComputeCommandEncoder()!
        e2.setComputePipelineState(psoExperts)
        e2.setBuffer(idsBuf, offset: 0, index: 0)
        e2.setBuffer(arena, offset: 0, index: 1)
        e2.setBuffer(staging, offset: 0, index: 2)
        e2.setBuffer(outBuf, offset: 0, index: 3)
        var p4 = SIMD4<UInt32>(UInt32(L), UInt32(token), UInt32(residentPct), UInt32(TG_PER_EXPERT))
        var sizes = SIMD2<UInt32>(UInt32(EXPERT_W4), UInt32(ARENA_W4))
        e2.setBytes(&p4, length: 16, index: 4)
        e2.setBytes(&sizes, length: 8, index: 5)
        e2.dispatchThreadgroups(MTLSize(width: NEXPERTS_USED * TG_PER_EXPERT, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e2.endEncoding()
    }
    return cb
}

// Read one expert's bytes into its staging slot. Returns bytes read.
func readExpert(token: Int, layer: Int, e: Int, id: UInt32) -> Int {
    let slot = layer * 8 + e
    let dst = staging.contents() + slot * EXPERT_BYTES
    let off = (Int64(hash3(UInt32(token), UInt32(layer), id)) % ((fsize - Int64(EXPERT_BYTES)) / 16384)) * 16384
    var done = 0
    while done < EXPERT_BYTES {
        let n = pread(fd, dst + done, EXPERT_BYTES - done, off + Int64(done))
        if n <= 0 { break }
        done += n
    }
    return done
}

// Load all misses for (token, layer) in parallel; returns (bytes, ioWallNs).
func loadLayer(token: Int, layer: Int) -> (Int, UInt64) {
    let ids = expertIds(token: token, layer: layer)
    let misses = ids.enumerated().filter { !isResident(layer: layer, id: $0.element, residentPct: residentPct) }
    if misses.isEmpty { return (0, 0) }
    let t0 = now()
    var bytes = 0
    let lock = NSLock()
    DispatchQueue.concurrentPerform(iterations: misses.count) { i in
        let (e, id) = misses[i]
        let n = readExpert(token: token, layer: layer, e: e, id: id)
        lock.lock(); bytes += n; lock.unlock()
    }
    return (bytes, now() - t0)
}

let POISON: UInt64 = .max
var poisoned = false

func pollRouter(_ value: UInt64) -> Bool {
    let t0 = now()
    while routerDone.signaledValue < value {
        if now() - t0 > 5_000_000_000 {
            print("  [watchdog] routerDone stuck at \(routerDone.signaledValue) waiting \(value) — poison-releasing")
            weightsReady.signaledValue = POISON
            poisoned = true
            return false
        }
    }
    return true
}

print("feeder proto: resident=\(residentPct)% mode=\(prefetch ? "prefetch" : "demand") tokens=\(nTokens) killFeeder=\(killFeeder)")
print("staging=\(NLAYERS * 8 * EXPERT_BYTES / (1 << 20))MiB arena=\(ARENA_BYTES >> 30)GiB spine=\(SPINE_BYTES >> 20)MiB/layer x\(NLAYERS)")

var tokenTimes: [Double] = []
for T in 0..<nTokens {
    let tEnc0 = now()
    let cb = encodeToken(T)
    let encMs = Double(now() - tEnc0) / 1e6
    var ioBytes = 0
    var ioNs: UInt64 = 0
    var exposedNs: UInt64 = 0     // sum of routerDone-observed -> weightsReady-signaled (GPU-blocked window)
    var idsMismatch = 0
    let t0 = now()
    cb.commit()

    if prefetch {
        // One layer ahead: reads for L+1 happen while the GPU runs layer L.
        var (b, n) = loadLayer(token: T, layer: 0)  // layer 0 has no overlap partner
        ioBytes += b; ioNs += n
        for L in 0..<NLAYERS {
            if killFeeder && T == 1 && L == 30 { print("  [test] feeder dying at layer 30"); break }
            if !pollRouter(vVal(T, L)) { break }
            let tSeen = now()
            // Verify router matches prediction (exact here; readback exercises the path).
            let want = expertIds(token: T, layer: L)
            for e in 0..<8 where idsPtr[L * 8 + e] != want[e] { idsMismatch += 1 }
            weightsReady.signaledValue = vVal(T, L)   // L's weights were loaded during layer L-1
            exposedNs += now() - tSeen
            if L + 1 < NLAYERS {
                let (b2, n2) = loadLayer(token: T, layer: L + 1)
                ioBytes += b2; ioNs += n2
            }
        }
    } else {
        for L in 0..<NLAYERS {
            if killFeeder && T == 1 && L == 30 { print("  [test] feeder dying at layer 30"); break }
            if !pollRouter(vVal(T, L)) { break }
            let tSeen = now()
            let want = expertIds(token: T, layer: L)
            for e in 0..<8 where idsPtr[L * 8 + e] != want[e] { idsMismatch += 1 }
            let (b, n) = loadLayer(token: T, layer: L)
            ioBytes += b; ioNs += n
            weightsReady.signaledValue = vVal(T, L)
            exposedNs += now() - tSeen
        }
    }
    if killFeeder && T == 1 && !poisoned {
        // Feeder is "dead" mid-token: demonstrate the un-wedge property.
        usleep(3_000_000)
        print("  [watchdog] feeder dead; poison-releasing all encoded waits")
        weightsReady.signaledValue = POISON
        poisoned = true
    }
    cb.waitUntilCompleted()
    let ms = Double(now() - t0) / 1e6
    if poisoned {
        print("  [recovery] command buffer drained after poison (status=\(cb.status.rawValue), error=\(cb.error == nil ? "none" : String(describing: cb.error!)))")
        print("  [recovery] GPU released without reboot; token discarded; exiting cleanly")
        exit(0)
    }
    tokenTimes.append(ms)
    let rate = ioNs > 0 ? Double(ioBytes) / (Double(ioNs) / 1e9) / 1e9 : 0
    print(String(format: "  token %d: %.1fms (encode %.1fms)  io=%dMB feederBusy=%.0fms @%.2fGB/s  gpuBlocked=%.0fms  idsMismatch=%d",
                 T, ms, encMs, ioBytes >> 20, Double(ioNs) / 1e6, rate, Double(exposedNs) / 1e6, idsMismatch))
}
if tokenTimes.count > 1 {
    let steady = tokenTimes.dropFirst().sorted()
    let med = steady[steady.count / 2]
    print(String(format: "median steady-state token: %.1fms  (%.2f tok/s)", med, 1000.0 / med))
}
print("done")
