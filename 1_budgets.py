"""
Lab 1 - Counting the three budgets: parameters, MACs, memory, energy.
Pure NumPy: no deep-learning framework needed. Runs on a laptop or Raspberry Pi.

Goal: build intuition for WHY a model that is fine in the cloud can be
infeasible on a microcontroller, before touching any framework.
"""
import numpy as np

# ---------------------------------------------------------------
# Part A: analytical cost model for common layers
# ---------------------------------------------------------------
def conv2d_cost(h, w, c_in, c_out, k, stride=1, groups=1):
    """Return (params, MACs, out_h, out_w) for a conv layer with 'same' padding."""
    oh, ow = h // stride, w // stride
    params = (k * k * (c_in // groups) * c_out) + c_out
    macs = oh * ow * c_out * (k * k * (c_in // groups))
    return params, macs, oh, ow

def dense_cost(n_in, n_out):
    return n_in * n_out + n_out, n_in * n_out

def build_standard_cnn(h=96, w=96, c=3):
    """A small plain CNN: every layer is a standard convolution."""
    layers, params, macs, act = [], 0, 0, []
    ch = c
    for c_out, stride in [(32, 2), (64, 2), (128, 2), (256, 2)]:
        p, m, h, w = conv2d_cost(h, w, ch, c_out, 3, stride)
        params += p; macs += m; act.append(h * w * c_out); ch = c_out
        layers.append(("conv3x3", c_out, p, m))
    p, m = dense_cost(ch, 10)
    params += p; macs += m
    return params, macs, max(act), layers

def build_separable_cnn(h=96, w=96, c=3):
    """Same widths, but depthwise-separable convolutions (MobileNet idea)."""
    layers, params, macs, act = [], 0, 0, []
    ch = c
    for c_out, stride in [(32, 2), (64, 2), (128, 2), (256, 2)]:
        pd, md, h, w = conv2d_cost(h, w, ch, ch, 3, stride, groups=ch)   # depthwise
        pp, mp, _, _ = conv2d_cost(h, w, ch, c_out, 1, 1)                # pointwise
        params += pd + pp; macs += md + mp; act.append(h * w * c_out); ch = c_out
        layers.append(("dw3x3+pw1x1", c_out, pd + pp, md + mp))
    p, m = dense_cost(ch, 10)
    params += p; macs += m
    return params, macs, max(act), layers

# ---------------------------------------------------------------
# Part B: does it fit on the device?
# ---------------------------------------------------------------
DEVICES = {
    # name: (flash_bytes, sram_bytes, approx peak MACs/s)  -- ORDER-OF-MAGNITUDE
    # figures for teaching; students should replace with datasheet values.
    "Cortex-M4 MCU (e.g. STM32F4)": (1_000_000,   192_000,        50e6),
    "Cortex-M7 MCU (e.g. STM32H7)": (2_000_000,   1_000_000,     400e6),
    "Raspberry Pi 4 (CPU)":         (8 * 2**30,   4 * 2**30,      10e9),
}

def report(name, params, macs, peak_act):
    print(f"\n=== {name} ===")
    print(f"  parameters      : {params:>10,}")
    print(f"  MACs / inference: {macs:>10,}  ({macs/1e6:.2f} M)")
    for bits, label in [(32, "FP32"), (8, "INT8")]:
        weights = params * bits // 8
        act = peak_act * bits // 8            # crude: largest single activation map
        print(f"  {label}: weights {weights/1e3:8.1f} kB | peak activation ~{act/1e3:8.1f} kB")
    print("  Fit check (INT8 weights -> flash, peak activation -> SRAM):")
    for dev, (flash, sram, rate) in DEVICES.items():
        w8, a8 = params, peak_act
        ok = (w8 <= flash) and (a8 <= sram)
        lat_ms = macs / rate * 1e3
        print(f"    {dev:32s} fits={str(ok):5s}  ideal latency >= {lat_ms:9.2f} ms")

if __name__ == "__main__":
    for nm, fn in [("Standard CNN", build_standard_cnn),
                   ("Depthwise-separable CNN", build_separable_cnn)]:
        p, m, a, _ = fn()
        report(nm, p, m, a)

    p1, m1, _, _ = build_standard_cnn()
    p2, m2, _, _ = build_separable_cnn()
    print(f"\nSeparable convs cut MACs by {m1/m2:.1f}x and params by {p1/p2:.1f}x.")

    # ---------------------------------------------------------------
    # Part C (STUDENT TASK): the arithmetic-intensity / roofline view
    # ---------------------------------------------------------------
    print("\n--- Roofline exercise ---")
    peak_flops = 10e9          # 10 GOPS compute roof   (edit for your device)
    mem_bw     = 4e9           # 4 GB/s memory bandwidth (edit for your device)
    ridge = peak_flops / mem_bw
    print(f"Ridge point = {ridge:.2f} ops/byte. Layers below it are MEMORY-bound.")
    # Dense layer, batch 1: 2*n_in*n_out ops, must read n_in*n_out weights
    for n_in, n_out in [(1024, 1000), (4096, 4096)]:
        ops = 2 * n_in * n_out
        bytes_moved = n_in * n_out * 4        # FP32 weights read once, batch=1
        ai = ops / bytes_moved
        bound = "memory-bound" if ai < ridge else "compute-bound"
        print(f"  Dense {n_in}x{n_out}: AI = {ai:.2f} ops/byte -> {bound}")
    print("  TODO (students): why does batch size fix this on a GPU but not at the edge?")
