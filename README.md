# Edge Computing Labs: Bridging AI/ML to the Edge

These codes are for supporting Lectures 11 and 12. The 1, 2, 3 and 5 were executed end to end during preparation and produced the outputs quoted in the slides. 

## Setup
```
pip install numpy scikit-learn onnx onnxruntime matplotlib
# Lab 4 only:
pip install torch torchvision
```
Python 3.10+. No GPU needed. Everything runs on a laptop CPU in under a couple of minutes.

| Lab | File | Needs | Time | What it teaches |
|---|---|---|---|---|
| 1 | `1_budgets.py` | NumPy | 20 min | Params, MACs, weight/activation memory, device fit, roofline arithmetic |
| 2 | `2_quantization.py` | NumPy | 25 min | Affine INT8 from scratch: outliers, per-channel, INT32 accumulate |
| 3 | `3_pruning_and_benchmark.py` | NumPy, ONNX, ONNX Runtime | 30 min | Pruning vs speed; correct benchmarking; tail latency under contention |
| 4 | `4_pytorch_to_edge.py` | PyTorch + ONNX Runtime | 45 min | Full lifecycle: train, prune, export, INT8, compare. **Untested here.** |
| 5 | `5_distill_and_lifecycle.py` | scikit-learn | 20 min | Distillation with a control; bit-width cliff |

## Expected results (from the authoring run; yours will differ in the last digits)
- **Lab 1:** separable convs cut MACs 8.2x and params 8.1x. Analytic check: ratio = 1/(1/C_out + 1/k^2).
- **Lab 2:** 8-bit Gaussian SQNR about 39.6 dB; one outlier drops it to about 19 dB; percentile clipping restores about 40 dB; per-channel beats per-tensor (43.2 vs 34.6 dB).
- **Lab 3:** at 50% sparsity structured pruning gives a real 2x MAC cut; ONNX Runtime INT8 was about 4x smaller and about 3.6x faster on one CPU thread (a large dense MLP). Under 4 busy background threads the p99 latency rose from 0.4 ms to about 63 ms while the median barely moved.
- **Lab 5:** in a data-scarce setting distilled students (about 91%) beat a hard-label student (about 88%), but a **control** (teacher hard pseudo-labels on the same pool, about 92%) matches or beats distillation. Temperature 4 fails. Bit-width: flat from 32 to 4 bits, cliff at 2 bits.

## Reproducibility notes
- Timings depend on CPU, thread count, thermal state and background load. Report your own. Never compare timings across machines.
- Seeds are fixed but scikit-learn and ONNX Runtime can vary slightly across versions.
- Lab 3's tail-latency demo spawns busy-loop threads on purpose; it will make your machine briefly sluggish.

## Known limitations (be honest with students)
- Labs 2 and 5 use small models and small data (sklearn digits). Conclusions about *mechanisms* transfer; *numbers* do not.
- Lab 5's distillation result is a single small setting. It shows attribution must be tested; it does not show soft targets never help.
- Lab 1's device table (flash, SRAM, MACs/s) is order-of-magnitude for teaching. Have students replace it with real datasheet values.
- The roofline device (10 GOPS, 4 GB/s) is illustrative, not a specific chip.
