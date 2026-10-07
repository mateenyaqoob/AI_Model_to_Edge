"""
Lab 4 - The edge-AI lifecycle, end to end:
    train (PyTorch) -> compress (prune + QAT/PTQ) -> export (ONNX) -> deploy (ONNX Runtime)
    -> profile -> compare.

REQUIREMENTS (students' machines / Colab):
    pip install torch torchvision onnx onnxruntime onnxruntime-tools
NOTE TO INSTRUCTOR: this file was syntax-checked but NOT executed in the authoring
environment (no PyTorch available there). Run it once on Colab before class.
Dataset: MNIST is used only so it trains in ~1-2 min on CPU. Encourage students to
repeat the pipeline on CIFAR-10 / their own sensor data.
"""
import time, os, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"

# ---------------------------------------------------------------- 1. data
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tf)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=512)

# ---------------------------------------------------------------- 2. model
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

def evaluate(model):
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl: correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(test_ds)

def train(model, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        print(f"  epoch {ep+1}: test acc = {evaluate(model):.4f}")

def count_nonzero(model): return sum((p != 0).sum().item() for p in model.parameters())
def model_kb(path): return os.path.getsize(path) / 1e3

# ---------------------------------------------------------------- 3. baseline
print("[1] Train FP32 baseline")
base = SmallNet(); train(base)
acc_base = evaluate(base)
torch.save(base.state_dict(), "base.pt")

# ---------------------------------------------------------------- 4. prune + fine-tune
print("\n[2] Global magnitude pruning (50%) + fine-tune")
pruned = copy.deepcopy(base)
params_to_prune = [(m, "weight") for m in pruned.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=0.5)
print(f"  accuracy right after pruning : {evaluate(pruned):.4f}")
train(pruned, epochs=1, lr=5e-4)                       # recover accuracy
for m, n in params_to_prune: prune.remove(m, n)        # make the mask permanent
acc_pruned = evaluate(pruned)
print(f"  non-zero params: {count_nonzero(base):,} -> {count_nonzero(pruned):,}")

# ---------------------------------------------------------------- 5. export to ONNX
print("\n[3] Export to ONNX")
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(base, dummy, "base.onnx", input_names=["x"], output_names=["y"],
                  dynamic_axes={"x": {0: "batch"}}, opset_version=13)
torch.onnx.export(pruned, dummy, "pruned.onnx", input_names=["x"], output_names=["y"],
                  dynamic_axes={"x": {0: "batch"}}, opset_version=13)

# ---------------------------------------------------------------- 6. INT8 PTQ with ONNX Runtime
print("\n[4] Post-training INT8 quantization (ONNX Runtime, static, calibrated)")
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class Calib(CalibrationDataReader):
    def __init__(self, n=200):
        self.data = iter([{"x": test_ds[i][0].unsqueeze(0).numpy()} for i in range(n)])
    def get_next(self): return next(self.data, None)

quantize_static("base.onnx", "base_int8.onnx", Calib(),
                quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8)

# ---------------------------------------------------------------- 7. compare all variants
print("\n[5] Compare accuracy / size / latency")
import onnxruntime as ort
def ort_acc(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); correct = 0
    for x, y in test_dl:
        out = sess.run(None, {"x": x.numpy()})[0]; correct += (out.argmax(1) == y.numpy()).sum()
    return correct / len(test_ds)

def ort_latency(path, runs=300):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, 28, 28).astype(np.float32)
    for _ in range(30): sess.run(None, {"x": x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); sess.run(None, {"x": x}); t.append((time.perf_counter()-t0)*1e3)
    return np.percentile(t, 50), np.percentile(t, 99)

print(f"{'variant':<14}{'acc':>8}{'size kB':>10}{'p50 ms':>9}{'p99 ms':>9}")
for name, path in [("FP32", "base.onnx"), ("Pruned 50%", "pruned.onnx"), ("INT8 (PTQ)", "base_int8.onnx")]:
    p50, p99 = ort_latency(path)
    print(f"{name:<14}{ort_acc(path):8.4f}{model_kb(path):10.1f}{p50:9.3f}{p99:9.3f}")

print("\nQUESTIONS FOR STUDENTS")
print(" 1. Pruned model has ~50% zeros. Is the .onnx file smaller? Is it faster? Why / why not?")
print(" 2. INT8 shrinks the file ~4x. Did latency drop 4x? What determines the real speed-up?")
print(" 3. Re-run quantization with 10 vs 200 vs 2000 calibration samples. What changes?")
