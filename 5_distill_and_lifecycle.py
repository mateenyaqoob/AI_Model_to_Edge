"""
Lab 5 - Knowledge distillation and the compression trade-off (scikit-learn + NumPy only).

A large "teacher" MLP is distilled into a tiny "student" using soft targets at
temperature T (Hinton, Vinyals, Dean 2015). Regime: DATA-SCARCE (150 labels) with a
larger UNLABELLED transfer pool, which is where distillation is most useful.

HONEST CAVEATS (discuss in class):
  * The gain over the hard-label student comes partly from the EXTRA UNLABELLED
    transfer data, not only from the soft targets. Control experiment: give the
    hard-label student the same pool with teacher HARD pseudo-labels.
  * Temperature is a hyper-parameter. Too high a T over-smooths the targets.
  * On an easy, data-rich problem (full digits) the small student already matches
    the teacher and distillation shows no benefit. Try it and see.
"""
import numpy as np, copy, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def logits(m, X):
    a = X
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        a = a @ W + b
        if i < len(m.coefs_) - 1: a = np.maximum(a, 0)
    return a
def n_params(m): return sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)

X, y = load_digits(return_X_y=True); X = X / 16.0
Xall, Xte, yall, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

def train_soft_student(Xt, soft, T, hidden, seed, epochs=2500, lr=0.5):
    """Full-batch GD on  T^2 * KL(soft_T || student_T).  d/dz = T * (p_T - soft) / n."""
    r = np.random.default_rng(seed); n = len(Xt)
    W1 = r.normal(0, .3, (64, hidden)); b1 = np.zeros(hidden)
    W2 = r.normal(0, .3, (hidden, 10)); b2 = np.zeros(10)
    for _ in range(epochs):
        h = np.maximum(Xt @ W1 + b1, 0); z = h @ W2 + b2
        dz = T * (softmax(z, T) - soft) / n
        dW2 = h.T @ dz; db2 = dz.sum(0)
        dh = dz @ W2.T; dh[h <= 0] = 0
        dW1 = Xt.T @ dh; db1 = dh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return lambda Z: (np.maximum(Z @ W1 + b1, 0) @ W2 + b2).argmax(1)

def run(seed, n_lab=150, hidden=8):
    r = np.random.default_rng(seed)
    idx = r.permutation(len(Xall)); lab, pool = idx[:n_lab], idx[n_lab:]
    Xl, yl = Xall[lab], yall[lab]; Xt = np.vstack([Xl, Xall[pool]])
    teacher = MLPClassifier((128, 128), max_iter=800, random_state=seed, alpha=1e-3).fit(Xl, yl)
    out = {"teacher": teacher.score(Xte, yte)}
    out["student, hard labels (150)"] = MLPClassifier((hidden,), max_iter=800, random_state=seed, alpha=1e-3).fit(Xl, yl).score(Xte, yte)
    # CONTROL: same transfer pool but only teacher HARD pseudo-labels
    pseudo = teacher.predict(Xt)
    out["control: hard pseudo-labels on pool"] = MLPClassifier((hidden,), max_iter=800, random_state=seed, alpha=1e-3).fit(Xt, pseudo).score(Xte, yte)
    for T in [1, 2, 4]:
        pred = train_soft_student(Xt, softmax(logits(teacher, Xt), T), T, hidden, seed)
        out[f"distilled, T={T}"] = (pred(Xte) == yte).mean()
    out["_teacher_params"] = n_params(teacher); out["_student_params"] = hidden * 64 + hidden + hidden * 10 + 10
    return out

if __name__ == "__main__":
    res = [run(s) for s in range(5)]
    print("Distillation (5 seeds, mean +/- std). Test accuracy:")
    for k in [k for k in res[0] if not k.startswith("_")]:
        v = [r[k] for r in res]; print(f"  {k:38s} {np.mean(v):.4f} +/- {np.std(v):.4f}")
    print(f"\nTeacher params {res[0]['_teacher_params']:,}  ->  student params {res[0]['_student_params']:,}"
          f"  ({res[0]['_teacher_params']/res[0]['_student_params']:.0f}x fewer)")

    # ---- post-training weight quantisation sweep on a well-trained teacher ----
    print("\nPost-training weight quantisation (per-tensor, symmetric), full-data teacher:")
    teacher = MLPClassifier((256, 256), max_iter=400, random_state=0).fit(Xall, yall)
    def fq(W, b):
        m = np.abs(W).max(); q = 2 ** (b - 1) - 1; return np.round(W / m * q) / q * m
    for bits in [32, 8, 6, 4, 3, 2]:
        t = copy.deepcopy(teacher)
        if bits < 32: t.coefs_ = [fq(W, bits) for W in t.coefs_]
        print(f"  {bits:2d}-bit: acc={t.score(Xte, yte):.4f}  weights={n_params(t) * bits / 8 / 1e3:7.1f} kB")
    print("\nDISCUSS: why does accuracy hold at 8-4 bits then fall off a cliff at 2 bits?")
