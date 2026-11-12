# Retained leader integration verification

Base: `e6eff17` from the supplied `tidyhebo (2)(2).zip`.
The archive's apparent modified tracked files differ from Git only in line endings;
no scientific source edits from the archive were discarded.

Validation environment: Python 3.12, Torch 2.7.1+cpu, BoTorch 0.14.0,
GPyTorch 1.14, pymoo 0.6.1.5, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3,
pytest 9.0.2. CPU/BLAS threads limited to one for repeatable short tests.

The complete suite passes, including the existing API/GP/GA tests, the new
q=1,2,4,8,16 layered searches, grouped-versus-independent Adam checks,
seed-preserving inference-copy values/coordinate gradients, and real YJ/input
warp fitting and suggestion. Six CUDA-only cases are skipped on this CPU host.
The archived CPU reference front matches to approximately 3e-9 in coordinates
on the fixed small-budget fixture; its regression tolerance is 1e-7.

The standalone full-preset speed probe was executed on CPU for q=4 with a
12-observation Ackley GP, one warmup and one timed search. Both mini-GA and
main GA used all 48 candidates per generation; mini-GA total was 240 candidate
scores, main GA was 20 generations plus final memetic candidates. This is a
functional speed-probe check, not a representative Olympus performance estimate.

No GPU is present here. CUDA correctness tests are provided but actual CUDA
execution and speed of this integrated architecture must be checked on the
user's hardware. The earlier GPU benchmark was for a DIFFERENT no-GA prototype;
its speedups are not claimed for this implementation. Full Olympus quality of
this port has not been recomputed; historical bitwise replay is not promised.

GPyTorch's usual double-precision Cholesky jitter warnings occurred for some
near-coincident candidate points. These warnings are not suppressed.
