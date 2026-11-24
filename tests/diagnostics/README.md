# Optional plotting diagnostic

From the repository root, with the `test` and `paper` extras installed, run

```sh
python -m pytest tests/diagnostics/romberg_phase_velocity_kernel.py
```

The diagnostic compares finite-difference kernels through 512 layers and can
take over 15 minutes. Matplotlib writes `kernel_convergence_plot.png` to the
current directory. The regular regression suite runs under `tests/python`.
