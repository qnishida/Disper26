# Optional plotting diagnostic

romberg_phase_velocity_kernel.py repeats finite-difference kernels through 512 layers.
It was already excluded from the historical regression gate because it takes over 15 minutes.
Run explicitly with python -m pytest tests/diagnostics/romberg_phase_velocity_kernel.py.
The numerical loops and tolerances are unchanged. Normal regression collection is tests/python.
