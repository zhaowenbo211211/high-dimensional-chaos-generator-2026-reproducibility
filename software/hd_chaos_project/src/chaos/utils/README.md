# chaos.utils — Utility Functions

Auxiliary tools for LaTeX generation and binary sequence export.

## Modules

### latex.py — LaTeX Equation Generation

Generates LaTeX source for the chaotic system equations, supporting both symbolic and numeric forms.

```python
from chaos.utils import make_F, latex_system_like_figure_full
import numpy as np

# Create a callable map F with embedded metadata
F = make_F(4, np.array([271, 273, 275]), np.array([50, 48, 46]), 81)

# Generate LaTeX with actual parameter values
tex = latex_system_like_figure_full(F, with_mod=False)
print(tex)
```

| Function | Description |
|---|---|
| `make_F(n, a_j1, a_jn, b)` | Create callable map F with `._meta` for LaTeX generation |
| `latex_system_like_figure(F)` | Symbolic LaTeX matching paper figure layout |
| `latex_system_like_figure_full(F)` | Numeric LaTeX with actual parameter values |
| `latex_from_F(F, show_values, align_env)` | General-purpose LaTeX from F metadata |

### sequence.py — NIST Binary Sequence

Generates binary test sequences for NIST SP 800-22 statistical testing.

```python
from chaos.utils import generate_binary_sequence

binary_data = generate_binary_sequence(map_instance, length=1_000_000)
with open("nist_sequence.bin", "wb") as f:
    f.write(binary_data)
```

Extracts the lowest 16 bits of $x_n$ at each step, packs them into a byte stream.

---

**Dependencies**: `numpy`, `chaos.core`

**Documentation constraint**: If you add new utility functions or change existing signatures, update this README.
