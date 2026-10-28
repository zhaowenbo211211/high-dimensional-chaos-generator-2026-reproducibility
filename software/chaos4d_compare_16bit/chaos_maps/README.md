# chaos_maps

This folder contains the chaotic-system implementations used by the project.
All maps expose the same public shape:

- internal state is stored as a NumPy `int64` vector of 16-bit words
- `step()` advances one iteration and returns the normalized float view
- `iterate(num_steps, discard_transient)` returns a trajectory with shape `(T, n)`
- `get_state_u16()` and `get_state_float()` expose the current state without mutating it

## File Relationships

`base.py` defines the common inheritance root:

- `ChaosMapNDBase`: n-dimensional state handling, reset logic, trajectory generation, and u16/float views.
- `ChaosMap4DBase`: 4D specialization using the project common initial state `(0.1, 0.2, 0.3, 0.4)`.

The concrete files then split into two groups.

## Proposed System Family

- `proposed_base.py`
  - `ProposedNDBaseMap`: n-dimensional delayed base recurrence.
  - `Proposed4DBaseMap`: 4D wrapper preserving legacy parameter and delay names.
  - depends on `utils.r1ly.synthesize_r1_ly_params()` for validated parameter synthesis.
- `proposed_enhanced.py`
  - `ProposedNDEnhancedMap`: computes one base update, packs all 16-bit states into a `16n`-bit word, applies cyclic rotate-left enhancement, then unpacks.
  - `Proposed4DEnhancedMap`: 4D wrapper with compatibility helpers `_pack_state_to_u64`, `_rol64`, and `_unpack_u64_to_state`.
  - inherits the base recurrence from `ProposedNDBaseMap`.

The proposed enhanced map is the representative source used by the system-level
PRNG pipeline. That PRNG pipeline is separate from this folder: map enhancement
rotates the state after a base update, while PRNG output generation later forms
64-bit blocks from the four 16-bit state outputs.

## Literature Comparison Maps

Each comparison map inherits `ChaosMap4DBase` and implements one `step()` method.

- `ndcs_4d.py`: integer-coefficient natural mod-1 4D digital chaotic system.
- `uruk_4d_dcm.py`: tangent-heavy URUK-4D-DCM; uses safe tangent evaluation, finite cleanup, mod-1 wrapping, then u16 quantization.
- `mhm_4d.py`: memristive/trigonometric system; uses safe sine/cosine helpers.
- `idhnn_4d.py`: tanh/neural-coupled system with reciprocal-driven sine; uses safe division and safe nonlinear helpers.
- `mprc_4d.py`: multi-trigonometric cross-coupled system; evaluates in float form before mod-1 wrapping.
- `mslc_4d.py`: non-autonomous system with explicit `sin(n)` time drive; maintains `time_index` across steps.
- `hccdm_4d.py`: high-complexity system with Tent, `sinh`, and `exp` terms; uses the most defensive finite-value cleanup.

## Shared Numeric Semantics

The folder is intentionally conservative about numeric behavior:

- all public map states are aligned to the shared 16-bit domain `[0, 65535]`
- float-facing equations are wrapped back to `[0, 1)` before quantization
- unsafe nonlinear terms go through `utils.safe_math`
- conversion rules live in `utils.fixed16`
- initialization rules live in `utils.initializers`

This makes cross-map metrics compare digital behavior under one common
implementation model rather than mixing unrelated state precisions.

## Import Surface

`__init__.py` re-exports the map classes used by experiments and tests:

- `Proposed4DBaseMap`, `Proposed4DEnhancedMap`
- `ProposedNDBaseMap`, `ProposedNDEnhancedMap`
- `NDCS4DMap`, `URUK4DDCMMap`, `MHM4DMap`, `IDHNN4DMap`
- `MPRC4DMap`, `MSLC4DMap`, `HCCDM4DMap`

Use this folder when changing update equations, parameter handling, fixed-point
state semantics, or the proposed-map enhancement step.
