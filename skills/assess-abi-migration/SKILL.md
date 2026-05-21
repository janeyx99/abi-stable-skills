---
name: assess-abi-migration
description: Audit a PyTorch C++/CUDA extension repo for ABI-stable migration readiness. Use when the user wants to know whether their library can be made ABI stable with libtorch, what work is required, and where coverage gaps in the stable ABI exist. Produces an ordered migration plan as both a JSON file and a markdown summary.
allowed-tools: Read, Bash, Glob, Grep
---

# Assess ABI Migration

Inspect a third-party PyTorch extension repo (e.g., flash-attention, vllm, a custom CUDA op library) and determine:

1. Whether the project is migrateable to the stable ABI for the PyTorch version installed.
2. What concrete steps are required, in dependency order.
3. Where the stable ABI has **missing coverage** for this project — APIs the project uses that have no `torch::stable` or `torch::headeronly` equivalent yet.

This skill does not modify code. It produces a machine-readable plan that `orchestrate-stable-abi-migration` consumes.

## Prerequisites

- The user's extension repo (the migration target) is checked out locally.
- PyTorch 2.10+ is installed in the active environment (this is the `build_torch_version`):
  ```bash
  python3 -c "import torch; print(torch.__version__)"
  ```
  If below 2.10, stop and tell the user to upgrade — the stable ABI surface is much more complete in 2.10+.

## Inputs

- `repo_root`: absolute path to the extension repo root.

- `build_torch_version`: the PyTorch version the extension will be **compiled against**. Determines which headers are present during the build and which symbols the assess step can grep for. **Should match the currently installed torch** (`python3 -c 'import torch; print(torch.__version__)'`). If not provided, default to the installed version and confirm with the user, e.g.:

  > "I'll assess against the installed torch `<x.y.z>` as the build-time version. Use a different version? (If so, install it in the active env first — the assess step greps real headers.)"

  Constraint: must equal the installed torch version. If they differ, stop and ask the user to either install the requested version or accept the installed one.

- `target_torch_version`: the **minimum runtime** PyTorch version the resulting wheel should support — gets encoded as `TORCH_TARGET_VERSION` in the build. Lower = wider compatibility (one wheel covers more torch releases); higher = more stable APIs available. **If not provided, ask** before continuing:

  > "What's the lowest PyTorch version you want this wheel to support at runtime? Common answers: `2.10` (widest), `2.11` (standard), or `<build_torch_version>` (match the build, narrowest support window). This becomes `TORCH_TARGET_VERSION` and determines what counts as a coverage gap."

  Constraints:
  - ≥ 2.9 (older versions don't have a usable stable surface).
  - ≤ `build_torch_version` (can't promise runtime support for a torch newer than what we compiled against).

### Why two versions

- The **build** version determines what's available in headers right now, on this machine, for the migration to use.
- The **target** version determines what's available in headers on **every other user's machine** when they install the resulting wheel. It's the lowest common denominator the binary must work with at runtime.

A coverage gap is "API exists in build, doesn't exist at target" — those are APIs you literally cannot use even though they're sitting in the headers, because the wheel must also run on older torch installations. Step 4 checks for this by greping the build headers and then verifying each found symbol's `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_0` guard against the target.

## Discovery steps

### 1. Identify the build system

Look for, in priority order:
- `setup.py` with `CUDAExtension` / `CppExtension` imported from `torch.utils`.
- `CMakeLists.txt` referencing `Torch` / `find_package(Torch)`.
- `pyproject.toml` with a build backend.

Record which is present.

### 2. Find C++/CUDA source files in scope

Ignore test files and any files (e.g., in an `experimental` directory) that are not part of the shipped library. Look for files matching `*.cpp`, `*.cc`, `*.cu`, `*.cuh`, usually in a `csrc` directory, that include any of the following headers:
- `<torch/extension.h>`
- `<ATen/...>`
- `<c10/...>`
- `<torch/library.h>`
- `<pybind11/...>`

### 3. Detect migration concerns per file

For each source file, record which of the following apply (these drive sub-skill dispatch):

| Concern | Sub-skill | Detection signal |
|---|---|---|
| Pybind bindings | `pybind-to-torch-library` | `PYBIND11_MODULE`, `m.def("name", ...)` outside `TORCH_LIBRARY`, `<pybind11/...>` includes |
| C++ meta functions | `migrate-meta-fns-to-python` | `TORCH_LIBRARY_IMPL(..., Meta, m)`, registrations with dispatch key `Meta` |
| C++ autograd impls | `migrate-autograd-fns-to-python` | `TORCH_LIBRARY_IMPL(..., Autograd, m)` / `..., AutogradCUDA, m)`, C++ `torch::autograd::Function` subclasses registered for the op |
| Unstable API usage | `migrate-to-stable-apis` | Any `at::Tensor`, `TORCH_CHECK`, `c10::cuda::CUDAGuard`, `at::empty`, `TORCH_LIBRARY_IMPL`, etc. |

The `split-stable-unstable` subskill is dispatched separately based on the project size, specifically the **count** of source files that require migration (see Step 7 below).

### 4. Detect stable-ABI coverage gaps

Coverage gaps are APIs the project uses that have **no equivalent in the installed PyTorch's stable surface**. The stable surface changes between PyTorch releases, so this check must be against the actually-installed `torch`, not a hardcoded list.

**Resolve the stable headers directory:**

```bash
TORCH_STABLE_DIR=$(python3 -c "
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'csrc', 'stable'))
")
ls "$TORCH_STABLE_DIR"
# Expected: ops.h, tensor.h, library.h, device.h, accelerator.h, c/, ...
```

Also locate `torch/headeronly/`:

```bash
TORCH_HO_DIR=$(python3 -c "
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'headeronly'))
")
```

**For each candidate gap**, grep the stable headers for the expected replacement symbol. Examples:

| Unstable API in user code | Expected stable symbol | Verification |
|---|---|---|
| `at::empty_strided(...)` | `new_empty_strided` or `empty_strided` | `grep -E 'new_empty_strided\|^.*empty_strided\b' "$TORCH_STABLE_DIR"/*.h "$TORCH_STABLE_DIR"/c/*.h` |
| `tensor.layout() == at::Layout::Strided` | `Layout` enum or a `layout()` method on `torch::stable::Tensor` | `grep -E 'enum.*Layout\|::Layout\b\|::layout\(' "$TORCH_HO_DIR"/**/*.h "$TORCH_STABLE_DIR"/*.h` |
| `at::native::<op>` | `<op>` in `ops.h` | `grep -E '\b<op>\b' "$TORCH_STABLE_DIR"/ops.h` |
| `at::parallel_for(...)` | `torch::stable::parallel_for` in `ops.h` (backed by `torch_parallel_for` C shim) | `grep -n 'parallel_for' "$TORCH_STABLE_DIR"/ops.h "$TORCH_STABLE_DIR"/c/shim.h` |
| `at::TensorIterator` | n/a — no current stable equivalent | confirm absence: `grep -i tensoriterator "$TORCH_STABLE_DIR"/*.h "$TORCH_HO_DIR"/**/*.h` |
| `torch::autograd::Function` | n/a in C++ — move to Python | not a header check; route to `migrate-autograd-fns-to-python` instead |

> **Important:** This table is illustrative and reflects the stable surface at the time of writing. The stable ABI is actively growing — never trust this table over the actual `grep` against the installed headers. Symbols that were "absent" in one PyTorch release often appear in the next.

A symbol is **available at build** if it's declared in any header under `$TORCH_STABLE_DIR` or `$TORCH_HO_DIR`. The C shims under `$TORCH_STABLE_DIR/c/` and `<torch_include>/torch/csrc/inductor/aoti_torch/c/` also count.

### Resolving the minimum target version for each symbol

Two cases, each handled differently:

**Case 1: C shim functions** (anything declared in `torch/csrc/stable/c/shim.h` or `torch/csrc/inductor/aoti_torch/c/*.h`). Use the authoritative version map:

```
torch/csrc/stable/c/shim_function_versions.txt
```

Format: one `function_name: TORCH_VERSION_X_Y_Z` line per shim. Per the file's own header: **"If a function is not in this file, it was available before 2.10.0."** Since our minimum supported target is 2.9, that means unlisted = available at every target we care about.

Lookup logic for a C shim symbol `<sym>`:
- Listed as `<sym>: TORCH_VERSION_X_Y_Z` → minimum version is `X.Y.Z`.
- Not listed → available at all supported targets (treat as no version constraint).

The file does **not** ship in the installed torch's `include/` tree, so to access the file:
- If `<pytorch_src>/torch/csrc/stable/c/shim_function_versions.txt` is available locally, read it directly.
- Otherwise fetch from GitHub pinned to the build torch's release tag:
  `https://raw.githubusercontent.com/pytorch/pytorch/v<build_torch_version>/torch/csrc/stable/c/shim_function_versions.txt`

**Case 2: C++ wrappers** (anything in `torch/csrc/stable/*.h` outside `c/`, or `torch/headeronly/`). No authoritative version map; use grep around the symbol declaration as an estimate:

```bash
grep -B 5 -A 2 "<symbol>" "$TORCH_STABLE_DIR"/*.h
# Look for the enclosing `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_Z`
```

Three outcomes:

- **Clear guard found** with `X.Y.Z > target_torch_version` → coverage gap; `min_target_to_close: "X.Y.Z"`.
- **Clear guard found** with `X.Y.Z ≤ target_torch_version` → available at target; not a gap.
- **No guard found, or guards are nested/combined and grep is ambiguous** → cannot determine the minimum version with certainty. Default to:
  - Available at `build_torch_version` (it's in the build headers, so the build works).
  - "Maybe supported" at older targets. Record `min_target_to_close: "<build_torch_version>"` with `reason: "version guard inconclusive; available at build, may not be at older runtimes"`. Surface in chat summary so the user can spot-check or accept the conservative answer.

The grep approach is rough — it can miss nested `#if`s, `||`/`&&` combinations, and macro indirection. For C shims that's not a problem (the txt file is authoritative). For C++ wrappers, accept some false-positive gaps as the price of a fast assess; the user can override via the `reason` field or run `migrate-to-stable-apis` and let the build confirm.

#### For each gap, evaluate whether bumping the target would close it

Three categories:

1. **Gap closeable by raising target.** Symbol exists in build headers under a `TORCH_FEATURE_VERSION` guard with `X.Y.Z > target_torch_version`. Bumping target to `X.Y.Z` closes the gap. Record `min_target_to_close: "X.Y.Z"`.

2. **Gap not closeable with current offering.** Symbol is absent from build headers for the installed torch. Options to try a documented workaround, rerunning the skill on a newer torch version, or filing an issue for the API to be added. Record `min_target_to_close: null` with a `reason` field, e.g., `"absent from torch::stable through build version <build_torch_version>"`.

3. **Gap structural, no stable equivalent expected.** E.g., `at::TensorIterator`, `torch::autograd::Function` — these are libtorch-internal designs without a planned stable surface (autograd specifically moves to Python via `migrate-autograd-fns-to-python`). Record `min_target_to_close: "n/a"` with `reason: "no planned stable equivalent"`. Be conservative classifying things here — if you're not sure whether an API is structurally absent vs. just not added yet, prefer category 2 (`null` with a `reason`).

Aggregate the closeable gaps for the chat summary. If raising the target by one minor version would close N of M gaps, surface that as actionable guidance: the user can decide whether the narrower runtime support window is worth shedding those workarounds.

**Heuristic for naming.** Most legacy ATen ops have a same-named stable counterpart in `torch::stable::`. So for `at::<op>(...)`, the first thing to check is `grep -E '\b<op>\b' "$TORCH_STABLE_DIR/ops.h"`. If that fails, also try variants prefixed with `new_` (PyTorch's convention for factory ops that take a reference tensor — `new_empty`, `new_zeros`).

**Record the target PyTorch version with each gap.** Use `target_torch_version` from the skill's inputs (not the installed version, if they differ). Future readers need to know "this gap was true at the target the assessment was scoped to" — the same API may be added in a later release that the user might be willing to raise the target to.

For each gap found, record in the plan: file, line, old API, the expected stable symbol that was checked for, the torch version checked against, suggested workaround if one is documented (see [api-mapping.md § Strided Tensors](../migrate-to-stable-apis/references/api-mapping.md) and [common-issues.md](../migrate-to-stable-apis/references/common-issues.md)), or `BLOCKED` if no workaround.

The plan's `coverage_gaps` entry should look like:

```json
{
  "path": "src/bar.cpp",
  "line": 142,
  "old_api": "at::empty_strided",
  "checked_symbol": "torch::stable::new_empty_strided",
  "checked_against_target_version": "2.11",
  "min_target_to_close": "2.13",   // or null / "n/a" — see Step 4 categories above
  "reason": null,                  // populated when min_target_to_close is null or "n/a"
  "workaround": "new_empty + transpose",
  "blocked": false
}
```

### 5. Capture build flags

Note whether the project already uses:
- `-DUSE_CUDA` (needed for `aoti_torch_get_current_cuda_stream`).
- `-DTORCH_TARGET_VERSION=...` (pins the target PyTorch version and bans unstable headers).
- `-DPy_LIMITED_API=0x...` (enables single-wheel-across-Python-versions). Also check `py_limited_api=True` in `cpp_extension.*Extension(...)` and `options={"bdist_wheel": {"py_limited_api": "..."}}` in `setup()`.

### 6. Detect existing Python fake/meta registrations

Grep for `@torch.library.register_fake(` and `torch.library.impl(..., "Meta")`. Record any ops that **already** have a Python fake — these don't need `migrate-meta-fns-to-python` even if a corresponding C++ Meta impl exists (the C++ one becomes redundant and gets deleted).

### 7. Count translation units → decide on `split-stable-unstable`

Count the C++/CUDA source files in scope from Step 2. Form a recommendation:

- **≥ 10 files** → recommend including `split-stable-unstable` (run migration into a parallel stable target).
- **< 10 files** → recommend skipping it; `migrate-to-stable-apis` rewrites in place.

The threshold isn't strict — bump it down if the project is unusually complex (custom build system, many shared headers, intricate setup.py), or bump it up for trivially structured projects. The right framing is: "is the per-file move + parallel-target overhead worth it vs. just rewriting in place?"

**Confirm with the user before recording the decision.** Surface the count, the recommendation, and the reasoning, e.g.:

> "Found 14 translation units to migrate. Recommend including `split-stable-unstable` (set up a parallel stable build target, migrate files into it one at a time). Alternative: skip it and rewrite in place — bulkier for a project this size, but faster. Which do you want?"

Only after the user confirms (or overrides), record the decision in the plan's `dispatch_order`. Save the user's choice + reasoning to the plan so future re-runs don't re-prompt unnecessarily.

### 8. Detect test setup

Look for `pytest.ini`, `tests/`, `test_*.py`, `pyproject.toml`'s `[tool.pytest.ini_options]`, or a `Makefile` test target. Record the `test_cmd`. Sometimes testing the whole library is too expansive, in that case identify a few relevant smoke tests. If no tests exist at all, flag that as a `blocker` — a migration without a baseline test signal is too risky to auto-execute.

### Background: versioned shims (for explaining coverage gaps)

When this skill flags a "coverage gap," the underlying mechanic is usually shim versioning. The C shims in `torch/csrc/inductor/aoti_torch/c/shim.h` (and the wrappers in `torch/csrc/stable/`) are guarded:

```cpp
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
inline torch::stable::Tensor from_blob(...) { ... }
#endif
```

So an API can simultaneously "exist in PyTorch" and "not be available at this `TORCH_TARGET_VERSION`." When reporting a gap, mention which torch version first exposed the missing API if you can determine it — that lets the user weigh raising `TORCH_TARGET_VERSION` against keeping wider compatibility.

The C-shim APIs (under `torch/csrc/inductor/aoti_torch/c/` and `torch/csrc/stable/c/`) carry a **minimum 2-year backward-compatibility window** from the release that introduced them. The higher-level wrappers in `torch/csrc/stable/` and `torch/headeronly/` follow PyTorch's standard FC/BC policy. Cite this when users ask "how long will this binary keep working?"

## Outputs

### File: `<repo_root>/.abi-migration-plan.json`

Source of truth for the migration. Schema:

```json
{
  "version": 1,
  "generated_at": "<ISO 8601>",
  "repo_root": "<absolute path>",
  "build_torch_version": "<x.y.z>",      // version compiled against; matches installed torch
  "target_torch_version": "<x.y.z>",       // TORCH_TARGET_VERSION; ≤ build_torch_version
  "build_system": "setup.py | cmake | pyproject",
  "files": [
    {
      "path": "src/foo.cu",
      "concerns": ["unstable-api", "pybind"],
      "old_apis": ["at::Tensor", "TORCH_CHECK", "PYBIND11_MODULE"]
    }
  ],
  "build_flags": {
    "use_cuda": false,
    "torch_target_version": null,
    "py_limited_api": null
  },
  "existing_python_fakes": ["mylib::op_with_fake"],
  "coverage_gaps": [
    {
      "path": "src/bar.cpp",
      "line": 142,
      "old_api": "at::empty_strided",
      "checked_symbol": "torch::stable::new_empty_strided",
      "checked_against_target_version": "2.11",
      "min_target_to_close": "2.13",
      "reason": null,
      "workaround": "new_empty + transpose",
      "blocked": false
    }
  ],
  "gap_closure_summary": {
    "total": 5,
    "closeable_by_bumping_target": {
      "2.12": 1,
      "2.13": 3
    },
    "structural": 1
  },
  "dispatch_order": [
    "pybind-to-torch-library",
    "migrate-meta-fns-to-python",
    "migrate-autograd-fns-to-python",
    "split-stable-unstable",
    "migrate-to-stable-apis"
  ],
  "blockers": []
}
```

Include only the sub-skills in `dispatch_order` whose concerns appear in the file list. Drop ones that don't apply.

### Chat summary

Print a short markdown summary to the user:

- Build torch version: `<build_torch_version>` (must match installed).
- Target torch version (`TORCH_TARGET_VERSION`): `<target_torch_version>`. Resulting wheel will require this version or newer at runtime.
- Migrateable: yes / yes-with-workarounds / no (with reason).
- Files in scope: count.
- Sub-skills that are recommended (in order).
- Coverage gaps requiring workarounds, with closure analysis:
  - "Bumping target to 2.12 would close 1 of 5 gaps."
  - "Bumping target to 2.13 would close 4 of 5 gaps."
  - "1 gap is structural (`at::TensorIterator`) and won't close at any runtime version."
  This lets the user weigh "narrower runtime support" against "fewer workarounds to write."
- Hard blockers, if any.

## Handoff

If the user wants to proceed, invoke `orchestrate-stable-abi-migration` and pass it the path to the plan JSON.
