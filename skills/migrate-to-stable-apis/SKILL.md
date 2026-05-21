---
name: migrate-to-stable-apis
description: Rewrite the legacy PyTorch C++/CUDA API calls in a third-party extension to their `torch::stable` / `torch::headeronly` / `aoti_torch` equivalents, swap registration macros to the `STABLE_TORCH_LIBRARY_*` family with `TORCH_BOX`, and update build flags to enforce ABI safety via `TORCH_TARGET_VERSION`. Use after `pybind-to-torch-library`, `migrate-meta-fns-to-python`, and `migrate-autograd-fns-to-python` have run. If `split-stable-unstable` set up a parallel stable target, this skill moves files into it one at a time; otherwise it rewrites files in place.
allowed-tools: Read, Bash, Edit, Write, Glob, Grep
---

# Migrate to Stable APIs

This is the bulk-rewrite step. By the time this skill runs, the project should:

- Use `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` for op registration (pybind removed).
- Have any C++ Meta impls moved to Python `register_fake`.
- Have any C++ Autograd impls moved to Python `register_autograd`.
- Either have a parallel stable build target set up by `split-stable-unstable`, or be a small enough project (< 10 translation units) to rewrite in place.

## Operating modes

This skill runs in one of two modes depending on whether `split-stable-unstable` ran:

**Mode A — Move-and-rewrite (large project, parallel target exists).** For each legacy source file:
1. Copy from `<legacy_pkg>/csrc/` to `<stable_pkg>/csrc/`.
2. Apply the API rewrites to the copy.
3. Add the new path to the stable target's `sources=` in `setup.py`.
4. Remove the old path from the legacy target's `sources=`.
5. Delete the original from `<legacy_pkg>/csrc/`.
6. Build both targets. Run tests. Advance.

**Mode B — Rewrite in place (small project, no parallel target).** For each legacy source file:
1. Apply the API rewrites in place.
2. Build. Run tests. Advance.

In Mode B, the project's single `setup.py` entry also gets `TORCH_TARGET_VERSION` and `Py_LIMITED_API` added once on the first file, since there's no pre-existing stable target with those flags. In Mode A, the stable target already has them — don't touch the legacy target's flags.

The rest of this doc describes the rewrite work, which is identical between modes. Mode A wraps each file's rewrite in copy/move bookkeeping; Mode B doesn't.

## Prerequisites

- PyTorch 2.11+ (verify: `python3 -c "import torch; print(torch.__version__)"`).
- The full reference docs in this skill's `references/`:
  - [api-mapping.md](references/api-mapping.md) — old API → stable replacement table.
  - [migration-example.md](references/migration-example.md) — concrete before/after.
  - [common-issues.md](references/common-issues.md) — known build/runtime pitfalls.
  - [stack-and-boxing.md](references/stack-and-boxing.md) — `TORCH_BOX` internals, `StableIValue` conversions, `torch_call_dispatcher`. Only needed for advanced cases.

## Inputs

- `plan_path`: path to the migration plan JSON.
- Mode determined by presence of `<stable_pkg>/` (set up by `split-stable-unstable`):
  - **Mode A** if it exists. The list of legacy files comes from the legacy target's `sources=` in `setup.py`.
  - **Mode B** otherwise. The list of files comes from the plan's `unstable-api` concern entries.

## Workflow

Process one file at a time. After each file is fully migrated and the project builds + tests pass, advance to the next. Resist the urge to batch — file-at-a-time keeps failures localized and bisection trivial.

### 0. (Mode A only) Move the file into the stable target

For the next legacy file `<legacy_pkg>/csrc/<file>`:

1. Copy it to `<stable_pkg>/csrc/<file>`. Don't modify yet.
2. In `setup.py`, append `"<stable_pkg>/csrc/<file>"` to the stable target's `sources=`, and remove `"<legacy_pkg>/csrc/<file>"` from the legacy target's `sources=`.
3. The original under `<legacy_pkg>/csrc/<file>` will be deleted after the rewrite succeeds (Step 6 below).

This step is skipped entirely in Mode B.

### 1. Update includes

Remove legacy includes and replace with the stable subset.

```cpp
// REMOVE:
// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <torch/library.h>

// ADD (include only what you need):
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

// For CUDA stream access:
#include <torch/csrc/inductor/aoti_torch/c/shim.h>  // or torch/csrc/stable/c/shim.h on newer versions
#include <cuda_runtime.h>
```

Allowed header roots (anything outside these will fail to compile under `TORCH_TARGET_VERSION`):

- `torch/csrc/stable/`
- `torch/headeronly/`
- `torch/csrc/inductor/aoti_torch/`

### 2. Apply API replacements

Walk each file and apply the substitutions from [references/api-mapping.md](references/api-mapping.md). The most frequent ones:

| Old | Stable |
|---|---|
| `at::Tensor` / `torch::Tensor` | `torch::stable::Tensor` |
| `TORCH_CHECK(...)` | `STD_TORCH_CHECK(...)` (arguments unchanged) |
| `at::empty({...}, options)` | `torch::stable::new_empty(ref, {sizes}, dtype)` |
| `at::zeros({...}, options)` | `torch::stable::new_zeros(ref, {sizes}, dtype)` |
| `at::empty_like(t)` | `torch::stable::empty_like(t)` |
| `t.contiguous()` | `torch::stable::contiguous(t)` |
| `t.data_ptr<T>()` (read) | `t.const_data_ptr<T>()` |
| `t.data_ptr<T>()` (write) | `t.mutable_data_ptr<T>()` |
| `t.sizes()` / `t.strides()` | `t.size(i)` / `t.stride(i)` (per-element) |
| `at::kFloat` | `torch::headeronly::ScalarType::Float` |
| `at::kCUDA` | `torch::headeronly::kCUDA` |
| `tensor.device() == kCUDA` | `tensor.device().type() == torch::headeronly::kCUDA` |
| `at::pad(x, ...)` / `x.pad(...)` | `torch::stable::pad(x, ...)` (always free-function form; Tensor methods → ATen variant) |
| `c10::cuda::CUDAGuard guard(t.device())` | `torch::stable::accelerator::DeviceGuard guard(t.get_device_index())` |
| `at::cuda::getCurrentCUDAStream()` | `aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr)` (see common-issues) |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `STD_CUDA_KERNEL_LAUNCH_CHECK()` |

See [api-mapping.md](references/api-mapping.md) for the full table, including strided tensor creation, layout checks, and scalar type enums.

### 3. Swap registration macros to STABLE_*

After this step, op registrations move from `TORCH_LIBRARY` (set up by `pybind-to-torch-library`) to the stable form:

```cpp
// Before:
TORCH_LIBRARY(mylib, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("mymuladd", &mymuladd_cuda);
}

// After:
STABLE_TORCH_LIBRARY(mylib, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
}
```

Rules:

- `TORCH_LIBRARY` → `STABLE_TORCH_LIBRARY` (schema definitions).
- `TORCH_LIBRARY_FRAGMENT` → `STABLE_TORCH_LIBRARY_FRAGMENT`.
- `TORCH_LIBRARY_IMPL` → `STABLE_TORCH_LIBRARY_IMPL`.
- Wrap each kernel function pointer in `TORCH_BOX(&fn)` inside `m.impl(...)`.
- Do **not** wrap the op name in `TORCH_SELECTIVE_NAME`.
- Do **not** wrap the function pointer in `TORCH_FN`.
- Do **not** manually write boxed kernels — `TORCH_BOX` generates the boxed wrapper that adapts `(stack, num_args, num_outputs)` to your typed function signature using `to<T>` / `from<T>` over `StableIValue`.

In Mode A, change the library namespace to the stable package name (e.g., `STABLE_TORCH_LIBRARY(extension_cpp_stable, m)` instead of `STABLE_TORCH_LIBRARY(extension_cpp, m)`). This prevents the dispatcher from seeing duplicate registrations while the legacy target is still live, and matches the library name the stable package's `__init__.py` loader expects.

### 4. Update build flags

In **Mode A**, the stable target already has `TORCH_TARGET_VERSION` and `Py_LIMITED_API` set up by `split-stable-unstable`. Don't touch the legacy target's flags. Skip this step.

In **Mode B** (rewriting in place, no parallel target), add `TORCH_TARGET_VERSION` to the single existing build target on the first file you migrate. The hex encoding is `0xMMmm0000_00000000` — e.g., `0x020b000000000000` for 2.11, `0x020a000000000000` for 2.10.

```python
extra_compile_args={
    "cxx": [
        "-DTORCH_TARGET_VERSION=0x020b000000000000",
        "-DPy_LIMITED_API=0x03090000",  # already added in pybind-to-torch-library
    ],
    "nvcc": [
        "-DTORCH_TARGET_VERSION=0x020b000000000000",
        "-DUSE_CUDA",
    ],
},
```

`TORCH_TARGET_VERSION` serves two purposes:

1. **Bans unstable headers.** Any `#include <ATen/...>`, `<c10/...>`, `<torch/extension.h>`, etc. in a translation unit compiled with this flag fails with:
   > "This file should not be included when either `TORCH_STABLE_ONLY` or `TORCH_TARGET_VERSION` is defined."
   This is your compile-time guarantee that the binary is ABI-stable.
2. **Selects the minimum runtime PyTorch version.** Shims in `shim.h` are guarded by `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_Z`. APIs added after the target version are unavailable. The compiled binary will load against any PyTorch ≥ the target version.

Constraint: `2.9 ≤ TORCH_TARGET_VERSION ≤ <libtorch version at build time>`. Choose the lowest version the project is willing to support; lower = wider compatibility, higher = more APIs.

### 5. Build and iterate

Run `build_cmd`. In Mode A, this builds both the legacy and stable targets — both must succeed. Most failures fall into two buckets:

- **"Header not allowed" errors** — a legacy include slipped through. Find and replace it. If the file genuinely needs an API with no stable equivalent, see [references/common-issues.md](references/common-issues.md) for workarounds, or stop and flag a blocker.
- **Substitution misses** — an unstable API call wasn't replaced. Apply the mapping.

Common pitfalls are documented in [references/common-issues.md](references/common-issues.md): missing `-DUSE_CUDA`, undeclared `aoti_torch_get_current_cuda_stream`, layout check rewrites, and the `empty_strided` workaround.

### 6. Test, then delete the legacy original (Mode A)

Run the project's existing test suite (`test_cmd` from the plan). The suite should pass against the migrated binary.

In **Mode A**, once tests pass for this file:

1. Delete the original file at `<legacy_pkg>/csrc/<file>`. It's been replaced by the migrated copy under `<stable_pkg>/csrc/<file>`.
2. If this op had Python-side registrations under the legacy package (e.g., `_meta.py`, `_autograd.py` from earlier skills), update them to point at the stable namespace (`torch.ops.<stable_pkg>::<op>`) and uncomment their imports in `<stable_pkg>/__init__.py`. The legacy-side registrations can be removed.

In **Mode B**, the file was rewritten in place — no deletion or move needed. Advance to the next file.

### 7. Advance

Pick the next legacy source file from `setup.py`'s legacy `sources=` (Mode A) or the plan's `unstable-api` list (Mode B). Repeat from Step 0.

Stop when:
- (Mode A) The legacy target's `sources=` is empty. Hand off to `split-stable-unstable`'s completion section for final cleanup (removing the legacy entry, possibly renaming the stable package back to the original name).
- (Mode B) All files in the plan are migrated.

## Critical rules

These are mistakes that AI-assisted migrations commonly make — follow strictly:

1. **Do NOT use `torch::stable::new_empty_strided`** — it does not exist. To create a strided tensor, create a contiguous tensor with transposed dimensions and call `torch::stable::transpose()`. See [common-issues.md § Strided Tensors](references/common-issues.md).
2. **Do NOT define a dummy `_C` module** accessible from Python. Operators reach Python through `torch.ops.<libname>.<op>`.
3. **Do NOT forward-declare `aoti_torch_get_current_cuda_stream`** — include it from `torch/csrc/inductor/aoti_torch/c/shim.h`.
4. **Do NOT hand-write boxed kernels** — use `TORCH_BOX(&fn)` in `m.impl()`.
5. **Do NOT change switch statements into if/else** as part of the migration. The stable ABI works fine with switch statements; rewriting just adds review noise.
6. **When replacing `TORCH_CHECK` with `STD_TORCH_CHECK`, change only the macro name** — leave the condition and message arguments untouched.
7. **For `DeviceGuard`**, pass `t.get_device_index()` (an integer), **not** `t.device()`. Do not cast to `char` or narrow type. DeviceGuard is RAII — don't move its declaration into an inner scope.
8. **For CUDA streams**, derive the device index from a tensor (`t.get_device_index()`), **not** from `torch::stable::accelerator::getCurrentDeviceIndex()`. Multi-GPU scenarios silently produce the wrong stream otherwise.
9. **Scalar type enums use the headeronly form** — `torch::headeronly::ScalarType::Float` (not `torch::kFloat32`), `::Half` (not `torch::kFloat16`), `::BFloat16`, `::Float8_e4m3fn`, etc.

## Verification

Build with `build_cmd`. Run the project's existing tests with `test_cmd`. Then prove ABI stability empirically:

```bash
# Build against torch X.Y
pip install torch==X.Y
<build_cmd>

# Then install a different torch minor and re-test without rebuilding the extension
pip install torch==X.Z
<test_cmd>
```

If the second `test_cmd` passes, the migration is verified.

## Real-world references

- **Official tutorial:** [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — walks through the same migration end-to-end on a toy `mymuladd` op, with side-by-side ABI-stable vs non-stable tabs.
- **Reference repo:** [`pytorch/extension-cpp`](https://github.com/pytorch/extension-cpp) — contains `extension_cpp/` (legacy) and `extension_cpp_stable/` (stable) implementations of the same op, the canonical worked example.
- **Real-world migration:** FlashAttention 3's [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp) (legacy) vs [`flash_api_stable.cpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api_stable.cpp) (stable). The diff is a concrete production-scale example.
- **Other adoptions:** xformers, torchaudio, torchao, vLLM (in progress).

## Handoff

This is typically the last step in the migration. After this passes:

- The plan's `dispatch_order` is fully `completed`.
- **Mode A:** the legacy target's `sources=` is empty. Return to [split-stable-unstable § Completion](../split-stable-unstable/SKILL.md) to remove the legacy entry from `setup.py`, delete the legacy directory, and (optionally) rename the stable package back to the original name.
- **Mode B:** all files have been rewritten in place. The single existing target now compiles under stable flags.
- The orchestrator's final summary should note: target PyTorch version pinned, single-wheel goal achieved.
