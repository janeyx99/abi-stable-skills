---
name: split-stable-unstable
description: Set up a parallel ABI-stable build target alongside the existing legacy extension, so that files can be migrated into it incrementally by `migrate-to-stable-apis`. The end state is that the legacy target is empty and removed, and the project ships a single stable wheel. Use unless the project has fewer than ~10 translation units to migrate (in which case `migrate-to-stable-apis` rewrites in place).
allowed-tools: Read, Bash, Edit, Write, Glob, Grep
---

# Set Up Parallel Stable Build Target

The goal of this skill is to **stand up a new ABI-stable wheel target** that will eventually replace the legacy extension entirely.

By the end of the full migration:

- The new stable target compiles with `-DTORCH_TARGET_VERSION` and `-DPy_LIMITED_API`.
- All C++/CUDA sources have been moved into the stable target.
- The original legacy unstable directory is empty and removed.
- The project ships **one** wheel — the stable one.

This skill creates the scaffolding. The actual per-file moves + API rewrites happen in `migrate-to-stable-apis`, which copies sources from the unstable directory into the new stable directory, applies the API substitutions, and verifies the build.

## When to use

Run this skill if the migration plan has **10 or more** translation units to migrate. The threshold isn't strict; adjust based on project complexity. If the project is smaller, don't create a parallel build and instead change the existing one so that `migrate-to-stable-apis` is set up to rewrite small projects in place.

If you skip this skill, `migrate-to-stable-apis` will:
- Rewrite files in their existing location.
- Add `TORCH_TARGET_VERSION` and `Py_LIMITED_API` to the existing setup.py.
- Produce one wheel directly from the existing layout.

## Why a parallel target

The migration is multi-step and involves API substitutions, build flag changes, and a switch from `TORCH_LIBRARY` to `STABLE_TORCH_LIBRARY`. Doing this in place on a large project means:

- The repo is in a half-migrated, non-building state for the duration.
- Reverting a problematic file means undoing several intertwined changes.
- The legacy build can't be exercised for comparison during migration.

A parallel target avoids all three: the legacy extension keeps working until the migration is complete, and progress is visible as files accumulate on the stable side.

This is the pattern used by [`pytorch/extension-cpp`](https://github.com/pytorch/extension-cpp), which ships `extension_cpp/` (legacy) and `extension_cpp_stable/` (stable) side by side as a reference.

## Inputs

- `plan_path`: path to the migration plan JSON.
- `legacy_pkg`: the existing Python package name (e.g., `extension_cpp`). Discovered from `setup.py` or `pyproject.toml`.
- `stable_pkg` (optional): the new stable package name. Default: `<legacy_pkg>_stable`.

## Setup steps

### 1. Create the stable package directory

Mirror the layout of the legacy package. Typical structure:

```
<repo_root>/
├── <legacy_pkg>/
│   ├── __init__.py
│   ├── ops.py
│   └── csrc/
│       ├── muladd.cu
│       └── ...
├── <stable_pkg>/                  ← NEW
│   ├── __init__.py                ← NEW (skeleton)
│   ├── ops.py                     ← NEW (copy from legacy, will be updated)
│   └── csrc/                      ← NEW (empty; files arrive via migrate-to-stable-apis)
├── setup.py
└── tests/
```

Create empty `csrc/` — files arrive as the next step migrates them.

### 2. Add the stable extension to `setup.py`

Keep the legacy `ext_modules` entry untouched. Add a second entry for the stable target with the right flags. Final shape:

```python
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="extension_cpp",   # project name unchanged
    ext_modules=[
        # Legacy — unchanged. Shrinks as files move out of csrc/.
        cpp_extension.CUDAExtension(
            "extension_cpp._C",
            sources=["extension_cpp/csrc/muladd.cu", "extension_cpp/csrc/other.cu"],
        ),
        # New stable target — starts with no sources and grows.
        cpp_extension.CUDAExtension(
            "extension_cpp_stable._C",
            sources=[],   # populated as migrate-to-stable-apis moves files in
            extra_compile_args={
                "cxx": [
                    "-DTORCH_TARGET_VERSION=0x020b000000000000",
                    "-DPy_LIMITED_API=0x03090000",
                ],
                "nvcc": [
                    "-DTORCH_TARGET_VERSION=0x020b000000000000",
                    "-DUSE_CUDA",
                ],
            },
            py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
```

Pick `TORCH_TARGET_VERSION` based on the plan's `pytorch_version` — typically the lowest the project is willing to support. See [migrate-to-stable-apis](../migrate-to-stable-apis/SKILL.md) for the flag's two purposes.

### 3. Skeleton `__init__.py` for the stable package

The stable package needs a loader. Mirror the legacy loader pattern from [pybind-to-torch-library § 4](../pybind-to-torch-library/SKILL.md):

```python
# <stable_pkg>/__init__.py
import torch
from pathlib import Path

so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

# Python registrations (fake kernels, autograd, etc.) — populate as ops migrate over.
# from . import _meta       # noqa: F401
# from . import _autograd   # noqa: F401

# Re-export the Python-facing API. During migration this can defer to torch.ops directly.
from . import ops  # noqa: F401
```

### 4. Stage Python-side registrations

If `migrate-meta-fns-to-python` or `migrate-autograd-fns-to-python` ran earlier, they already wrote `_meta.py` / `_autograd.py` under the legacy package. Copy those into the stable package and update imports inside them to reference the **stable** library namespace (e.g., `torch.ops.extension_cpp_stable::mymuladd` instead of `extension_cpp::mymuladd`) once `migrate-to-stable-apis` swaps the registration macros.

For now (skeleton phase), leave them commented out in `__init__.py` and let `migrate-to-stable-apis` activate them as it migrates the corresponding ops.

### 5. Verify both targets build

```bash
<build_cmd>
```

Expected:
- Legacy target builds and links as before — unchanged behavior.
- Stable target builds successfully but produces an essentially empty `.so` (no sources yet). That's fine; it proves the build flags are accepted.

If the stable target fails to build with "no sources" — depends on the build system — temporarily add a single trivial `.cpp` containing only `STABLE_TORCH_LIBRARY(<stable_pkg>_lib, m) {}` so the build has something to compile. `migrate-to-stable-apis` will replace this once a real file arrives.

### 6. Run the project's existing tests

The legacy target is untouched, so `test_cmd` should still pass. This establishes a green baseline before file migration begins.

## Critical rules

1. **Don't classify or move any files in this step.** Setup only. The legacy `csrc/` is unchanged; the stable `csrc/` is empty. File migration is `migrate-to-stable-apis`' job.
2. **Use a distinct library namespace for the stable target.** If the legacy registers ops under `extension_cpp`, the stable target registers under `extension_cpp_stable` (or similar). This keeps the dispatcher from seeing duplicate registrations during the transition. Python users can keep calling `extension_cpp.<op>` because the legacy is still live; once the migration completes, the user renames/redirects.
3. **Do not delete the legacy directory yet.** It's the safety net during migration. Decommissioning happens in the completion step below, after all files have moved and tests pass against the stable target.
4. **Keep `TORCH_TARGET_VERSION` and `Py_LIMITED_API` only on the stable target.** Putting them on the legacy target will cause legacy includes to fail and erase the whole point of having a parallel build.

## Completion (after `migrate-to-stable-apis` finishes)

When `migrate-to-stable-apis` reports that all legacy sources have been migrated and the legacy target's `sources=` list is empty:

1. **Confirm the legacy target is empty.** `sources=[]` for the legacy entry in `setup.py`.
2. **Run the project's tests against the stable target only.** If they pass, the legacy target is no longer needed.
3. **Remove the legacy entry from `setup.py`.** Delete the legacy `<legacy_pkg>/csrc/` directory.
4. **Decide what to do with the legacy Python package.** Two options:
   - **Rename in place**: rename `<stable_pkg>/` → `<legacy_pkg>/` and update `setup.py` so the stable target installs under the original package name. Users see no change.
   - **Keep both names**: have `<legacy_pkg>/__init__.py` re-export from `<stable_pkg>` as a transitional alias. Useful if downstream code imports specific submodule paths.
5. **Final build** with the cleaned-up `setup.py`. Run tests one more time. The project now ships a single stable wheel.

## Handoff

The orchestrator next dispatches `migrate-to-stable-apis`, which performs the file-by-file move + rewrite + per-file build/test loop into the stable target this skill just set up.
