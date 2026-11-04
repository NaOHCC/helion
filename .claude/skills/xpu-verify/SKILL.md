---
name: xpu-verify
description: Verify changes to the Helion Intel XPU Triton backend and autotuner by running lint, XPU/Roofline test suites, and a code review subagent — then fix any issues found.
---

Run all checks, then fix any issues:

1. **Lint & Type Check** — `./lint.sh` (Ruff format, Ruff check, codespell, and Pyrefly).
2. **XPU & Roofline Core Tests** — `pytest test/test_roofline.py test/test_xpu_tunables.py test/test_xpu_hw_collect.py -x -vv -s -n0`.
3. **Relevant Compiler / Autotuner Tests** — Run the `pytest test/test_*.py` suites covering all modified files in `helion/_compiler/`, `helion/autotuner/`, and `helion/runtime/`.
4. **Code Review Subagent** — spawn a `code-review` or `general-purpose` subagent to inspect `git diff` against the base commit:

> Review the Intel XPU Triton backend / autotuner changes in `<repo root>`. Focus on:
> - **Generality (No Kernel-Specific Hacks):** Ensure compiler passes, tunables, and autotuner seeds are general and not hardcoded to specific benchmark tensor shapes or kernel names.
> - **Correctness & Cross-Backend Safety:** Verify changes do not break CUDA/ROCm paths or alter numerical semantics outside allowed precision rules.
> - **Cleanliness:** Check for redundant code, unguarded XPU-only assumptions when running on CPU/CUDA mocks, or missing test coverage.
>
> Report concrete `file:line` issues under 400 words, or say "LGTM" if clean.

If any check fails, fix the underlying issue and re-run until all checks pass cleanly.

Final report:

| Check | Result |
| :--- | :--- |
| `./lint.sh` | clean / N issues fixed |
| `pytest` (XPU & Roofline tests) | `<pass>/<skip>` |
| `pytest` (Compiler / Autotuner tests) | `<pass>/<skip>` |
| Code Review | LGTM / `<N>` issues fixed |
