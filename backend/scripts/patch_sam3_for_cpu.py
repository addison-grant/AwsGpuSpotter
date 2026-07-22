"""Patch the installed sam3 package to tolerate running without CUDA.

sam3 assumes a GPU is always present in a couple of places that break
import/build on macOS (no CUDA, no triton wheels) even though the actual
code paths involved are never exercised off-GPU:

  - model/edt.py unconditionally `import triton`, even though the kernel
    it defines is only ever invoked on CUDA tensors (`assert data.is_cuda`
    in `edt_triton`). Triton has no macOS wheels.
  - model/sam3_multiplex_base.py calls `torch.cuda.get_device_properties(0)`
    at module import time to decide whether to enable TF32, with no
    `torch.cuda.is_available()` guard.
  - model/position_encoding.py hardcodes `device="cuda"` when precomputing
    its positional-encoding cache at model build time.

The real GPU spot instance (Linux + CUDA) never touches these patches --
triton installs for real there and the CUDA calls succeed.

Run this once after every `uv sync` on macOS (which re-fetches sam3 from
git and wipes any previous patch):

    uv run python scripts/patch_sam3_for_cpu.py

Idempotent -- safe to re-run.
"""

import sys
import sysconfig

MARKER_PREFIX = "# --- patched by patch_sam3_for_cpu.py"

TRITON_SHIM = f'''{MARKER_PREFIX}: triton ---
import torch

try:
    import triton
    import triton.language as tl
except ImportError:

    def _jit(fn=None, **_kwargs):
        return fn if fn is not None else _jit

    class _FakeLanguage:
        constexpr = object

        def __getattr__(self, _name):
            def _unavailable(*_a, **_k):
                raise RuntimeError(
                    "triton is unavailable on this platform (no CUDA); "
                    "the EDT kernel cannot run off-GPU."
                )

            return _unavailable

    class _FakeTriton:
        jit = staticmethod(_jit)

    triton = _FakeTriton()
    tl = _FakeLanguage()
'''

TF32_SHIM = f'''{MARKER_PREFIX}: tf32 ---
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
'''

POSITION_ENCODING_SHIM = f'''{MARKER_PREFIX}: position_encoding ---
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tensors = torch.zeros((1, 1) + size, device=device)
'''

DECODER_SHIM = f'''{MARKER_PREFIX}: decoder ---
                coords_h, coords_w = self._get_coords(
                    feat_size, feat_size,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
'''

PATCHES = [
    {
        "relpath": "sam3/model/edt.py",
        "marker": f"{MARKER_PREFIX}: triton ---",
        "original": "import torch\nimport triton\nimport triton.language as tl\n",
        "replacement": TRITON_SHIM,
    },
    {
        "relpath": "sam3/model/sam3_multiplex_base.py",
        "marker": f"{MARKER_PREFIX}: tf32 ---",
        "original": "if torch.cuda.get_device_properties(0).major >= 8:\n",
        "replacement": TF32_SHIM,
    },
    {
        "relpath": "sam3/model/position_encoding.py",
        "marker": f"{MARKER_PREFIX}: position_encoding ---",
        "original": '                tensors = torch.zeros((1, 1) + size, device="cuda")\n',
        "replacement": POSITION_ENCODING_SHIM,
    },
    {
        "relpath": "sam3/model/decoder.py",
        "marker": f"{MARKER_PREFIX}: decoder ---",
        "original": (
            "                coords_h, coords_w = self._get_coords(\n"
            "                    feat_size, feat_size, device=\"cuda\"\n"
            "                )\n"
        ),
        "replacement": DECODER_SHIM,
    },
]


def apply_patch(site_packages, patch):
    path = f"{site_packages}/{patch['relpath']}"

    with open(path) as f:
        content = f.read()

    if patch["marker"] in content:
        print(f"{path} already patched, nothing to do.")
        return

    if patch["original"] not in content:
        print(
            f"error: expected snippet not found in {path}; "
            "sam3 may have changed upstream, patch manually.",
            file=sys.stderr,
        )
        sys.exit(1)

    content = content.replace(patch["original"], patch["replacement"], 1)

    with open(path, "w") as f:
        f.write(content)

    print(f"patched {path}")


def main():
    site_packages = sysconfig.get_paths()["purelib"]
    for patch in PATCHES:
        apply_patch(site_packages, patch)


if __name__ == "__main__":
    main()
