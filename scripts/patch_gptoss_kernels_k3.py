"""Patch kernels-community/triton_kernels 0.12.x for gpt-oss top_k=3.

The MXFP4 expert matmul's writeback-index kernel uses
``tl.arange(0, N_EXPTS_ACT)``. Triton requires arange lengths to be powers of
two, so k=3 cannot compile. gpt-oss has at most four active experts; use a
four-lane vector and mask the fourth lane for k=2/3. This preserves the MXFP4
expert matmuls and works together with bench_fixed_shapes.py's arbitrary-k
PyTorch router.

Run this as a short setup process before launching the benchmark process; the
kernel module must be imported after its source has been patched.
"""
from pathlib import Path

from kernels import get_kernel


hub = get_kernel("kernels-community/triton_kernels")
kernel_root = Path(hub.__file__).parent
source = (
    kernel_root
    / "matmul_ogs_details"
    / "_matmul_ogs.py"
)
text = source.read_text()

old_first = """    src_offs = (dst_idxs // N_EXPTS_ACT) * N_EXPTS_ACT
    src_offs = src_offs[:, None] + tl.arange(0, N_EXPTS_ACT)[None, :]
    src_idxs = tl.load(ScatterSrcIndx + src_offs, mask=mask[:, None], other=-1)
"""
new_first = """    src_offs = (dst_idxs // N_EXPTS_ACT) * N_EXPTS_ACT
    offs_expt = tl.arange(0, 4)
    src_offs = src_offs[:, None] + offs_expt[None, :]
    src_idxs = tl.load(
        ScatterSrcIndx + src_offs,
        mask=mask[:, None] & (offs_expt[None, :] < N_EXPTS_ACT),
        other=-1,
    )
"""

old_second = """    src_offs = offs_m[:, None] * N_EXPTS_ACT + tl.arange(0, N_EXPTS_ACT)[None, :]
    src_idxs = tl.load(ScatterSrcIndx + src_offs, mask=mask_m[:, None], other=-1)
"""
new_second = """    offs_expt = tl.arange(0, 4)
    src_offs = offs_m[:, None] * N_EXPTS_ACT + offs_expt[None, :]
    src_idxs = tl.load(
        ScatterSrcIndx + src_offs,
        mask=mask_m[:, None] & (offs_expt[None, :] < N_EXPTS_ACT),
        other=-1,
    )
"""

old_third = """    tl.store(Ptr + N_EXPTS_ACT * arange[:, None] + tl.arange(0, N_EXPTS_ACT)[None, :], s, mask=(arange < finalize_scatter_count)[:, None])
"""
new_third = """    tl.store(
        Ptr + N_EXPTS_ACT * arange[:, None] + offs_expt[None, :],
        s,
        mask=(arange < finalize_scatter_count)[:, None]
        & (offs_expt[None, :] < N_EXPTS_ACT),
    )
"""

replacements = (
    (old_first, new_first),
    (old_second, new_second),
    (old_third, new_third),
)
changed = False
for old, new in replacements:
    if new in text:
        continue
    if old not in text:
        raise RuntimeError(
            f"Expected triton_kernels 0.12.x source not found: {source}"
        )
    text = text.replace(old, new, 1)
    changed = True

if not changed:
    print(f"already patched: {source}")
else:
    source.write_text(text)
    print(f"patched: {source}")


finalize_source = (
    kernel_root
    / "matmul_ogs_details"
    / "_finalize_matmul.py"
)
finalize_text = finalize_source.read_text()
finalize_replacements = (
    (
        """        src_offs = pid_m * EXPT_PER_TOK + tl.arange(0, EXPT_PER_TOK)
""",
        """        offs_expt = tl.arange(0, 4)
        active_expt_mask = offs_expt < EXPT_PER_TOK
        src_offs = pid_m * EXPT_PER_TOK + offs_expt
""",
    ),
    (
        """            src_idxs = tl.load(FinalizeScatterIdxs + M + src_offs)
""",
        """            src_idxs = tl.load(
                FinalizeScatterIdxs + M + src_offs,
                mask=active_expt_mask,
                other=-1,
            )
""",
    ),
    (
        """            src_idxs = tl.load(ScatterSrcIndx + src_offs)
""",
        """            src_idxs = tl.load(
                ScatterSrcIndx + src_offs,
                mask=active_expt_mask,
                other=-1,
            )
""",
    ),
    (
        """            src_idxs = src_offs
""",
        """            src_idxs = tl.where(active_expt_mask, src_offs, -1)
""",
    ),
)
finalize_changed = False
for old, new in finalize_replacements:
    if new in finalize_text:
        continue
    if old not in finalize_text:
        raise RuntimeError(
            f"Expected triton_kernels 0.12.x source not found: {finalize_source}"
        )
    finalize_text = finalize_text.replace(old, new, 1)
    finalize_changed = True

if finalize_changed:
    finalize_source.write_text(finalize_text)
    print(f"patched: {finalize_source}")
else:
    print(f"already patched: {finalize_source}")
