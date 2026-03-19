import torch
import triton
import triton.language as tl


@triton.jit
def cu_hog_alu_kernel(out_ptr, n_iters, BLOCK: tl.constexpr):
    """Pure ALU CU-hog (no memory pressure)."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = (offs + pid).to(tl.float32)
    i = 0
    while i < n_iters:
        acc = acc * 1.00001 + 0.00001
        i += 1
    tl.store(out_ptr + pid * BLOCK + offs, acc)


@triton.jit
def cu_hog_mem_kernel(buf_ptr, n_iters, stride, BLOCK: tl.constexpr):
    """Memory-bound CU-hog: reads/writes a large buffer in a loop.
    Each WG streams through 'stride' elements per iteration."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = pid * stride + offs
    i = 0
    while i < n_iters:
        # Read block, do minimal FMA, write back (streaming pattern)
        vals = tl.load(buf_ptr + base)
        vals = vals + 0.001
        tl.store(buf_ptr + base, vals)
        base = base + BLOCK
        # Wrap around within this WG's stride region
        wrap_mask = base >= (pid + 1) * stride
        base = tl.where(wrap_mask, pid * stride + offs, base)
        i += 1


torch.cuda.set_device(0)
BLOCK = 256
N_WGS = 32

out_alu = torch.empty(N_WGS * BLOCK, dtype=torch.float32, device='cuda')

# Memory buffer: 32 WGs * 1MB each = 32MB (similar to 4Kx4K comm)
# bf16: 2 bytes/elem, so 512K elems per WG (1MB)
STRIDE = 512 * 1024  # elements per WG
buf_mem = torch.randn(N_WGS * STRIDE, dtype=torch.bfloat16, device='cuda')

# Warmup
cu_hog_alu_kernel[(N_WGS,)](out_alu, 1000, BLOCK=BLOCK)
cu_hog_mem_kernel[(N_WGS,)](buf_mem, 100, STRIDE, BLOCK=BLOCK)
torch.cuda.synchronize()

print("=== ALU-only CU-hog (32 WGs) ===")
for iters in [50000, 100000, 200000]:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    cu_hog_alu_kernel[(N_WGS,)](out_alu, iters, BLOCK=BLOCK)
    e.record()
    torch.cuda.synchronize()
    print(f'  N_ITERS={iters:>8d}  duration={s.elapsed_time(e):.3f} ms')

print("\n=== Memory-streaming CU-hog (32 WGs, 32MB buffer) ===")
for iters in [5000, 7000, 8000, 9000, 12000, 15000, 20000]:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    cu_hog_mem_kernel[(N_WGS,)](buf_mem, iters, STRIDE, BLOCK=BLOCK)
    e.record()
    torch.cuda.synchronize()
    print(f'  N_ITERS={iters:>8d}  duration={s.elapsed_time(e):.3f} ms')
