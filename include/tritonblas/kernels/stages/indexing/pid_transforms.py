import triton
import triton.language as tl

@triton.jit()
def chiplet_transform(
    pid,
    num_workgroups: tl.constexpr,
    num_xcds: tl.constexpr
):
    xcd = pid % num_xcds 
    pos_in_xcd = pid // num_xcds 
    min_per_xcd = num_workgroups // num_xcds 
    extra_sms = num_workgroups % num_xcds 
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd

@triton.jit()
def chiplet_transform_chunked(
    pid, 
    num_workgroups: tl.constexpr, 
    num_xcds: tl.constexpr, 
    chunk_size: tl.constexpr
):
    if pid > (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size):
        # Outside of the contiguous chunked region, leave unchanged.
        return pid
    
    local_pid = pid // num_xcds 
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // chunk_size 
    pos_in_chunk = local_pid % chunk_size 

    # Calculate new PID
    xcd = pid % num_xcds 
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid

@triton.jit
def remap_xcd_chunked(
    pid, GRID_MN, NUM_XCDS: tl.constexpr = 8, CHUNK_SIZE: tl.constexpr = 2
):
    # Compute current XCD and local PID
    xcd = pid % NUM_XCDS
    # distribute the modulo pids in round robin
    if pid > (GRID_MN // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    # Calculate chunk index and position within chunk
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    # Calculate new PID
    new_pid = chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk
    return new_pid
