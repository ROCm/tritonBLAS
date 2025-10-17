import triton

@triton.jit()
def chiplet_transform(
    pid,
    num_workgroups,
    num_xcds
):
    xcd = pid % num_xcds 
    pos_in_xcd = pid // num_xcds 
    min_per_xcd = num_workgroups // num_xcds 
    extra_sms = num_workgroups % num_xcds 
    offset = xcd * min_per_xcd + min(xcd, extra_sms)
    return offset + pos_in_xcd
