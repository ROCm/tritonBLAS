// HIP GEMM benchmark with optional RCCL overlap for rocprofv3 counter collection.
//
// Build:
//   hipcc -O2 -o hip_gemm_bench hip_gemm_bench.cpp \
//       -I/opt/rocm/include -L/opt/rocm/lib \
//       -lhipblas -lrccl -lamdhip64
//
// Usage (single GPU):
//   ./hip_gemm_bench --mode alone --m 8192 --n 8192 --k 8192 --steps 20
//   ./hip_gemm_bench --mode rotating --m 8192 --n 8192 --k 8192 --n-bufs 4
//
// Usage (multi-GPU with RCCL):
//   mpirun -np 8 ./hip_gemm_bench --mode rccl --m 8192 --n 8192 --k 8192
//   mpirun -np 8 ./hip_gemm_bench --mode rccl-rotating --m 8192 --n 8192 --k 8192

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rccl/rccl.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define HIPBLAS_CHECK(cmd) do { \
    hipblasStatus_t s = (cmd); \
    if (s != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error %d at %s:%d\n", s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = (cmd); \
    if (r != ncclSuccess) { \
        fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

struct GemmBuffers {
    hipblasHalf *A, *B, *C;
    int m, n, k;
};

GemmBuffers allocate_gemm(int m, int n, int k) {
    GemmBuffers buf;
    buf.m = m; buf.n = n; buf.k = k;
    HIP_CHECK(hipMalloc(&buf.A, (size_t)m * k * sizeof(hipblasHalf)));
    HIP_CHECK(hipMalloc(&buf.B, (size_t)k * n * sizeof(hipblasHalf)));
    HIP_CHECK(hipMalloc(&buf.C, (size_t)m * n * sizeof(hipblasHalf)));

    size_t max_sz = std::max({(size_t)m*k, (size_t)k*n, (size_t)m*n});
    std::vector<hipblasHalf> host(max_sz);
    for (size_t i = 0; i < max_sz; i++)
        host[i] = hipblasHalf(((double)rand() / RAND_MAX - 0.5) * 0.1);
    HIP_CHECK(hipMemcpy(buf.A, host.data(), (size_t)m*k*sizeof(hipblasHalf), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.B, host.data(), (size_t)k*n*sizeof(hipblasHalf), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(buf.C, 0, (size_t)m*n*sizeof(hipblasHalf)));
    return buf;
}

void free_gemm(GemmBuffers& buf) {
    (void)hipFree(buf.A); (void)hipFree(buf.B); (void)hipFree(buf.C);
}

void run_gemm(hipblasHandle_t handle, GemmBuffers& buf, hipStream_t stream) {
    hipblasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        buf.n, buf.m, buf.k,
        &alpha,
        buf.B, HIPBLAS_R_16F, buf.n,
        buf.A, HIPBLAS_R_16F, buf.k,
        &beta,
        buf.C, HIPBLAS_R_16F, buf.n,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

int main(int argc, char** argv) {
    int m = 8192, n = 8192, k = 8192;
    int steps = 20, warmup = 10, n_bufs = 4;
    int comm_m = 8192, comm_n = 8192;
    const char* mode = "alone";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--m") && i+1 < argc) m = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n") && i+1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-bufs") && i+1 < argc) n_bufs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--comm-m") && i+1 < argc) comm_m = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--comm-n") && i+1 < argc) comm_n = atoi(argv[++i]);
    }

    bool use_rccl = (strstr(mode, "rccl") != nullptr);
    bool use_rotating = (strstr(mode, "rotating") != nullptr);

    int rank = 0, world_size = 1;
    ncclComm_t nccl_comm = nullptr;

    if (use_rccl) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        HIP_CHECK(hipSetDevice(rank));

        ncclUniqueId nccl_id;
        if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
        NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));
    } else {
        HIP_CHECK(hipSetDevice(0));
    }

    if (rank == 0)
        printf("Mode: %s  GEMM: %dx%dx%d  steps=%d warmup=%d  world=%d\n",
               mode, m, n, k, steps, warmup, world_size);

    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    hipStream_t gemm_stream, comm_stream;
    HIP_CHECK(hipStreamCreate(&gemm_stream));
    HIP_CHECK(hipStreamCreate(&comm_stream));

    // Allocate communication buffer for RCCL
    hipblasHalf* comm_buf = nullptr;
    size_t comm_elems = (size_t)comm_m * comm_n;
    if (use_rccl) {
        HIP_CHECK(hipMalloc(&comm_buf, comm_elems * sizeof(hipblasHalf)));
        HIP_CHECK(hipMemset(comm_buf, 0, comm_elems * sizeof(hipblasHalf)));
    }

    if (!use_rotating) {
        // Single buffer mode
        GemmBuffers buf = allocate_gemm(m, n, k);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            if (use_rccl) {
                NCCL_CHECK(ncclAllReduce(comm_buf, comm_buf, comm_elems,
                    ncclFloat16, ncclSum, nccl_comm, comm_stream));
            }
            run_gemm(handle, buf, gemm_stream);
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Timed
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, gemm_stream));

        for (int i = 0; i < steps; i++) {
            if (use_rccl) {
                NCCL_CHECK(ncclAllReduce(comm_buf, comm_buf, comm_elems,
                    ncclFloat16, ncclSum, nccl_comm, comm_stream));
            }
            run_gemm(handle, buf, gemm_stream);
            if (use_rccl) HIP_CHECK(hipDeviceSynchronize());
        }

        HIP_CHECK(hipEventRecord(stop, gemm_stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        if (rank == 0) {
            const char* desc = use_rccl ? "GEMM + RCCL overlap (warm L2)" : "GEMM alone (warm L2)";
            printf("%s: %.3f ms/iter (%d iters)\n", desc, ms / steps, steps);
        }
        free_gemm(buf);
    }
    else {
        // Rotating buffer mode
        std::vector<GemmBuffers> bufs;
        for (int i = 0; i < n_bufs; i++) bufs.push_back(allocate_gemm(m, n, k));

        // Warmup
        for (int i = 0; i < std::max(warmup, n_bufs); i++) {
            if (use_rccl) {
                NCCL_CHECK(ncclAllReduce(comm_buf, comm_buf, comm_elems,
                    ncclFloat16, ncclSum, nccl_comm, comm_stream));
            }
            run_gemm(handle, bufs[i % n_bufs], gemm_stream);
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Timed
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, gemm_stream));

        for (int i = 0; i < steps; i++) {
            if (use_rccl) {
                NCCL_CHECK(ncclAllReduce(comm_buf, comm_buf, comm_elems,
                    ncclFloat16, ncclSum, nccl_comm, comm_stream));
            }
            run_gemm(handle, bufs[i % n_bufs], gemm_stream);
            if (use_rccl) HIP_CHECK(hipDeviceSynchronize());
        }

        HIP_CHECK(hipEventRecord(stop, gemm_stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        if (rank == 0) {
            const char* desc = use_rccl ? "GEMM + RCCL rotating (cold L2)" : "GEMM rotating (cold L2)";
            printf("%s [%d bufs]: %.3f ms/iter (%d iters)\n", desc, n_bufs, ms / steps, steps);
        }
        for (auto& b : bufs) free_gemm(b);
    }

    if (comm_buf) (void)hipFree(comm_buf);
    HIPBLAS_CHECK(hipblasDestroy(handle));
    HIP_CHECK(hipStreamDestroy(gemm_stream));
    HIP_CHECK(hipStreamDestroy(comm_stream));

    if (use_rccl) {
        ncclCommDestroy(nccl_comm);
        MPI_Finalize();
    }
    return 0;
}
