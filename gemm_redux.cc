/*! @copyright (c) 2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * Gemm example, backed by the StarPU with redux mode
 *
 * @file gemm_redux.cc
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-18
 * */

#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <starpu.h>
#include <starpu_cublas_v2.h>

void zero_func(void *buffers[], void *cl_args)
{
    float *data = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int nx = (int)STARPU_MATRIX_GET_NX(buffers[0]);
    int ny = (int)STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
    if(ld != nx)
    {
        std::cerr << "ld != nx\n";
    }
    // Get CUDA stream
    cudaStream_t stream = starpu_cuda_get_local_stream();
    // Clear buffer
    std::size_t size = sizeof(float) * ld * ny;
    cudaMemsetAsync(data, 0, size, stream);
}

void add_func(void *buffers[], void *cl_args)
{
    float *dst = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int nx_dst = (int)STARPU_MATRIX_GET_NX(buffers[0]);
    int ny_dst = (int)STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned ld_dst = STARPU_MATRIX_GET_LD(buffers[0]);
    const float *src = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
    int nx_src = (int)STARPU_MATRIX_GET_NX(buffers[1]);
    int ny_src = (int)STARPU_MATRIX_GET_NY(buffers[1]);
    unsigned ld_src = STARPU_MATRIX_GET_LD(buffers[1]);
    if(ld_dst != nx_dst)
    {
        std::cerr << "ld_dst != nx_dst\n";
    }
    if(ld_src != nx_src)
    {
        std::cerr << "ld_src != nx_src\n";
    }
    if(nx_dst != nx_src)
    {
        std::cerr << "nx_dst != nx_src\n";
    }
    if(ny_dst != ny_src)
    {
        std::cerr << "ny_dst != ny_src\n";
    }
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasSetStream(handle, stream);
    // Call corresponding cuBLAS routine
    const float one = 1.0;
    cublasSaxpy(handle, ld_src*ny_src, &one, src, 1, dst, 1);
}

void gemm_func(void *buffers[], void *cl_args)
{
    const float *A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
    int nxA = (int)STARPU_MATRIX_GET_NX(buffers[0]);
    int nyA = (int)STARPU_MATRIX_GET_NY(buffers[0]);
    unsigned ldA = STARPU_MATRIX_GET_LD(buffers[0]);
    const float *B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
    int nxB = (int)STARPU_MATRIX_GET_NX(buffers[1]);
    int nyB = (int)STARPU_MATRIX_GET_NY(buffers[1]);
    unsigned ldB = STARPU_MATRIX_GET_LD(buffers[1]);
    float *C = (float *)STARPU_MATRIX_GET_PTR(buffers[2]);
    int nxC = (int)STARPU_MATRIX_GET_NX(buffers[2]);
    int nyC = (int)STARPU_MATRIX_GET_NY(buffers[2]);
    unsigned ldC = STARPU_MATRIX_GET_LD(buffers[2]);
    if(nxA != ldA)
    {
        std::cerr << "nxA != ldA\n";
    }
    if(nxB != ldB)
    {
        std::cerr << "nxB != ldB\n";
    }
    if(nxC != ldC)
    {
        std::cerr << "nxC != ldC\n";
    }
    if(nxA != nxC)
    {
        std::cerr << "nxA != nxC\n";
    }
    if(nyA != nxB)
    {
        std::cerr << "nyA != nxB\n";
    }
    if(nyB != nyC)
    {
        std::cerr << "nyB != nyC\n";
    }
    // Get cuBLAS handle and CUDA stream
    cublasHandle_t handle = starpu_cublas_get_local_handle();
    cudaStream_t stream = starpu_cuda_get_local_stream();
    cublasSetStream(handle, stream);
    // Call corresponding cuBLAS routine
    const float one = 1.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nxC, nyC, nyA, &one, A, ldA,
            B, ldB, &one, C, ldC);
}

starpu_perfmodel zero_perf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "zero"
};

starpu_perfmodel add_perf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "add"
};

starpu_perfmodel gemm_perf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "gemm"
};

starpu_codelet zero_cl =
{
    .cuda_funcs = {zero_func},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 1,
    .modes = {STARPU_W},
    .model = &zero_perf
};

starpu_codelet add_cl =
{
    .cuda_funcs = {add_func},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 2,
    .modes = {static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
        STARPU_R},
    .model = &add_perf
};

starpu_codelet gemm_cl =
{
    .cuda_funcs = {gemm_func},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_REDUX},
    .model = &gemm_perf
};

int main(int argc, char **argv)
{
    // Init StarPU and cuBLAS
    int ret = starpu_init(nullptr);
    if(ret != 0)
    {
        throw std::runtime_error("StarPU init problem");
    }
    starpu_cublas_init();
    // Check arguments
    if(argc != 6)
    {
        throw std::runtime_error("Wrong number of parameters: gemm_redux M N K D NB");
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int D = atoi(argv[4]);
    int NB = atoi(argv[5]);
    std::cout << "M=" << M << " N=" << N << " K=" << K << " D=" << D << " NB="
        << NB << "\n";
    // Initialize data handles and fill with zeros
    std::vector<starpu_data_handle_t> A(D*NB), B(D*NB), C(D);
    for(int i = 0; i < D*NB; ++i)
    {
        starpu_matrix_data_register(&A[i], -1, 0, M, M, K, sizeof(float));
        starpu_task_insert(&zero_cl, STARPU_W, A[i], 0);
        starpu_matrix_data_register(&B[i], -1, 0, K, K, N, sizeof(float));
        starpu_task_insert(&zero_cl, STARPU_W, B[i], 0);
    }
    for(int i = 0; i < D; ++i)
    {
        starpu_matrix_data_register(&C[i], -1, 0, M, M, N, sizeof(float));
        starpu_task_insert(&zero_cl, STARPU_W, C[i], 0);
        starpu_data_set_reduction_methods(C[i], &add_cl, &zero_cl);
    }
    // Run warmup
    for(int i = 0; i < NB; ++i)
    {
        for(int j = 0; j < D; ++j)
        {
            starpu_task_insert(&gemm_cl, STARPU_R, A[i*D+j], STARPU_R, B[i*D+j],
                    STARPU_REDUX, C[j], 0);
        }
    }
    // Run compute
    starpu_task_wait_for_all();
    const auto start{std::chrono::steady_clock::now()};
    for(int i = 0; i < NB; ++i)
    {
        for(int j = 0; j < D; ++j)
        {
            starpu_task_insert(&gemm_cl, STARPU_R, A[i*D+j], STARPU_R, B[i*D+j],
                    STARPU_REDUX, C[j], 0);
        }
    }
    // Wait for all computations to finish
    starpu_task_wait_for_all();
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout << "Time: " << elapsed_seconds.count() << "\n";
    // Unregister data and shut down StarPU
    for(int i = 0; i < D*NB; ++i)
    {
        starpu_data_unregister(A[i]);
        starpu_data_unregister(B[i]);
    }
    for(int i = 0; i < D; ++i)
    {
        starpu_data_unregister(C[i]);
    }
    starpu_shutdown();
}

