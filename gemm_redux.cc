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

// Make all entries of a buffer zero (CUDA)
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

// Add two vectors (CUDA)
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

// Multiply two matrices and accumulate result
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

// Perf model for the clearing function
starpu_perfmodel zero_perf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "zero"
};

// Perf model for the result accumulation function
starpu_perfmodel add_perf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "add"
};

// Perf model for matrix multiplication
starpu_perfmodel gemm_perf =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "gemm"
};

// Codelet for the clearing function
starpu_codelet zero_cl =
{
    .cuda_funcs = {zero_func},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 1,
    .modes = {STARPU_W},
    .model = &zero_perf
};

// Codelet for the result accumulation function
starpu_codelet add_cl =
{
    .cuda_funcs = {add_func},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 2,
    .modes = {static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE),
        STARPU_R},
    .model = &add_perf
};

// Codelet for the matrix multiplication
starpu_codelet gemm_cl =
{
    .cuda_funcs = {gemm_func},
    .cuda_flags = {STARPU_CUDA_ASYNC},
    .nbuffers = 3,
    .modes = {STARPU_R, STARPU_R, STARPU_REDUX},
    .model = &gemm_perf
};

// Main executable
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
    if(argc != 8)
    {
        std::cerr << "Wrong number of parameters. This executable takes "
            "following arguments:\n\t./gemm_redux M N K D NB R mode\n"
            "This executable initializes a list of D*NB input matrices A of a shape "
            "M-by-K, a list of D*NB input matrices B of a shape K-by-N and a list of D matrices C"
            " of a shape M-by-N. Then, each matrix C[i] accumulates product of"
            "NB pairs of corresponding matrices A[j] and B[j] (with properly "
            "defined indices j), this process repeats R times. This process can be simply described as "
            "follows:\nfor r = 0, 1, ..., R-1\n\tfor i = 0, 1, ..., D-1\n\t\tfor j = 0, 1, ..., NB-1"
            "\n\t\t\tC[i] += A[i][j] * B[i][j]\nParameters:\n\tM: number of "
            "rows of matrices A[i][j] and C[i]\n\tN: number of columns of "
            "matrices B[i][j] and C[i]\n\tK: number of columns of matrices "
            "A[i][j] and number of rows of matrices B[i][j]\n\tD: number of "
            "matrices in a list C\n\tNB: number of matrices in lists A[i] and "
            "B[i]\n\tmode: Access mode for matrices C[i]. It is 0 for "
            "STARPU_RW and 1 for STARPU_REDUX\n";
        return 1;
    }
    // Read arguments
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int D = atoi(argv[4]);
    int NB = atoi(argv[5]);
    int R = atoi(argv[6]);
    int mode = atoi(argv[7]);
    if(M <= 0 or N <= 0 or K <=0 or D <= 0 or NB <= 0 or R <= 0 or mode < 0 or mode > 1)
    {
        throw std::runtime_error("Invalid input value");
    }
    std::cout << "M=" << M << " N=" << N << " K=" << K << " D=" << D << " NB="
        << NB << " R=" << R << " mode=" << mode << "\n";
    enum starpu_data_access_mode C_mode;
    if(mode == 0)
    {
        C_mode = STARPU_RW;
    }
    else
    {
        C_mode = STARPU_REDUX;
    }
    gemm_cl.modes[2] = C_mode;
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
                    C_mode, C[j], 0);
        }
    }
    // Run compute with one way of task submission and measure time
    starpu_task_wait_for_all();
    //starpu_pause();
    for(int r = 0; r < R; ++r)
    {
        for(int i = 0; i < NB; ++i)
        {
            for(int j = 0; j < D; ++j)
            {
                starpu_task_insert(&gemm_cl, STARPU_R, A[i*D+j], STARPU_R, B[i*D+j],
                        C_mode, C[j], 0);
            }
        }
    }
    const auto start1{std::chrono::steady_clock::now()};
    //starpu_resume();
    // Wait for all computations to finish
    starpu_task_wait_for_all();
    const auto end1{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds1{end1 - start1};
    std::cout << "Time for one way of task submission: "
        << elapsed_seconds1.count() << " seconds\n";
    // Run compute with another way of task submission
    starpu_task_wait_for_all();
    //starpu_pause();
    for(int r = 0; r < R; ++r)
    {
        for(int j = 0; j < D; ++j)
        {
            for(int i = 0; i < NB; ++i)
            {
                starpu_task_insert(&gemm_cl, STARPU_R, A[i*D+j], STARPU_R, B[i*D+j],
                        C_mode, C[j], 0);
            }
        }
    }
    const auto start2{std::chrono::steady_clock::now()};
    //starpu_resume();
    // Wait for all computations to finish
    starpu_task_wait_for_all();
    const auto end2{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds2{end2 - start2};
    std::cout << "Time for another way of task submission: "
        << elapsed_seconds2.count() << " seconds\n";
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
    starpu_cublas_shutdown();
    starpu_shutdown();
}

