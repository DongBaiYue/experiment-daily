#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define L2_LINE_SIZE 128

__global__ void prefetch_data(int const* weight, size_t weight_size){
    size_t const idx = (blockDim.x * blockIdx.x + threadIdx.x)*L2_LINE_SIZE;
    size_t const stride = (blockDim.x * gridDim.x)*L2_LINE_SIZE;
    size_t const weight_byte_size = weight_size * sizeof(int);
    for(size_t i{idx}; i<weight_byte_size; i+=stride){
        __asm__ __volatile__("prefetch.global.L2 [%0];" :: "l"((char*)(weight)+i));
    }
}

__global__ void reset_data(int* input, int const* weight,
                           size_t input_size,
                           size_t weight_size)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < input_size; i += stride)
    {
        input[i] = weight[i % weight_size];
    }
}

void launch_prefetch_data(int const* weight, size_t weight_size){
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    prefetch_data<<<blocks_per_grid, threads_per_block>>>(weight, weight_size);
    CHECK_LAST_CUDA_ERROR();
}

/**
 * @brief Reset the input using weight so that the
 * input is weight repeatedly.
 *
 * @param input The data for reseting.
 * @param weight The values for resetting input.
 * @param input_size The size for input.
 * @param weight_size The size for weight.
 * @param stream The CUDA stream.
 */
void launch_reset_data(int* input, int const* weight,
                       size_t input_size, size_t weight_size)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    reset_data<<<blocks_per_grid, threads_per_block>>>(
        input, weight, input_size,
        weight_size);
    CHECK_LAST_CUDA_ERROR();
}

void full_function(){

}

float measure_performance(int* input, int const* weight,
                       size_t input_size, size_t weight_size, int num_repeats = 10, int num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        launch_prefetch_data(weight, weight_size);
        launch_reset_data(input, weight, input_size, weight_size);
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i{0}; i < num_repeats; ++i)
    {
        launch_prefetch_data(weight, weight_size);
        launch_reset_data(input, weight, input_size, weight_size);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

__global__ void a(){
   double value = 1.0;
   __asm__ __volatile__("prefetch.local.L2 [%0];" :: "l"(&value));
   printf("%f\n", value);
}

int main(){

    size_t const input_size_mb = 100;
    size_t const weight_size_mb = 4;
    std::cout << "Input Data Size: " << input_size_mb << " MB" << std::endl;
    std::cout << "Weight Data Size: " << weight_size_mb << " MB" << std::endl;
    size_t const input_size = input_size_mb * 1024 * 1024 / sizeof(int);
    size_t const weight_size = weight_size_mb * 1024 * 1024 / sizeof(int);
    size_t const input_size_byte = input_size_mb * 1024 * 1024;
    size_t const weight_size_byte = weight_size_mb * 1024 * 1024;
    std::vector<int> input_vec(input_size, 0);
    for (size_t i=0; i<input_size; i++){
        input_vec[i] = i;
    }
    std::vector<int> weight_vec(weight_size, 0);
    for (size_t i=0; i<weight_size; i++){
        weight_vec[i] = -i;
    }

    int * input_host = input_vec.data();
    int * weight_host = weight_vec.data();
    int * input_device;
    int * weight_device;
    CHECK_CUDA_ERROR(
        cudaMalloc(&input_device, input_size * sizeof(int)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&weight_device, weight_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(input_device, input_host,
                                input_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(weight_device, weight_host,
                                weight_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaProfilerStart();
    launch_prefetch_data(weight_device, weight_size);
    launch_reset_data(input_device, weight_device, input_size, weight_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaProfilerStop();
    CHECK_CUDA_ERROR(cudaMemcpy(input_host, input_device,
                                input_size * sizeof(int),
                                cudaMemcpyDeviceToHost));
    for (size_t i=0; i<10; i++){
        std::cout << input_host[i] << " ";
    }
    std::cout << std::endl;

    float const latency = measure_performance(input_device, weight_device, input_size, weight_size, 10, 10);
    std::cout << std::fixed << std::setprecision(3) << "Latency : "
              << latency << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaFree(input_device));
    CHECK_CUDA_ERROR(cudaFree(weight_device));
    return 0;
}
// nvcc -std=c++14 prefetch.cu -o prefetch
