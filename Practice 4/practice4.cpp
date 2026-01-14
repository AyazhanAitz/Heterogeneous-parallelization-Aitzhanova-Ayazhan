%%writefile main.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

// ===================== ПАРАМЕТРЫ =====================
const int BLOCK_SIZE = 256;

// ===================== ПРОВЕРКА CUDA =====================
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : "
                  << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// ===================== РЕДУКЦИЯ: GLOBAL ONLY =====================
__global__ void reduceGlobalOnly(const int* data, int n, unsigned long long* result) {
    unsigned long long localSum = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        localSum += data[i];
    }

    atomicAdd(result, localSum);
}

// ===================== РЕДУКЦИЯ: GLOBAL + SHARED =====================
__global__ void reduceGlobalShared(const int* data, int n, unsigned long long* result) {
    __shared__ unsigned long long shared[BLOCK_SIZE];

    unsigned long long localSum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        localSum += data[i];
    }

    shared[threadIdx.x] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, shared[0]);
    }
}

// ===================== ГЕНЕРАЦИЯ МАССИВА =====================
std::vector<int> generateData(int n) {
    std::vector<int> v(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 100);

    for (int i = 0; i < n; i++)
        v[i] = dist(gen);

    return v;
}

// ===================== MAIN =====================
int main() {
    std::ofstream csv("results.csv");
    csv << "N,global_only_ms,global_shared_ms\n";

    std::vector<int> sizes = {10000, 100000, 1000000};

    for (int n : sizes) {
        std::vector<int> h_data = generateData(n);

        int* d_data;
        unsigned long long* d_sum;

        checkCuda(cudaMalloc(&d_data, n * sizeof(int)), "malloc d_data");
        checkCuda(cudaMalloc(&d_sum, sizeof(unsigned long long)), "malloc d_sum");

        checkCuda(cudaMemcpy(d_data, h_data.data(),
                              n * sizeof(int), cudaMemcpyHostToDevice),
                  "copy to device");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // -------- GLOBAL ONLY --------
        checkCuda(cudaMemset(d_sum, 0, sizeof(unsigned long long)), "memset");
        cudaEventRecord(start);

        reduceGlobalOnly<<<120, BLOCK_SIZE>>>(d_data, n, d_sum);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeGlobal;
        cudaEventElapsedTime(&timeGlobal, start, stop);

        // -------- GLOBAL + SHARED --------
        checkCuda(cudaMemset(d_sum, 0, sizeof(unsigned long long)), "memset");
        cudaEventRecord(start);

        reduceGlobalShared<<<120, BLOCK_SIZE>>>(d_data, n, d_sum);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeShared;
        cudaEventElapsedTime(&timeShared, start, stop);

        csv << n << "," << timeGlobal << "," << timeShared << "\n";

        cudaFree(d_data);
        cudaFree(d_sum);

        std::cout << "N=" << n
                  << " global=" << timeGlobal << " ms"
                  << " shared=" << timeShared << " ms\n";
    }

    csv.close();
    return 0;
}

