%%writefile scan_shared.cu
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Макрос для проверки каждого CUDA-вызова
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

// Количество потоков в одном блоке
constexpr int BLOCK = 512;

// Один блок обрабатывает в 2 раза больше элементов, чем потоков
// Каждый поток загружает по два элемента
constexpr int ELEMS_PER_BLOCK = 2 * BLOCK;

// CPU-реализация inclusive prefix sum
// Каждый элемент — это сумма всех предыдущих элементов + текущий
void cpuInclusiveScan(const std::vector<float>& in, std::vector<float>& out) {
    out.resize(in.size());           // выходной массив того же размера
    double s = 0.0;                  // накапливаем сумму в double для лучшей точности
    for (size_t i = 0; i < in.size(); ++i) {
        s += in[i];                  // добавляем текущий элемент
        out[i] = (float)s;           // записываем результат как float
    }
}

// Kernel: скан внутри блока с использованием shared memory (Blelloch scan)
// На выходе получаем inclusive scan для каждого блока
__global__ void scanBlockBlellochInclusive(
    const float* d_in,               // входной массив на GPU
    float* d_out,                    // выходной массив на GPU
    float* d_blockSums,              // массив для хранения суммы каждого блока
    int n                             // реальный размер входного массива
) {
    // Shared memory для одного блока
    // Здесь храним данные, с которыми будут работать потоки блока
    __shared__ float s[ELEMS_PER_BLOCK];

    int tid = threadIdx.x;            // номер потока внутри блока
    int blockStart = blockIdx.x * ELEMS_PER_BLOCK; // начало данных для этого блока

    // Индексы двух элементов, которые обрабатывает один поток
    int i0 = blockStart + tid;
    int i1 = blockStart + tid + BLOCK;

    // Загружаем данные из глобальной памяти в shared memory
    // Если вышли за пределы массива — кладём 0
    s[tid] = (i0 < n) ? d_in[i0] : 0.0f;
    s[tid + BLOCK] = (i1 < n) ? d_in[i1] : 0.0f;

    __syncthreads();                  // ждём, пока все потоки загрузят данные

    // Фаза up-sweep (reduce)
    // Строим дерево сумм в shared memory
    for (int offset = 1; offset < ELEMS_PER_BLOCK; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < ELEMS_PER_BLOCK) {
            s[idx] += s[idx - offset];
        }
        __syncthreads();              // синхронизация после каждого шага
    }

    // В корне дерева лежит сумма всего блока
    // Сохраняем её и подготавливаем массив к down-sweep
    if (tid == 0) {
        d_blockSums[blockIdx.x] = s[ELEMS_PER_BLOCK - 1];
        s[ELEMS_PER_BLOCK - 1] = 0.0f;
    }
    __syncthreads();

    // Фаза down-sweep
    // Превращаем результат в exclusive scan
    for (int offset = ELEMS_PER_BLOCK >> 1; offset >= 1; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < ELEMS_PER_BLOCK) {
            float t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }

    // Перевод exclusive scan в inclusive scan
    // inclusive = exclusive + исходное значение
    if (i0 < n) d_out[i0] = s[tid] + d_in[i0];
    if (i1 < n) d_out[i1] = s[tid + BLOCK] + d_in[i1];
}

// Kernel для сканирования маленького массива (сумм блоков)
// Выполняется в одном блоке
__global__ void scanSmallInclusive(const float* d_in, float* d_out, int n) {
    __shared__ float s[2048];         // shared memory под суммы блоков

    int tid = threadIdx.x;

    // Загружаем данные в shared memory
    if (tid < n) s[tid] = d_in[tid];
    else s[tid] = 0.0f;

    __syncthreads();

    // Простой Hillis–Steele inclusive scan
    for (int offset = 1; offset < n; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset && tid < n) val = s[tid - offset];
        __syncthreads();
        if (tid < n) s[tid] += val;
        __syncthreads();
    }

    if (tid < n) d_out[tid] = s[tid];
}

// Kernel: добавляем префиксную сумму предыдущих блоков ко всем элементам блока
__global__ void addBlockOffsets(float* d_data, const float* d_blockPrefix, int n) {
    int blockId = blockIdx.x;          // номер текущего блока
    int tid = threadIdx.x;             // номер потока в блоке

    int blockStart = blockId * ELEMS_PER_BLOCK;

    // Оффсет равен сумме всех предыдущих блоков
    float offset = (blockId == 0) ? 0.0f : d_blockPrefix[blockId - 1];

    int i0 = blockStart + tid;
    int i1 = blockStart + tid + BLOCK;

    // Добавляем оффсет к элементам текущего блока
    if (i0 < n) d_data[i0] += offset;
    if (i1 < n) d_data[i1] += offset;
}

int main() {
    const int N = 1'000'000;           // размер массива
    const int ITERS = 20;              // количество запусков для усреднения времени

    // Генерация входных данных на CPU
    std::vector<float> h_in(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_in[i] = dist(rng);
    }

    // CPU prefix sum и измерение времени
    std::vector<float> h_cpu;
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    cpuInclusiveScan(h_in, h_cpu);
    auto cpu_t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_t2 - cpu_t1).count();

    // Выделяем память на GPU
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Копируем входной массив на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Считаем количество блоков
    int numBlocks = (N + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;

    // Память под суммы блоков и их префиксные суммы
    float *d_blockSums = nullptr, *d_blockPrefix = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_blockPrefix, numBlocks * sizeof(float)));

    // CUDA-события для измерения времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Прогрев GPU
    scanBlockBlellochInclusive<<<numBlocks, BLOCK>>>(d_in, d_out, d_blockSums, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;

    // Основные измерения
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaEventRecord(start));

        scanBlockBlellochInclusive<<<numBlocks, BLOCK>>>(d_in, d_out, d_blockSums, N);
        scanSmallInclusive<<<1, 1024>>>(d_blockSums, d_blockPrefix, numBlocks);
        addBlockOffsets<<<numBlocks, BLOCK>>>(d_out, d_blockPrefix, N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    float gpu_ms_avg = total_ms / ITERS;

    // Копируем результат обратно на CPU
    std::vector<float> h_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Проверка корректности на нескольких индексах
    auto diff = [&](int idx) {
        return std::abs((double)h_cpu[idx] - (double)h_gpu[idx]);
    };

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    std::cout << "N = " << N << "\n";
    std::cout << "CPU scan last  = " << h_cpu[N - 1] << "\n";
    std::cout << "GPU scan last  = " << h_gpu[N - 1] << "\n";
    std::cout << "Abs diff [0]   = " << diff(0) << "\n";
    std::cout << "Abs diff [N/2] = " << diff(N / 2) << "\n";
    std::cout << "Abs diff [N-1] = " << diff(N - 1) << "\n\n";

    std::cout << "CPU time (1 run)        = " << cpu_ms << " ms\n";
    std::cout << "GPU time (avg " << ITERS << " runs) = "
              << gpu_ms_avg << " ms\n";
    std::cout << "Speedup (CPU/GPU avg)   = "
              << (cpu_ms / gpu_ms_avg) << "x\n";

    // Освобождаем ресурсы
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockPrefix));

    return 0;
}
