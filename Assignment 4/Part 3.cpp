%%writefile hybrid_cpu_gpu.cu
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

// Функция обработки одного элемента, используется одинаково на CPU и на GPU
// __host__ __device__ позволяет вызывать эту функцию и на процессоре, и внутри CUDA kernel
__host__ __device__ inline float processValue(float x) {
    // Вычисление без ветвлений, чтобы поведение CPU и GPU было максимально одинаковым
    // sqrtf используется для float, чтобы не было лишних преобразований типов
    return sqrtf(x * x + 1.0f) + x * 0.5f;
}

// CUDA kernel: обрабатывает часть массива, начиная с startIdx, длиной count
// Результат записывается в d_out по тем же индексам, что и вход
__global__ void processKernel(const float* d_in, float* d_out, int startIdx, int count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;   // индекс потока внутри обрабатываемого диапазона
    int i = startIdx + t;                            // глобальный индекс элемента в исходном массиве
    if (t < count) {                                 // границы диапазона, последний блок может быть неполным
        d_out[i] = processValue(d_in[i]);            // вычисление функции обработки и запись результата
    }
}

// CPU обработка диапазона массива [start, start+count)
// Вектор out должен быть заранее нужного размера, чтобы запись по индексу была корректной
void processCpuRange(const std::vector<float>& in, std::vector<float>& out, int start, int count) {
    for (int k = 0; k < count; ++k) {                // перебор элементов диапазона
        int i = start + k;                           // глобальный индекс элемента
        out[i] = processValue(in[i]);                // вычисление и запись результата
    }
}

int main() {
    const int N = 1'000'000;                         // размер массива для эксперимента
    const int BLOCK = 256;                           // размер блока CUDA, типичное значение для T4
    const int ITERS = 20;                            // количество повторов для усреднения времени

    std::vector<float> h_in(N);                      // входной массив на CPU
    std::vector<float> h_cpu(N);                     // результат CPU-only
    std::vector<float> h_gpu(N);                     // результат GPU-only
    std::vector<float> h_hybrid(N);                  // результат hybrid

    std::mt19937 rng(42);                            // генератор случайных чисел с фиксированным seed
    std::uniform_real_distribution<float> dist(0.0f, 10.0f); // диапазон входных значений
    for (int i = 0; i < N; ++i) {                    // заполнение входного массива
        h_in[i] = dist(rng);                         // генерация очередного значения
    }

    auto cpu_t1 = std::chrono::high_resolution_clock::now(); // старт измерения CPU-only
    processCpuRange(h_in, h_cpu, 0, N);               // обработка всего массива на CPU
    auto cpu_t2 = std::chrono::high_resolution_clock::now(); // конец измерения CPU-only
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t2 - cpu_t1).count(); // время CPU-only в мс

    float* d_in = nullptr;                            // указатель на входной массив на GPU
    float* d_out = nullptr;                           // указатель на выходной массив на GPU
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));  // выделение памяти под вход на GPU
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float))); // выделение памяти под выход на GPU

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // копирование входа CPU->GPU

    cudaEvent_t gs, ge;                               // события CUDA для замеров времени GPU-only kernel
    CUDA_CHECK(cudaEventCreate(&gs));                 // создание события начала
    CUDA_CHECK(cudaEventCreate(&ge));                 // создание события конца

    int grid_all = (N + BLOCK - 1) / BLOCK;           // количество блоков для обработки всего массива

    processKernel<<<grid_all, BLOCK>>>(d_in, d_out, 0, N); // прогрев GPU: запуск kernel на весь массив
    CUDA_CHECK(cudaGetLastError());                   // проверка ошибок запуска kernel
    CUDA_CHECK(cudaDeviceSynchronize());              // ожидание завершения прогрева

    float gpu_total_ms = 0.0f;                        // накопитель времени GPU-only по итерациям
    for (int it = 0; it < ITERS; ++it) {              // повторные замеры для усреднения
        CUDA_CHECK(cudaEventRecord(gs));              // отметка времени start на GPU
        processKernel<<<grid_all, BLOCK>>>(d_in, d_out, 0, N); // запуск GPU-only обработки
        CUDA_CHECK(cudaEventRecord(ge));              // отметка времени stop на GPU

        CUDA_CHECK(cudaGetLastError());               // проверка ошибок запуска kernel
        CUDA_CHECK(cudaEventSynchronize(ge));         // ожидание завершения kernel по событию stop

        float ms = 0.0f;                              // время одной итерации
        CUDA_CHECK(cudaEventElapsedTime(&ms, gs, ge)); // вычисление времени между start и stop
        gpu_total_ms += ms;                           // суммирование времени
    }
    float gpu_ms_avg = gpu_total_ms / ITERS;          // среднее время GPU-only kernel

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost)); // копирование результата GPU->CPU

    int mid = N / 2;                                  // точка разделения массива на две части
    int gpu_count = N - mid;                          // размер второй части, которую обрабатывает GPU

    cudaEvent_t hs, he;                               // события CUDA для замера времени kernel во время hybrid
    CUDA_CHECK(cudaEventCreate(&hs));                 // событие начала hybrid-kernel
    CUDA_CHECK(cudaEventCreate(&he));                 // событие конца hybrid-kernel

    int grid_half = (gpu_count + BLOCK - 1) / BLOCK;  // количество блоков для обработки второй половины

    processCpuRange(h_in, h_hybrid, 0, mid);          // прогрев hybrid: первая половина на CPU
    processKernel<<<grid_half, BLOCK>>>(d_in, d_out, mid, gpu_count); // прогрев hybrid: вторая половина на GPU
    CUDA_CHECK(cudaGetLastError());                   // проверка ошибок запуска
    CUDA_CHECK(cudaDeviceSynchronize());              // ожидание завершения kernel
    CUDA_CHECK(cudaMemcpy(h_hybrid.data() + mid, d_out + mid, gpu_count * sizeof(float), cudaMemcpyDeviceToHost)); // перенос второй половины результата

    float hybrid_total_ms = 0.0f;                     // накопитель полного времени hybrid по итерациям
    for (int it = 0; it < ITERS; ++it) {              // повторные замеры hybrid
        auto hcpu_t1 = std::chrono::high_resolution_clock::now(); // старт CPU-части hybrid
        processCpuRange(h_in, h_hybrid, 0, mid);      // CPU обрабатывает первую половину
        auto hcpu_t2 = std::chrono::high_resolution_clock::now(); // конец CPU-части hybrid
        double cpu_part_ms = std::chrono::duration<double, std::milli>(hcpu_t2 - hcpu_t1).count(); // время CPU-части hybrid

        CUDA_CHECK(cudaEventRecord(hs));              // старт замера GPU-kernel в hybrid
        processKernel<<<grid_half, BLOCK>>>(d_in, d_out, mid, gpu_count); // GPU обрабатывает вторую половину
        CUDA_CHECK(cudaEventRecord(he));              // конец замера GPU-kernel в hybrid

        CUDA_CHECK(cudaGetLastError());               // проверка ошибок запуска kernel
        CUDA_CHECK(cudaEventSynchronize(he));         // ожидание завершения kernel

        float gpu_part_ms = 0.0f;                     // время GPU-kernel в hybrid
        CUDA_CHECK(cudaEventElapsedTime(&gpu_part_ms, hs, he)); // вычисление времени kernel

        auto hcopy_t1 = std::chrono::high_resolution_clock::now(); // старт копирования второй половины результата
        CUDA_CHECK(cudaMemcpy(h_hybrid.data() + mid, d_out + mid, gpu_count * sizeof(float), cudaMemcpyDeviceToHost)); // копирование второй половины GPU->CPU
        auto hcopy_t2 = std::chrono::high_resolution_clock::now(); // конец копирования результата
        double copy_part_ms = std::chrono::duration<double, std::milli>(hcopy_t2 - hcopy_t1).count(); // время копирования

        hybrid_total_ms += (float)(cpu_part_ms + (double)gpu_part_ms + copy_part_ms); // суммарное время hybrid итерации
    }
    float hybrid_ms_avg = hybrid_total_ms / ITERS;    // среднее время hybrid

    auto absdiff = [&](const std::vector<float>& a, const std::vector<float>& b, int idx) {
        return std::abs((double)a[idx] - (double)b[idx]); // абсолютная разница для проверки совпадения
    };

    int i0 = 0;                                       // проверка в начале массива
    int i1 = N / 3;                                   // проверка в середине массива
    int i2 = N - 1;                                   // проверка в конце массива

    double d_cpu_gpu_0 = absdiff(h_cpu, h_gpu, i0);    // CPU vs GPU, индекс 0
    double d_cpu_gpu_1 = absdiff(h_cpu, h_gpu, i1);    // CPU vs GPU, индекс N/3
    double d_cpu_gpu_2 = absdiff(h_cpu, h_gpu, i2);    // CPU vs GPU, индекс N-1

    double d_cpu_hy_0 = absdiff(h_cpu, h_hybrid, i0);  // CPU vs Hybrid, индекс 0
    double d_cpu_hy_1 = absdiff(h_cpu, h_hybrid, i1);  // CPU vs Hybrid, индекс N/3
    double d_cpu_hy_2 = absdiff(h_cpu, h_hybrid, i2);  // CPU vs Hybrid, индекс N-1

    std::cout.setf(std::ios::fixed);                   // фиксированный формат вывода чисел
    std::cout.precision(6);                            // 6 знаков после запятой

    std::cout << "N = " << N << "\n";                  // печать размера массива
    std::cout << "Split: CPU [0.." << (mid - 1)        // печать границ разделения
              << "], GPU [" << mid << ".." << (N - 1) << "]\n\n";

    std::cout << "Check CPU vs GPU abs diff:  [0]=" << d_cpu_gpu_0
              << "  [N/3]=" << d_cpu_gpu_1
              << "  [N-1]=" << d_cpu_gpu_2 << "\n";

    std::cout << "Check CPU vs HYB abs diff:  [0]=" << d_cpu_hy_0
              << "  [N/3]=" << d_cpu_hy_1
              << "  [N-1]=" << d_cpu_hy_2 << "\n\n";

    std::cout << "CPU time (1 run)                    = " << cpu_ms << " ms\n";
    std::cout << "GPU time (avg " << ITERS << " kernel runs)       = " << gpu_ms_avg << " ms\n";
    std::cout << "Hybrid time (avg " << ITERS << " runs, incl copy) = " << hybrid_ms_avg << " ms\n\n";

    std::cout << "Speedup GPU vs CPU    = " << (cpu_ms / gpu_ms_avg) << "x\n";
    std::cout << "Speedup Hybrid vs CPU = " << (cpu_ms / hybrid_ms_avg) << "x\n";

    CUDA_CHECK(cudaEventDestroy(gs));                  // удаление события начала GPU-only
    CUDA_CHECK(cudaEventDestroy(ge));                  // удаление события конца GPU-only
    CUDA_CHECK(cudaEventDestroy(hs));                  // удаление события начала hybrid-kernel
    CUDA_CHECK(cudaEventDestroy(he));                  // удаление события конца hybrid-kernel
    CUDA_CHECK(cudaFree(d_in));                        // освобождение памяти входа на GPU
    CUDA_CHECK(cudaFree(d_out));                       // освобождение памяти выхода на GPU

    return 0;                                          // успешное завершение программы
}
