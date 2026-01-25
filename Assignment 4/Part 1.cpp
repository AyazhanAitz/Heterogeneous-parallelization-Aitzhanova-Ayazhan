%%writefile sum_global_memory.cu
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// Макрос для проверки ошибок CUDA. Оборачиваем каждый вызов CUDA, и если он вернул ошибку — печатаем её и завершаем программу.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

// CUDA-ядро: каждый поток берёт один элемент массива и атомарно прибавляет его к общей сумме
__global__ void sumGlobalAtomic(const float* d_arr, int n, float* d_sum) {
    // blockIdx.x  — номер блока по оси X
    // blockDim.x  — размер блока (сколько потоков в блоке по X)
    // threadIdx.x — номер потока внутри блока
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // вычисляем глобальный индекс элемента массива
    if (i < n) { // проверяем границы (последний блок может быть неполным)
        atomicAdd(d_sum, d_arr[i]); // атомарно прибавляем d_arr[i] к общей сумме d_sum
    }
}

// CPU последовательная сумма
double cpuSum(const std::vector<float>& a) { // a передаём по ссылке (без копирования)
    double s = 0.0; // переменная суммы на CPU
    for (size_t i = 0; i < a.size(); ++i) s += a[i];  // проходимся циклом по всем элементам и накапливаем сумму
    return s;
}

int main() {
    const int N = 100000;         // размер массива 
    const int BLOCK = 256;        // число потоков в блоке
    const int GRID = (N + BLOCK - 1) / BLOCK; // число блоков: округление вверх
    const int ITERS = 30;         // количество повторов, чтобы усреднить время GPU

    // 1) Генерация данных на CPU
    std::vector<float> h(N);  // массив на CPU (host), размер N
    std::mt19937 rng(42); // генератор случайных чисел 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // равномерное распределение в диапазоне [0,1]
    for (int i = 0; i < N; ++i) h[i] = dist(rng); // заполняем массив, генерируем случайное число и записываем

    // 2) CPU сумма + время
    auto cpu_t1 = std::chrono::high_resolution_clock::now(); // старт таймера CPU
    double cpu_res = cpuSum(h); // считаем сумму последовательно на CPU
    auto cpu_t2 = std::chrono::high_resolution_clock::now(); // стоп таймера CPU
    double cpu_ms = std::chrono::duration < double, std::milli > (cpu_t2 - cpu_t1).count(); // время в миллисекундах

    // 3) Выделение памяти на GPU
    float * d_arr = nullptr; // указатель на массив на GPU
    float * d_sum = nullptr; // указатель на одну переменную суммы на GPU
    CUDA_CHECK(cudaMalloc( & d_arr, N * sizeof(float))); // выделяем память под N float на GPU
    CUDA_CHECK(cudaMalloc( & d_sum, sizeof(float))); // выделяем память под 1 float на GPU

    // 4) Копирование массива на GPU
    CUDA_CHECK(cudaMemcpy(d_arr, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 5) Создание CUDA событий для замера времени на GPU
    // События измеряют время на GPU (по сути kernel-time), без CPU overhead.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    // Прогрев (warm-up)
    // Первый запуск CUDA часто медленнее из-за инициализации контекста.
    // Поэтому делаем один запуск без учета в статистике.
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    sumGlobalAtomic<<<GRID, BLOCK>>>(d_arr, N, d_sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6) Замеры CUDA (несколько итераций)
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(start));
        sumGlobalAtomic<<<GRID, BLOCK>>>(d_arr, N, d_sum);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    float gpu_ms_avg = total_ms / ITERS;

    // 7) Получаем результат суммы с GPU на CPU
    float gpu_res = 0.0f; // переменная для результата на CPU
    CUDA_CHECK(cudaMemcpy(&gpu_res, d_sum, sizeof(float), cudaMemcpyDeviceToHost)); // device -> host

    // 8) Сравнение результатов CPU и GPU
    double abs_diff = std::abs(cpu_res - (double)gpu_res);
    double rel_diff = abs_diff / (std::abs(cpu_res) + 1e-12);

    // Настраиваем вывод чисел (фиксированный формат и 6 знаков после запятой)
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    // Печатаем результаты и погрешность
    std::cout << "N = " << N << "\n";
    std::cout << "CPU sum           = " << cpu_res << "\n";
    std::cout << "GPU sum (atomic)  = " << gpu_res << "\n";
    std::cout << "Abs diff          = " << abs_diff << "\n";
    std::cout << "Rel diff          = " << rel_diff << "\n\n";

    std::cout << "CPU time (1 run)        = " << cpu_ms << " ms\n";
    std::cout << "GPU time (avg " << ITERS << " runs) = " << gpu_ms_avg << " ms\n";
    std::cout << "Speedup (CPU/GPU avg)   = " << (cpu_ms / gpu_ms_avg) << "x\n";

    // 9) Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaFree(d_sum));

    return 0;
}
