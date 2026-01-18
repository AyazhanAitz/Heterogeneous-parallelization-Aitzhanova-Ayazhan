%%writefile coalescing_demo.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Макрос CUDA_CHECK используется для проверки ошибок после вызовов CUDA-функций.
// Если функция вернула ошибку, программа выводит сообщение и завершает выполнение.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA ошибка: " \
                  << cudaGetErrorString(err) \
                  << " (файл " << __FILE__ \
                  << ", строка " << __LINE__ << ")\n"; \
        std::exit(1); \
    } \
} while (0)

// Данное CUDA-ядро демонстрирует КОАЛЕСЦИРОВАННЫЙ доступ к глобальной памяти.
// Каждый поток обрабатывает элемент с индексом i и обращается к памяти подряд.
// Это позволяет GPU объединять обращения потоков варпа в одну транзакцию памяти.
__global__ void kernel_coalesced(const float* in, float* out, float k, int n) {

    // Вычисляем глобальный индекс элемента массива для текущего потока
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, чтобы индекс не выходил за границы массива
    if (i < n) {

        // Читаем элемент in[i] и записываем результат в out[i]
        // Доступ последовательный, поэтому он коалесцированный
        out[i] = in[i] * k;
    }
}

// Данное CUDA-ядро демонстрирует НЕКОАЛЕСЦИРОВАННЫЙ доступ к глобальной памяти.
// Потоки одного варпа обращаются к памяти с большим шагом (stride),
// из-за чего обращения не объединяются и выполняются медленнее.
__global__ void kernel_noncoalesced(const float* in,
                                   float* out,
                                   float k,
                                   int n,
                                   int stride) {

    // Вычисляем логический индекс элемента
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем выход за границы массива
    if (i < n) {

        // Вычисляем новый индекс с использованием шага stride
        // Такой доступ приводит к "прыжкам" по памяти
        int j = (i * stride) % n;

        // Читаем элемент из далёкого адреса памяти
        // Такой доступ не является коалесцированным
        out[i] = in[j] * k;
    }
}

// Функция для измерения времени выполнения CUDA-ядра.
// Используется cudaEvent, что позволяет точно измерять время на GPU.
float benchmark_kernel(bool coalesced,
                       const float* d_in,
                       float* d_out,
                       float k,
                       int n,
                       int blockSize,
                       int iters,
                       int stride) {

    // Вычисляем количество блоков, необходимое для обработки всего массива
    int gridSize = (n + blockSize - 1) / blockSize;

    // Создаём CUDA-события для замера времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Синхронизируем устройство перед началом измерений
    CUDA_CHECK(cudaDeviceSynchronize());

    // Записываем момент начала измерения времени
    CUDA_CHECK(cudaEventRecord(start));

    // Запускаем ядро несколько раз для усреднения результата
    for (int i = 0; i < iters; ++i) {

        if (coalesced) {
            // Запуск ядра с коалесцированным доступом
            kernel_coalesced<<<gridSize, blockSize>>>(d_in, d_out, k, n);
        } else {
            // Запуск ядра с некоалесцированным доступом
            kernel_noncoalesced<<<gridSize, blockSize>>>(d_in, d_out, k, n, stride);
        }
    }

    // Проверяем, что ядро запустилось без ошибок
    CUDA_CHECK(cudaGetLastError());

    // Фиксируем момент окончания измерения времени
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Переменная для хранения измеренного времени
    float ms = 0.0f;

    // Вычисляем прошедшее время в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Освобождаем CUDA-события
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Возвращаем среднее время одного запуска ядра
    return ms / iters;
}

// Проверка корректности для коалесцированного доступа
bool check_coalesced(const std::vector<float>& in,
                     const std::vector<float>& out,
                     float k) {

    const float eps = 1e-5f;

    for (size_t i = 0; i < in.size(); ++i) {
        float expected = in[i] * k;
        if (std::fabs(out[i] - expected) > eps) return false;
    }
    return true;
}

// Проверка корректности для некоалесцированного доступа
bool check_noncoalesced(const std::vector<float>& in,
                        const std::vector<float>& out,
                        float k,
                        int stride) {

    const float eps = 1e-5f;
    int n = static_cast<int>(in.size());

    for (int i = 0; i < n; ++i) {
        int j = (i * stride) % n;
        float expected = in[j] * k;
        if (std::fabs(out[i] - expected) > eps) return false;
    }
    return true;
}

int main() {

    // Размер массива, который обрабатывается на GPU
    const int N = 1'000'000;

    // Множитель для простой арифметической операции
    const float k = 2.0f;

    // Размер блока потоков CUDA
    const int blockSize = 256;

    // Количество повторных запусков для усреднения времени
    const int iters = 300;

    // Шаг для демонстрации некоалесцированного доступа
    const int stride = 97;

    std::cout << "Размер массива: " << N << "\n";

    // Создаём входной и выходной массивы на CPU
    std::vector<float> h_in(N), h_out(N);

    // Генерируем случайные данные
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        h_in[i] = dist(rng);
    }

    // Выделяем память на GPU
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Копируем входные данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Измеряем время коалесцированного доступа
    float t_coalesced = benchmark_kernel(true, d_in, d_out, k,
                                         N, blockSize, iters, stride);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok1 = check_coalesced(h_in, h_out, k);

    // Измеряем время некоалесцированного доступа
    float t_noncoalesced = benchmark_kernel(false, d_in, d_out, k,
                                            N, blockSize, iters, stride);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok2 = check_noncoalesced(h_in, h_out, k, stride);

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Coalesced доступ:     " << t_coalesced << " ms | "
              << (ok1 ? "OK" : "ERROR") << "\n";
    std::cout << "Non-coalesced доступ: " << t_noncoalesced << " ms | "
              << (ok2 ? "OK" : "ERROR") << "\n";

    std::cout << "Отношение времени (Non / Coalesced): "
              << t_noncoalesced / t_coalesced << "x\n";

    std::cout << "Вывод: коалесцированный доступ к памяти значительно быстрее,\n";
    std::cout << "поскольку обращения потоков объединяются в меньшее число\n";
    std::cout << "транзакций глобальной памяти.\n";

    return 0;
}
