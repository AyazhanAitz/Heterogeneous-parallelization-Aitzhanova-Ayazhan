%%writefile add_arrays.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call);           // вызываем CUDA-функцию и сохраняем результат \
    if (err != cudaSuccess) {           // если произошла ошибка \
        std::cerr << "CUDA ошибка: "    // выводим сообщение об ошибке \
                  << cudaGetErrorString(err) \
                  << " (файл " << __FILE__ \
                  << ", строка " << __LINE__ << ")\n"; \
        std::exit(1);                   // завершаем программу с ошибкой \
    } \
} while(0)

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // считаем глобальный индекс
    if (i < n) c[i] = a[i] + b[i];                   // поэлементно складываем
}

bool check_correctness(const std::vector<float>& a,
                       const std::vector<float>& b,
                       const std::vector<float>& c) {
    const float eps = 1e-5f;                         // допустимая погрешность
    for (size_t i = 0; i < a.size(); ++i) {          // идём по всем элементам
        float expected = a[i] + b[i];                // ожидаемое значение
        float diff = std::fabs(c[i] - expected);     // разница
        float tol = eps * (1.0f + std::fabs(expected)); // допуск
        if (diff > tol) return false;                // если слишком большая ошибка
    }
    return true;                                     // всё ок
}

float bench_add_once(const float* d_a,
                     const float* d_b,
                     float* d_c,
                     int n,
                     int blockSize,
                     int iters) {
    int gridSize = (n + blockSize - 1) / blockSize;  // считаем количество блоков

    cudaEvent_t start, stop;                         // события для времени
    CUDA_CHECK(cudaEventCreate(&start));             // создаём start
    CUDA_CHECK(cudaEventCreate(&stop));              // создаём stop

    CUDA_CHECK(cudaDeviceSynchronize());             // на всякий случай синхронизируемся

    CUDA_CHECK(cudaEventRecord(start));              // ставим старт времени
    for (int t = 0; t < iters; ++t) {                // повторяем несколько раз
        add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n); // запускаем kernel
    }
    CUDA_CHECK(cudaGetLastError());                  // проверяем запуск
    CUDA_CHECK(cudaEventRecord(stop));               // ставим конец времени
    CUDA_CHECK(cudaEventSynchronize(stop));          // ждём завершения

    float ms = 0.0f;                                 // сюда время
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // считаем мс

    CUDA_CHECK(cudaEventDestroy(start));             // удаляем start
    CUDA_CHECK(cudaEventDestroy(stop));              // удаляем stop

    return ms / iters;                               // возвращаем среднее время на 1 запуск
}

int main() {
    const int N = 1'000'000;                         // размер массивов
    const int iters = 200;                           // сколько раз запускать kernel для усреднения
    std::cout << "Размер массива: " << N << "\n";    // печатаем размер

    std::vector<float> h_a(N), h_b(N), h_c(N);       // массивы на CPU

    std::mt19937 rng(123);                           // генератор
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // распределение

    for (int i = 0; i < N; ++i) {                    // заполняем массивы
        h_a[i] = dist(rng);                          // случайное число
        h_b[i] = dist(rng);                          // случайное число
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr; // указатели GPU

    CUDA_CHECK(cudaMalloc(&d_a, (size_t)N * sizeof(float))); // выделяем память
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)N * sizeof(float))); // выделяем память
    CUDA_CHECK(cudaMalloc(&d_c, (size_t)N * sizeof(float))); // выделяем память

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice)); // копируем a
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice)); // копируем b

    std::vector<int> blockSizes = {64, 128, 256, 512}; // разные размеры блока (можно оставить минимум 3)

    std::cout << std::fixed << std::setprecision(4);  // форматируем вывод

    for (int bs : blockSizes) {                        // перебираем размеры блоков
        float ms = bench_add_once(d_a, d_b, d_c, N, bs, iters); // меряем время

        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат

        bool ok = check_correctness(h_a, h_b, h_c);    // проверяем правильность

        double bytes = (double)N * sizeof(float) * 3.0; // чтение a и b + запись c (в байтах)
        double gb_per_s = (bytes / (ms / 1000.0)) / 1e9; // примерная пропускная способность (GB/s)

        std::cout << "blockSize=" << std::setw(4) << bs
                  << " | time=" << std::setw(8) << ms << " ms"
                  << " | check=" << (ok ? "OK" : "FAIL")
                  << " | approx BW=" << gb_per_s << " GB/s\n";
    }

    CUDA_CHECK(cudaFree(d_a));                         // освобождаем память
    CUDA_CHECK(cudaFree(d_b));                         // освобождаем память
    CUDA_CHECK(cudaFree(d_c));                         // освобождаем память

    return 0;                                          // конец
}
