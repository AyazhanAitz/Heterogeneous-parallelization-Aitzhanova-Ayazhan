%%writefile main.cu
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

__global__ void mul_global(float* d_x, float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // вычисляем глобальный индекс элемента
    if (i < n) {                                   // проверяем выход за границы массива
        d_x[i] = d_x[i] * k;                       // умножаем элемент массива на число
    }
}

__global__ void mul_shared(float* d_x, float k, int n) {
    extern __shared__ float s[];                   // объявляем разделяемую память
    int tid = threadIdx.x;                         // получаем индекс потока в блоке
    int i = blockIdx.x * blockDim.x + tid;         // вычисляем глобальный индекс элемента

    if (i < n) {                                  // проверяем границы массива
        s[tid] = d_x[i];                           // копируем элемент из global в shared
    }

    __syncthreads();                               // синхронизируем все потоки блока

    if (i < n) {                                  // повторно проверяем границы
        s[tid] = s[tid] * k;                       // умножаем значение в shared memory
    }

    __syncthreads();                               // синхронизация перед записью

    if (i < n) {                                  // проверяем границы
        d_x[i] = s[tid];                           // записываем результат обратно в global
    }
}

float bench_global_once(float* d_x, float k, int n, int blockSize) {
    int gridSize = (n + blockSize - 1) / blockSize; // вычисляем количество блоков

    cudaEvent_t start, stop;                        // объявляем CUDA-события

    CUDA_CHECK(cudaEventCreate(&start));            // создаём событие начала
    CUDA_CHECK(cudaEventCreate(&stop));             // создаём событие окончания

    CUDA_CHECK(cudaEventRecord(start));             // фиксируем время старта
    mul_global<<<gridSize, blockSize>>>(d_x, k, n); // запускаем kernel с global memory
    CUDA_CHECK(cudaGetLastError());                 // проверяем ошибки запуска
    CUDA_CHECK(cudaEventRecord(stop));              // фиксируем время окончания
    CUDA_CHECK(cudaEventSynchronize(stop));         // ждём завершения kernel

    float ms = 0.0f;                                // переменная для времени выполнения
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // вычисляем время в мс

    CUDA_CHECK(cudaEventDestroy(start));            // удаляем событие начала
    CUDA_CHECK(cudaEventDestroy(stop));             // удаляем событие окончания

    return ms;                                      // возвращаем время выполнения
}

float bench_shared_once(float* d_x, float k, int n, int blockSize) {
    int gridSize = (n + blockSize - 1) / blockSize; // вычисляем количество блоков
    size_t sharedBytes = blockSize * sizeof(float); // размер shared memory на блок

    cudaEvent_t start, stop;                        // объявляем CUDA-события

    CUDA_CHECK(cudaEventCreate(&start));            // создаём событие начала
    CUDA_CHECK(cudaEventCreate(&stop));             // создаём событие окончания

    CUDA_CHECK(cudaEventRecord(start));             // фиксируем время старта
    mul_shared<<<gridSize, blockSize, sharedBytes>>>(d_x, k, n); // запускаем kernel с shared
    CUDA_CHECK(cudaGetLastError());                 // проверяем ошибки запуска
    CUDA_CHECK(cudaEventRecord(stop));              // фиксируем время окончания
    CUDA_CHECK(cudaEventSynchronize(stop));         // ждём завершения kernel

    float ms = 0.0f;                                // переменная для времени выполнения
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // считаем время в мс

    CUDA_CHECK(cudaEventDestroy(start));            // удаляем событие начала
    CUDA_CHECK(cudaEventDestroy(stop));             // удаляем событие окончания

    return ms;                                      // возвращаем время выполнения
}

bool check(const std::vector<float>& orig,
           const std::vector<float>& res,
           float k) {
    const float eps = 1e-5f;                        // допустимая погрешность
    for (size_t i = 0; i < orig.size(); ++i) {     // перебираем все элементы
        float expected = orig[i] * k;               // вычисляем ожидаемое значение
        float diff = std::fabs(res[i] - expected);  // считаем разницу
        float tol = eps * (1.0f + std::fabs(expected)); // допускаемую погрешность
        if (diff > tol) return false;                // если ошибка больше допустимой
    }
    return true;                                    // если всё корректно
}

int main() {
    const int N = 1'000'000;                        // размер массива
    const float k = 3.14159f;                       // множитель
    const int blockSize = 256;                      // размер блока CUDA

    std::cout << "Размер массива: " << N << "\n";  // вывод размера массива

    std::vector<float> h(N);                        // создаём массив на CPU

    std::mt19937 rng(123);                          // инициализируем генератор
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // диапазон значений

    for (int i = 0; i < N; ++i) {                   // заполняем массив
        h[i] = dist(rng);                           // случайными числами
    }

    std::vector<float> orig = h;                    // сохраняем исходный массив

    float* d = nullptr;                             // указатель на память GPU

    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));  // выделяем память на GPU

    CUDA_CHECK(cudaMemcpy(d, h.data(),              // копируем данные
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    float t_global = bench_global_once(d, k, N, blockSize); // замер global kernel

    std::vector<float> out_global(N);               // массив для результата
    CUDA_CHECK(cudaMemcpy(out_global.data(), d,     // копируем результат
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok1 = check(orig, out_global, k);           // проверяем корректность

    CUDA_CHECK(cudaMemcpy(d, h.data(),               // снова копируем исходные данные
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    float t_shared = bench_shared_once(d, k, N, blockSize); // замер shared kernel

    std::vector<float> out_shared(N);               // массив для результата
    CUDA_CHECK(cudaMemcpy(out_shared.data(), d,     // копируем результат
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok2 = check(orig, out_shared, k);           // проверяем корректность

    CUDA_CHECK(cudaFree(d));                         // освобождаем память GPU

    std::cout << std::fixed << std::setprecision(3); // форматируем вывод

    std::cout << "Global memory: " << t_global       // вывод времени global kernel
              << " ms, корректность: "
              << (ok1 ? "OK" : "ERROR") << "\n";

    std::cout << "Shared memory: " << t_shared       // вывод времени shared kernel
              << " ms, корректность: "
              << (ok2 ? "OK" : "ERROR") << "\n";

    return 0;                                       // завершение программы
}
