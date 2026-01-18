%%writefile tune_add.cu
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

// CUDA-ядро для поэлементного сложения массивов.
// Каждый поток обрабатывает ровно один элемент массивов a и b.
// Результат записывается в массив c.
__global__ void add_kernel(const float* a,
                           const float* b,
                           float* c,
                           int n) {

    // Вычисляем глобальный индекс элемента массива,
    // который соответствует текущему потоку.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за границы массива.
    // Это необходимо, так как количество потоков может быть больше n.
    if (i < n) {

        // Складываем соответствующие элементы массивов
        // и записываем результат в выходной массив.
        c[i] = a[i] + b[i];
    }
}

// Функция benchmark_add измеряет время выполнения CUDA-ядра
// при заданном размере блока потоков.
// Для повышения точности ядро запускается несколько раз,
// после чего считается среднее время одного запуска.
float benchmark_add(const float* d_a,
                    const float* d_b,
                    float* d_c,
                    int n,
                    int blockSize,
                    int iters) {

    // Вычисляем размер сетки (количество блоков).
    // Используется округление вверх, чтобы покрыть весь массив.
    int gridSize = (n + blockSize - 1) / blockSize;

    // Создаём CUDA-события для измерения времени выполнения.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Синхронизируем устройство, чтобы все предыдущие операции завершились.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Фиксируем момент начала измерения времени.
    CUDA_CHECK(cudaEventRecord(start));

    // Запускаем ядро несколько раз подряд.
    // Это позволяет уменьшить влияние случайных колебаний времени.
    for (int t = 0; t < iters; ++t) {

        // Запуск CUDA-ядра с заданной конфигурацией сетки и блоков.
        add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }

    // Проверяем, что запуск ядра прошёл без ошибок.
    CUDA_CHECK(cudaGetLastError());

    // Фиксируем момент окончания измерения времени.
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Переменная для хранения измеренного времени в миллисекундах.
    float ms = 0.0f;

    // Вычисляем время между событиями start и stop.
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Освобождаем CUDA-события.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Возвращаем среднее время одного запуска ядра.
    return ms / iters;
}

// Функция проверки корректности результата.
// Она сравнивает результат работы GPU с ожидаемым результатом на CPU.
bool check(const std::vector<float>& a,
           const std::vector<float>& b,
           const std::vector<float>& c) {

    // Допустимая погрешность для чисел с плавающей точкой.
    const float eps = 1e-5f;

    // Последовательно проверяем каждый элемент массива.
    for (size_t i = 0; i < a.size(); ++i) {

        // Вычисляем ожидаемое значение.
        float expected = a[i] + b[i];

        // Находим разницу между GPU-результатом и ожидаемым значением.
        float diff = std::fabs(c[i] - expected);

        // Если разница превышает допустимую погрешность,
        // считаем результат некорректным.
        if (diff > eps * (1.0f + std::fabs(expected))) {
            return false;
        }
    }

    // Если все элементы совпали в пределах погрешности,
    // возвращаем true.
    return true;
}

int main() {

    // Размер массивов для обработки на GPU.
    const int N = 1'000'000;

    // Количество повторов ядра для усреднения времени выполнения.
    const int iters = 300;

    std::cout << "Размер массива N = " << N << "\n";

    // Создаём входные и выходной массивы в оперативной памяти.
    std::vector<float> h_a(N), h_b(N), h_c(N);

    // Инициализируем генератор случайных чисел.
    std::mt19937 rng(123);

    // Определяем диапазон случайных значений.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Заполняем входные массивы случайными числами.
    for (int i = 0; i < N; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    // Объявляем указатели на память GPU.
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // Выделяем память на GPU для всех массивов.
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // Копируем входные данные с CPU на GPU.
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Набор размеров блока потоков для тестирования.
    // Обычно выбираются степени двойки.
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};

    // Выбираем явно неоптимальную конфигурацию —
    // слишком маленький размер блока.
    int block_bad = 32;

    int block_best = blockSizes[0];
    float best_time = 1e30f;

    std::cout << "\nТестирование различных размеров блока:\n";

    // Перебираем все размеры блока и измеряем время выполнения.
    for (int bs : blockSizes) {

        // Замеряем среднее время выполнения ядра.
        float ms = benchmark_add(d_a, d_b, d_c, N, bs, iters);

        // Копируем результат обратно на CPU.
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c,
                              N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Проверяем корректность вычислений.
        bool ok = check(h_a, h_b, h_c);

        // Выводим информацию по текущей конфигурации.
        std::cout << "blockSize = " << std::setw(4) << bs
                  << " | avg_time = " << std::fixed << std::setprecision(4)
                  << ms << " ms"
                  << " | check = " << (ok ? "OK" : "FAIL") << "\n";

        // Запоминаем лучшую конфигурацию по времени.
        if (ok && ms < best_time) {
            best_time = ms;
            block_best = bs;
        }
    }

    // Отдельно измеряем время для неоптимальной конфигурации.
    float time_bad = benchmark_add(d_a, d_b, d_c, N, block_bad, iters);

    // Измеряем время для оптимальной конфигурации.
    float time_best = benchmark_add(d_a, d_b, d_c, N, block_best, iters);

    std::cout << "\nСравнение конфигураций:\n";
    std::cout << "Неоптимальная: blockSize = " << block_bad
              << ", avg_time = " << time_bad << " ms\n";
    std::cout << "Оптимальная:   blockSize = " << block_best
              << ", avg_time = " << time_best << " ms\n";

    // Вычисляем ускорение, полученное за счёт оптимизации.
    std::cout << "Ускорение = " << (time_bad / time_best) << "x\n";

    // Освобождаем память GPU.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
