%%writefile prefix_scan.cu
#include <cuda_runtime.h>                     // CUDA Runtime API.
#include <iostream>                           // Вывод в консоль.
#include <vector>                             // Вектор на CPU.
#include <cstdlib>                            // Для exit().
#include <cmath>                              // Для fabs().

#define CUDA_CHECK(call) do {                 /* Макрос для проверки ошибок CUDA. */ \
    cudaError_t err = (call);                 /* Выполняем вызов CUDA. */ \
    if (err != cudaSuccess) {                 /* Если ошибка — выводим и выходим. */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1);                         /* Завершаем программу. */ \
    }                                         \
} while(0)

// ------------------------------
// ЯДРО: инклюзивная префиксная сумма (inclusive scan) внутри одного блока.
// Алгоритм: Hillis–Steele scan (O(n log n)).
// ------------------------------
__global__ void inclusive_scan_shared(const float* __restrict__ d_in,  // Входной массив на GPU.
                                      float* __restrict__ d_out,       // Выходной массив (префиксные суммы).
                                      int n)                           // Размер массива (n <= blockDim.x).
{
    extern __shared__ float s[];               // Разделяемая память: будем хранить текущие значения scan.

    int tid = threadIdx.x;                     // Индекс потока внутри блока.

    // 1) Загрузка данных из глобальной памяти в shared (быстрее для повторных операций).
    if (tid < n) {                             // Если поток соответствует элементу массива.
        s[tid] = d_in[tid];                    // Копируем элемент в shared memory.
    } else {
        s[tid] = 0.0f;                         // Для потоков "за пределами" n кладём 0, чтобы не мешали.
    }
    __syncthreads();                           // Ждём, пока все потоки загрузят s[].

    // 2) Hillis–Steele scan:
    // На шаге offset каждый поток добавляет значение элемента offset позади себя.
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {  // offset = 1,2,4,8,...
        float val = s[tid];                    // Берём текущее значение.
        if (tid >= offset) {                   // Если есть элемент слева на offset.
            val += s[tid - offset];            // Добавляем значение "offset позади".
        }
        __syncthreads();                       // Важно: перед записью синхронизируемся.
        s[tid] = val;                          // Записываем обновлённое значение.
        __syncthreads();                       // Ждём, пока все потоки обновят s[].
    }

    // 3) Запись результата в глобальную память.
    if (tid < n) {                             // Пишем только нужные элементы.
        d_out[tid] = s[tid];                   // Итоговая инклюзивная префиксная сумма.
    }
}

// CPU-версия для проверки (инклюзивная префиксная сумма).
std::vector<float> cpu_inclusive_scan(const std::vector<float>& a) {
    std::vector<float> out(a.size());          // Выходной массив на CPU.
    float running = 0.0f;                      // Накопленная сумма.
    for (size_t i = 0; i < a.size(); ++i) {    // Проходим по всем элементам.
        running += a[i];                       // Добавляем текущий элемент.
        out[i] = running;                      // Записываем префиксную сумму.
    }
    return out;                                // Возвращаем результат.
}

int main() {
    // -------- ТЕСТОВЫЙ МАССИВ --------
    // Берём размер <= 1024, чтобы всё поместилось в один блок (учебное ограничение).
    const int n = 16;                          // Можно поставить 128/256/512/1024 для тестов.
    std::vector<float> h_in(n);                // Входной массив на CPU.

    for (int i = 0; i < n; ++i) {              // Заполняем массив.
        h_in[i] = (i % 5) + 1.0f;              // Значения: 1..5 по кругу.
    }

    // CPU-эталон.
    std::vector<float> h_cpu = cpu_inclusive_scan(h_in);

    // -------- GPU ПАМЯТЬ --------
    float *d_in = nullptr;                     // Указатель на вход на GPU.
    float *d_out = nullptr;                    // Указатель на выход на GPU.

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));   // Выделяем память под вход.
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));  // Выделяем память под выход.

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // Копируем вход на GPU.

    // -------- ЗАПУСК ЯДРА --------
    int threads = 256;                         // Размер блока (можно 32/64/128/256/512/1024).
    // Важно: если threads < n — scan будет некорректен, поэтому threads >= n.
    if (threads < n) threads = n;              // Страхуемся: делаем threads не меньше n.

    int blocks = 1;                            // Учебная версия: один блок.
    size_t shared_bytes = threads * sizeof(float); // Shared memory: по одному float на поток.

    inclusive_scan_shared<<<blocks, threads, shared_bytes>>>(d_in, d_out, n); // Запуск ядра.
    CUDA_CHECK(cudaGetLastError());            // Проверка ошибок запуска.
    CUDA_CHECK(cudaDeviceSynchronize());       // Ждём завершения ядра.

    // -------- КОПИРОВАНИЕ РЕЗУЛЬТАТА --------
    std::vector<float> h_gpu(n);               // Буфер результата на CPU.
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // Копируем результат.

    // -------- ПРОВЕРКА КОРРЕКТНОСТИ --------
    float eps = 1e-5f;                         // Допуск для float.
    bool ok = true;                            // Флаг корректности.

    for (int i = 0; i < n; ++i) {              // Сравниваем поэлементно.
        float diff = std::fabs(h_cpu[i] - h_gpu[i]); // Разница CPU и GPU.
        if (diff > eps) {                      // Если превышает допуск — ошибка.
            ok = false;
            std::cout << "Mismatch at i=" << i
                      << " CPU=" << h_cpu[i]
                      << " GPU=" << h_gpu[i]
                      << " diff=" << diff << "\n";
            break;
        }
    }

    // -------- ВЫВОД --------
    std::cout << "Input:  ";
    for (int i = 0; i < n; ++i) std::cout << h_in[i] << " ";
    std::cout << "\nCPU:    ";
    for (int i = 0; i < n; ++i) std::cout << h_cpu[i] << " ";
    std::cout << "\nGPU:    ";
    for (int i = 0; i < n; ++i) std::cout << h_gpu[i] << " ";
    std::cout << "\n";

    if (ok) std::cout << "OK: result matches (eps=" << eps << ")\n";
    else    std::cout << "FAIL: mismatch\n";

    // -------- ОСВОБОЖДЕНИЕ ПАМЯТИ --------
    CUDA_CHECK(cudaFree(d_in));                // Освобождаем память входа.
    CUDA_CHECK(cudaFree(d_out));               // Освобождаем память выхода.

    return ok ? 0 : 1;                         // Возвращаем код завершения.
}
