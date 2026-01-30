%%writefile hybrid_cpu_gpu.cu

#include <iostream>                 // Для вывода в консоль
#include <vector>                   // Для работы с массивом через std::vector
#include <cuda_runtime.h>           // Для функций CUDA (cudaMalloc, cudaMemcpyAsync и т.д.)
#include <omp.h>                    // Для OpenMP параллельных циклов
#include <chrono>                   // Для замера времени выполнения

// CUDA-ядро: умножает элементы массива на 2
__global__ void multiplyByTwoKernel(int* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Считаем глобальный индекс потока
    if (idx < n) {                                   // Проверяем границы массива
        d_data[idx] *= 2;                            // Умножаем элемент на 2
    }
}

int main() {
    const int N = 1'000'000;                         // Размер массива (1 млн)
    const int sizeBytes = N * sizeof(int);           // Размер массива в байтах

    int* h_data = nullptr;                           // Указатель на массив в CPU (host)
    cudaMallocHost((void**)&h_data, sizeBytes);      // Выделяем pinned-память (быстрее и поддерживает async memcpy)

    for (int i = 0; i < N; i++) {                    // Заполняем массив данными
        h_data[i] = i;                               // Значение равно индексу
    }

    const int mid = N / 2;                           // Индекс разделения массива (половина)
    const int gpuCount = N - mid;                    // Количество элементов для GPU (вторая половина)
    const int gpuBytes = gpuCount * sizeof(int);     // Размер второй половины в байтах

    int* d_data = nullptr;                           // Указатель на массив в GPU
    cudaMalloc((void**)&d_data, gpuBytes);           // Выделяем память на GPU под вторую половину

    cudaStream_t stream;                             // CUDA stream для асинхронных операций
    cudaStreamCreate(&stream);                       // Создаём stream

    int threadsPerBlock = 256;                       // Потоков в одном блоке
    int blocksPerGrid = (gpuCount + threadsPerBlock - 1) / threadsPerBlock; // Сколько блоков нужно

    auto start = std::chrono::high_resolution_clock::now(); // Старт общего времени гибридной обработки

    // 1) Асинхронно копируем вторую половину массива CPU -> GPU (через stream)
    cudaMemcpyAsync(d_data, h_data + mid, gpuBytes,
                    cudaMemcpyHostToDevice, stream);

    // 2) Асинхронно запускаем CUDA-ядро в том же stream (после копирования)
    multiplyByTwoKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, gpuCount);

    // 3) Асинхронно копируем обработанную вторую половину обратно GPU -> CPU
    cudaMemcpyAsync(h_data + mid, d_data, gpuBytes,
                    cudaMemcpyDeviceToHost, stream);

    // 4) Одновременно (пока GPU занят) обрабатываем первую половину на CPU через OpenMP
    #pragma omp parallel for
    for (int i = 0; i < mid; i++) {                  // Проходим по первой половине массива
        h_data[i] *= 2;                              // Умножаем на 2 (CPU-часть)
    }

    // 5) Дожидаемся завершения всех GPU операций (копирование + ядро + копирование назад)
    cudaStreamSynchronize(stream);                   // Гарантируем, что GPU часть завершена

    auto end = std::chrono::high_resolution_clock::now(); // Конец общего времени

    std::chrono::duration<double, std::milli> total = end - start; // Считаем время в мс

    std::cout << "Общее время гибридной обработки (CPU+GPU): "
              << total.count() << " мс" << std::endl; // Выводим итоговое время

    std::cout << "Проверка первых 5 элементов (CPU часть): ";      // Печать начала массива
    for (int i = 0; i < 5; i++) {                                  // Берём 5 элементов
        std::cout << h_data[i] << " ";                             // Печатаем значения
    }
    std::cout << std::endl;                                        // Перевод строки

    std::cout << "Проверка 5 элементов после середины (GPU часть): "; // Печать части после mid
    for (int i = mid; i < mid + 5; i++) {                           // Берём 5 элементов со второй половины
        std::cout << h_data[i] << " ";                              // Печатаем значения
    }
    std::cout << std::endl;                                         // Перевод строки

    cudaStreamDestroy(stream);                      // Удаляем stream
    cudaFree(d_data);                               // Освобождаем память на GPU
    cudaFreeHost(h_data);                           // Освобождаем pinned память на CPU

    return 0;                                       // Завершаем программу
}
