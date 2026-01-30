%%writefile cuda_array.cu

#include <iostream>                 // Для вывода в консоль
#include <vector>                   // Для использования std::vector
#include <cuda_runtime.h>           // Основная библиотека CUDA
#include <chrono>                   // Для замера времени на CPU

// CUDA-ядро: каждый поток обрабатывает один элемент массива
__global__ void multiplyByTwo(int* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if (idx < n) {                                   // Проверка выхода за границы
        d_data[idx] *= 2;                            // Умножаем элемент на 2
    }
}

int main() {
    const int N = 1'000'000;                         // Размер массива
    const int size = N * sizeof(int);                // Размер массива в байтах

    std::vector<int> h_data(N);                      // Массив в памяти CPU (host)

    // Инициализация массива
    for (int i = 0; i < N; i++) {
        h_data[i] = i;                               // Значение равно индексу
    }

    int* d_data;                                     // Указатель на массив в GPU

    cudaMalloc((void**)&d_data, size);               // Выделение памяти на GPU

    cudaMemcpy(d_data, h_data.data(), size,
               cudaMemcpyHostToDevice);              // Копирование данных CPU → GPU

    int threadsPerBlock = 256;                       // Количество потоков в блоке
    int blocksPerGrid = (N + threadsPerBlock - 1)
                        / threadsPerBlock;           // Количество блоков

    // Начало замера времени GPU
    auto start = std::chrono::high_resolution_clock::now();

    // Запуск CUDA-ядра
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    cudaDeviceSynchronize();                          // Ожидание завершения GPU

    // Конец замера времени GPU
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_data.data(), d_data, size,
               cudaMemcpyDeviceToHost);               // Копирование данных GPU → CPU

    cudaFree(d_data);                                 // Освобождение памяти GPU

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Время выполнения (GPU CUDA): "
              << duration.count() << " мс" << std::endl;

    std::cout << "Первые 5 элементов массива после обработки:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << h_data[i] << " ";
    }

    return 0;                                         // Завершение программы
}
