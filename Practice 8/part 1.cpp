%%writefile openmp_array.cpp
#include <iostream>    
#include <vector>      
#include <omp.h>       
#include <chrono>      

int main() {
    const int N = 1'000'000; // Размер массива (1 миллион элементов)

    std::vector<int> data(N); // Создаём массив из N целых чисел

    // Инициализация массива начальными значениями
    for (int i = 0; i < N; i++) {
        data[i] = i; // Каждый элемент равен своему индексу
    }

    // Фиксируем время начала обработки
    auto start = std::chrono::high_resolution_clock::now();

    // Параллельный цикл OpenMP
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        data[i] *= 2; // Умножаем каждый элемент массива на 2
    }

    // Фиксируем время окончания обработки
    auto end = std::chrono::high_resolution_clock::now();

    // Вычисляем длительность выполнения в миллисекундах
    std::chrono::duration<double, std::milli> duration = end - start;

    // Выводим время выполнения
    std::cout << "Время выполнения (CPU + OpenMP): "
              << duration.count() << " мс" << std::endl;

    // Выводим несколько элементов для проверки корректности
    std::cout << "Первые 5 элементов массива после обработки:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << data[i] << " ";
    }

    return 0; // Завершение программы
}
