%%writefile main.cu
#include <cuda_runtime.h>       
#include <iostream>             
#include <vector>               
#include <random>               
#include <fstream>              

// Количество потоков в одном блоке (blockDim.x)
const int BLOCK_SIZE = 256;

/*
 * Функция checkCuda
 * Назначение: проверить результат CUDA-вызова и завершить программу, если произошла ошибка.
 */
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {                                        // если код ошибки не "успех"
        std::cerr << "CUDA error: " << msg << " : "                  // выводим сообщение
                  << cudaGetErrorString(err) << std::endl;           // и текст ошибки CUDA
        std::exit(1);                                                // аварийно завершаем программу
    }
}

// -------------------- Ядро редукции (GLOBAL ONLY) --------------------

/*
 * Ядро reduceGlobalOnly
 * Идея: каждый поток считает свою частичную сумму, затем делает atomicAdd в global результат.
 * Минус: атомик выполняется от КАЖДОГО потока => сильная конкуренция потоков.
 */
__global__ void reduceGlobalOnly(const int* data, int n, unsigned long long* result) {
    unsigned long long localSum = 0ULL;                              // локальная сумма потока (в регистрах/локальной памяти)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;                 // глобальный индекс потока
    int stride = blockDim.x * gridDim.x;                             // шаг "сеткой": сколько потоков всего в grid

    // Каждый поток проходит по массиву: idx, idx+stride, idx+2*stride ...
    // Так мы равномерно распределяем работу между потоками.
    for (int i = idx; i < n; i += stride) {                          // пока индекс не выйдет за пределы массива
        localSum += static_cast<unsigned long long>(data[i]);        // прибавляем элемент к локальной сумме
    }

    atomicAdd(result, localSum);                                     // атомарно добавляем локальную сумму в общий результат (global)
}

// -------------------- Ядро редукции (GLOBAL + SHARED) --------------------

/*
 * Ядро reduceGlobalShared
 * Идея:
 * 1) каждый поток считает localSum
 * 2) складываем localSum всех потоков блока через shared memory
 * 3) только thread 0 делает atomicAdd(итог блока) в global result
 * Плюс: атомиков в ~BLOCK_SIZE раз меньше, обычно быстрее.
 */
__global__ void reduceGlobalShared(const int* data, int n, unsigned long long* result) {
    __shared__ unsigned long long shared[BLOCK_SIZE];                // shared-память: видна всем потокам блока и очень быстрая

    unsigned long long localSum = 0ULL;                              // локальная сумма для текущего потока

    int idx = blockIdx.x * blockDim.x + threadIdx.x;                 // глобальный индекс потока
    int stride = blockDim.x * gridDim.x;                             // шаг сетки

    // -------- Шаг 1: каждый поток считает частичную сумму --------
    for (int i = idx; i < n; i += stride) {                          // идём по массиву с шагом stride
        localSum += static_cast<unsigned long long>(data[i]);        // добавляем очередной элемент в localSum
    }

    // -------- Шаг 2: кладём localSum в shared memory --------
    shared[threadIdx.x] = localSum;                                  // каждый поток пишет свою сумму в shared по индексу threadIdx.x

    __syncthreads();                                                 // барьер: ждём, пока ВСЕ потоки запишут shared[]

    // -------- Шаг 3: параллельная редукция внутри блока --------
    // blockDim.x/2: сначала суммируем пары (0+128, 1+129, ...) если BLOCK_SIZE=256
    // затем (0+64, 1+65, ...) и так далее, пока не останется один элемент shared[0]
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                   // s уменьшаем вдвое: 128, 64, 32, ...
        if (threadIdx.x < s) {                                       // только первые s потоков участвуют в этом шаге
            shared[threadIdx.x] += shared[threadIdx.x + s];          // прибавляем "пару" из второй половины
        }
        __syncthreads();                                             // барьер: чтобы следующий шаг начинался после завершения текущего
    }

    // -------- Шаг 4: один atomicAdd на блок --------
    if (threadIdx.x == 0) {                                          // только нулевой поток блока
        atomicAdd(result, shared[0]);                                // добавляет итоговую сумму блока в общий результат в global memory
    }
}

// -------------------- Генерация массива на CPU --------------------

/*
 * Функция generateData
 * Назначение: создать вектор из n случайных чисел в диапазоне [1, 100].
 */
std::vector<int> generateData(int n) {
    std::vector<int> v(n);                                           // создаём вектор размера n

    std::mt19937 gen(42);                                            // генератор случайных чисел (seed=42 для воспроизводимости)
    std::uniform_int_distribution<int> dist(1, 100);                 // равномерное распределение от 1 до 100

    for (int i = 0; i < n; i++) {                                    // заполняем все элементы
        v[i] = dist(gen);                                            // получаем случайное число и записываем в v[i]
    }

    return v;                                                        // возвращаем заполненный вектор
}


/*
 * Функция main
 * Назначение: прогнать тесты на 3 размерах массивов, измерить время 2 вариантов редукции,
 * и сохранить результаты в results.csv.
 */
int main() {
    std::ofstream csv("results.csv");                                // создаём/открываем файл results.csv для записи
    csv << "N,global_only_ms,global_shared_ms\n";                    // пишем заголовок CSV

    std::vector<int> sizes = {10000, 100000, 1000000};               // размеры массивов по заданию

    for (int n : sizes) {                                            // цикл по каждому размеру N
        std::vector<int> h_data = generateData(n);                   // генерируем массив на CPU (host)

        int* d_data = nullptr;                                       // указатель на массив в памяти GPU (device)
        unsigned long long* d_sum = nullptr;                         // указатель на сумму в памяти GPU (device)

        // Выделяем память на GPU под массив
        checkCuda(cudaMalloc(&d_data, n * sizeof(int)), "malloc d_data");

        // Выделяем память на GPU под результат суммы (1 число)
        checkCuda(cudaMalloc(&d_sum, sizeof(unsigned long long)), "malloc d_sum");

        // Копируем массив с CPU (host) на GPU (device)
        checkCuda(cudaMemcpy(d_data, h_data.data(),                  // куда (device), откуда (host)
                             n * sizeof(int),                        // сколько байт копировать
                             cudaMemcpyHostToDevice),                // направление копирования
                  "copy to device");

        // Создаём CUDA события для замера времени
        cudaEvent_t start, stop;                                     // переменные под события
        cudaEventCreate(&start);                                     // создаём событие start
        cudaEventCreate(&stop);                                      // создаём событие stop

        // -------------------- Вариант A: GLOBAL ONLY --------------------
        checkCuda(cudaMemset(d_sum, 0, sizeof(unsigned long long)), "memset d_sum (global only)"); // обнуляем сумму на GPU

        cudaEventRecord(start);                                      // ставим метку времени "старт"

        reduceGlobalOnly<<<120, BLOCK_SIZE>>>(d_data, n, d_sum);      // запускаем ядро: 120 блоков, BLOCK_SIZE потоков

        cudaEventRecord(stop);                                       // ставим метку времени "стоп"
        cudaEventSynchronize(stop);                                  // ждём завершения ядра (и события stop)

        float timeGlobal = 0.0f;                                     // сюда запишем время варианта A
        cudaEventElapsedTime(&timeGlobal, start, stop);              // вычисляем время между start и stop (в мс)

        // -------------------- Вариант B: GLOBAL + SHARED --------------------
        checkCuda(cudaMemset(d_sum, 0, sizeof(unsigned long long)), "memset d_sum (shared)");      // снова обнуляем сумму

        cudaEventRecord(start);                                      // старт замера
