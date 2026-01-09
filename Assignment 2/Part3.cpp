// Подключаем CUDA Runtime API 
#include <cuda_runtime.h>

// Подключаем определения для CUDA kernel
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <climits>

using namespace std;

// Макрос для проверки ошибок CUDA.
// call — это любой CUDA-вызов (cudaMalloc, cudaMemcpy и т.д.)
#define CUDA_CHECK(call) do {                               \
    cudaError_t err = (call);                               /* выполняем CUDA-вызов */ \
    if (err != cudaSuccess) {                               /* если произошла ошибка */ \
        cerr << "CUDA error: "                              /* выводим сообщение */ \
             << cudaGetErrorString(err)                     /* текст ошибки CUDA */ \
             << " at " << __FILE__ << ":" << __LINE__       /* файл и строка */ \
             << endl;                                       \
        exit(1);                                            /* завершаем программу */ \
    }                                                       \
} while (0)

// Функция деления с округлением вверх.
// Используется для расчёта количества CUDA-блоков.
static int divUp(int a, int b) {
    return (a + b - 1) / b;  // стандартная формула округления вверх
}

// Функция заполнения массива случайными числами на CPU
static void fillRandom(vector<int>& a, int lo = 1, int hi = 100000) {
    // Проходим по всем элементам массива
    for (int i = 0; i < (int)a.size(); i++) {
        // Генерируем случайное число в диапазоне [lo; hi]
        a[i] = lo + rand() % (hi - lo + 1);
    }
}

// Функция проверки, что массив отсортирован (на CPU)
static bool isSortedHost(const vector<int>& a) {
    // std::is_sorted возвращает true, если массив отсортирован по возрастанию
    return std::is_sorted(a.begin(), a.end());
}

// Количество потоков в одном CUDA-блоке
constexpr int BLOCK_THREADS = 256;

// Размер чанка, который обрабатывает один блок GPU
constexpr int CHUNK = 1024;

// Проверка на этапе компиляции, что CHUNK — степень двойки
static_assert((CHUNK & (CHUNK - 1)) == 0, "CHUNK must be power of two");

// CUDA kernel для сортировки чанков.
// Каждый блок сортирует один подмассив (чанк).
__global__ void sortChunksBitonicKernel(const int* __restrict__ d_in,
                                       int* __restrict__ d_out,
                                       int n) {
    // Shared memory — быстрая память, общая для потоков блока
    __shared__ int s[CHUNK];

    // Номер блока (какой чанк обрабатывается)
    int blockId = blockIdx.x;

    // Номер потока внутри блока
    int tid = threadIdx.x;

    // Начальный индекс чанка в глобальном массиве
    int base = blockId * CHUNK;

    // Каждый поток загружает несколько элементов чанка в shared memory
    for (int k = tid; k < CHUNK; k += blockDim.x) {
        int idx = base + k;                         // глобальный индекс
        s[k] = (idx < n) ? d_in[idx] : INT_MAX;     // если вышли за границу — кладём INT_MAX
    }

    // Синхронизация потоков внутри блока
    __syncthreads();

    // Реализация bitonic sort в shared memory
    for (int k = 2; k <= CHUNK; k <<= 1) {           // размер битонической последовательности
        for (int j = k >> 1; j > 0; j >>= 1) {       // шаг сравнения
            for (int i = tid; i < CHUNK; i += blockDim.x) {
                int ixj = i ^ j;                    // индекс элемента для сравнения

                if (ixj > i) {
                    bool ascending = ((i & k) == 0); // направление сортировки
                    int a = s[i];                    // текущий элемент
                    int b = s[ixj];                  // элемент для сравнения

                    // Меняем элементы местами при необходимости
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();                         // синхронизация после шага
        }
    }

    // Записываем отсортированный чанк обратно в глобальную память
    for (int k = tid; k < CHUNK; k += blockDim.x) {
        int idx = base + k;
        if (idx < n) {
            d_out[idx] = s[k];
        }
    }
}

// Device-функция для поиска позиции разреза при слиянии
__device__ int pickFromLeftCount(const int* A,
                                 int leftStart, int leftLen,
                                 int rightStart, int rightLen,
                                 int k) {
    int lo = max(0, k - rightLen);   // нижняя граница бинарного поиска
    int hi = min(k, leftLen);        // верхняя граница бинарного поиска

    // Бинарный поиск корректного разреза
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int i = mid;
        int j = k - i;

        int leftPrev  = (i == 0) ? INT_MIN : A[leftStart + i - 1];
        int leftCur   = (i == leftLen) ? INT_MAX : A[leftStart + i];
        int rightPrev = (j == 0) ? INT_MIN : A[rightStart + j - 1];
        int rightCur  = (j == rightLen) ? INT_MAX : A[rightStart + j];

        if (rightPrev > leftCur)
            lo = mid + 1;            // взяли слишком мало из левого массива
        else
            hi = mid;
    }
    return lo;
}

// CUDA kernel для параллельного слияния run-ов
__global__ void mergeRunsKernel(const int* __restrict__ d_in,
                               int* __restrict__ d_out,
                               int n,
                               int runSize) {
    int outPos = blockIdx.x * blockDim.x + threadIdx.x; // индекс результата

    if (outPos >= n) return;                            // проверка выхода за границу

    int pairSize = 2 * runSize;                         // размер пары run-ов
    int pairStart = (outPos / pairSize) * pairSize;    // начало пары
    int k = outPos - pairStart;                         // позиция внутри пары

    int leftStart = pairStart;                          // начало левого run-а
    int rightStart = pairStart + runSize;               // начало правого run-а

    int leftLen = min(runSize, n - leftStart);          // длина левого run-а
    int rightLen = (rightStart < n)                     // длина правого run-а
                     ? min(runSize, n - rightStart)
                     : 0;

    if (k >= leftLen + rightLen) return;                // если вне диапазона

    int i = pickFromLeftCount(d_in, leftStart, leftLen,
                              rightStart, rightLen, k);
    int j = k - i;

    int leftPrev  = (i == 0) ? INT_MIN : d_in[leftStart + i - 1];
    int rightPrev = (j == 0) ? INT_MIN : d_in[rightStart + j - 1];

    d_out[outPos] = (leftPrev > rightPrev) ? leftPrev : rightPrev;
}

// Функция запуска GPU merge sort и измерения времени
static float gpuMergeSort(int* d_A, int* d_B, int n) {
    cudaEvent_t start, stop;                       // CUDA события для тайминга
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));             // старт таймера

    int numChunks = divUp(n, CHUNK);                // количество чанков
    sortChunksBitonicKernel<<<numChunks, BLOCK_THREADS>>>(d_A, d_B, n);
    CUDA_CHECK(cudaGetLastError());

    int* in = d_B;                                 // входной буфер
    int* out = d_A;                                // выходной буфер

    int runSize = CHUNK;                           // начальный размер run-а
    while (runSize < n) {
        int threads = 256;
        int blocks = divUp(n, threads);

        mergeRunsKernel<<<blocks, threads>>>(in, out, n, runSize);
        CUDA_CHECK(cudaGetLastError());

        int* tmp = in;                             // меняем буферы местами
        in = out;
        out = tmp;

        runSize *= 2;                              // увеличиваем размер run-а
    }

    CUDA_CHECK(cudaEventRecord(stop));              // стоп таймера
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;                               // время в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

// Функция тестирования для одного размера массива
static void benchmark(int n) {
    vector<int> h(n);                              // массив на CPU
    fillRandom(h);                                 // заполняем случайными числами

    int* d_A = nullptr;                            // первый буфер GPU
    int* d_B = nullptr;                            // второй буфер GPU

    CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(int))); // выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h.data(),            // копируем CPU → GPU
                          n * sizeof(int),
                          cudaMemcpyHostToDevice));

    float gpuMs = gpuMergeSort(d_A, d_B, n);       // сортировка и замер времени

    vector<int> hA(n), hB(n);                      // массивы для результата
    CUDA_CHECK(cudaMemcpy(hA.data(), d_A, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hB.data(), d_B, n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool sortedA = isSortedHost(hA);               // проверяем d_A
    bool sortedB = isSortedHost(hB);               // проверяем d_B

    cout << "N = " << n << "\n";
    cout << "GPU time (chunk sort + merges): " << gpuMs << " ms\n";
    cout << "Sorted in d_A: " << (sortedA ? "true" : "false") << "\n";
    cout << "Sorted in d_B: " << (sortedB ? "true" : "false") << "\n";
    cout << "----------------------------------------\n";

    CUDA_CHECK(cudaFree(d_A));                     // освобождаем память GPU
    CUDA_CHECK(cudaFree(d_B));
}

// Точка входа программы
int main() {
    setlocale(LC_ALL, "Russian");                  // русская локаль
    srand((unsigned)time(nullptr));                // инициализация rand()

    benchmark(10000);                              // тест для 10 000 элементов
    benchmark(100000);                             // тест для 100 000 элементов

    cout << "Вывод:\n";
    cout << "1) Каждый блок GPU сортирует свой подмассив.\n";
    cout << "2) Затем выполняется параллельное слияние.\n";
    cout << "3) Используется ping-pong схема работы с памятью.\n";

    return 0;
}
