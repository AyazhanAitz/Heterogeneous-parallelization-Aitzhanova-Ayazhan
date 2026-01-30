%%writefile task3_perf.cu
#include <cuda_runtime.h>                             
#include <iostream>                                   
#include <vector>                                     
#include <numeric>                                    
#include <chrono>                                     
#include <cmath>                                      
#include <cstring>                                    
#include <algorithm>                                  

// ------------------------------
// Макрос проверки ошибок CUDA
// ------------------------------
#define CUDA_CHECK(call) do {                            /* Начинаем "обёртку" do-while для макроса. */ \
    cudaError_t err = (call);                            /* Выполняем CUDA-вызов и сохраняем код ошибки. */ \
    if (err != cudaSuccess) {                            /* Если код ошибки не равен успеху, */ \
        std::cerr << "CUDA error: "                      /* Печатаем текст ошибки. */ \
                  << cudaGetErrorString(err)             /* Печатаем строку с описанием ошибки. */ \
                  << " at " << __FILE__                  /* Печатаем имя файла. */ \
                  << ":" << __LINE__                     /* Печатаем номер строки. */ \
                  << std::endl;                          /* Переходим на новую строку. */ \
        std::exit(1);                                    /* Завершаем программу с кодом ошибки. */ \
    }                                                    /* Закрываем if. */ \
} while(0)                                               /* Закрываем do-while (ВАЖНО: без лишних символов!). */

// ------------------------------
// Таймер GPU на cudaEvent
// ------------------------------
struct GpuTimer {                                        // Объявляем структуру таймера.
    cudaEvent_t start;                                   // Событие начала.
    cudaEvent_t stop;                                    // Событие конца.
    GpuTimer() {                                         // Конструктор.
        CUDA_CHECK(cudaEventCreate(&start));             // Создаём start.
        CUDA_CHECK(cudaEventCreate(&stop));              // Создаём stop.
    }                                                    // Конец конструктора.
    ~GpuTimer() {                                        // Деструктор.
        cudaEventDestroy(start);                         // Удаляем start.
        cudaEventDestroy(stop);                          // Удаляем stop.
    }                                                    // Конец деструктора.
    void tic(cudaStream_t s = 0) {                       // Начать замер.
        CUDA_CHECK(cudaEventRecord(start, s));           // Записать start в поток.
    }                                                    // Конец tic.
    float toc(cudaStream_t s = 0) {                      // Закончить замер и вернуть мс.
        CUDA_CHECK(cudaEventRecord(stop, s));            // Записать stop в поток.
        CUDA_CHECK(cudaEventSynchronize(stop));          // Дождаться stop.
        float ms = 0.0f;                                 // Переменная времени.
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // Разница времени в мс.
        return ms;                                       // Возвращаем мс.
    }                                                    // Конец toc.
};                                                       // Конец структуры.

// ============================================================
// 1) REDUCTION: две версии (плохая и хорошая)
// ============================================================

// (A) Плохая версия редукции: atomicAdd в глобальную память.
__global__ void reduce_atomic_global(const float* d_in,  // Входной массив на GPU.
                                     float* d_out,       // Один элемент-сумма на GPU.
                                     int n)              // Размер массива.
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;      // Глобальный индекс потока.
    if (idx < n) {                                        // Проверка границ.
        atomicAdd(d_out, d_in[idx]);                      // Атомарное сложение (медленно при больших n).
    }                                                     // Конец if.
}                                                         // Конец ядра.

// (B) Быстрая версия редукции: shared memory внутри блока + частичные суммы.
__global__ void reduce_shared(const float* __restrict__ d_in, // Вход на GPU.
                              float* __restrict__ d_out,      // Выход: сумма каждого блока.
                              int n)                          // Размер массива.
{
    extern __shared__ float s[];                             // Shared memory на блок.
    unsigned tid = threadIdx.x;                              // Индекс потока в блоке.
    unsigned i = blockIdx.x * (blockDim.x * 2) + tid;        // Читаем по 2 элемента на поток.

    float sum = 0.0f;                                        // Локальная сумма потока.

    if (i < (unsigned)n) sum += d_in[i];                     // Добавляем первый элемент.
    if (i + blockDim.x < (unsigned)n) sum += d_in[i + blockDim.x]; // Добавляем второй элемент.

    s[tid] = sum;                                            // Кладём в shared.
    __syncthreads();                                         // Синхронизация.

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) { // Редукция деревом.
        if (tid < stride) {                                  // Активны потоки первой половины.
            s[tid] += s[tid + stride];                        // Складываем пары.
        }                                                    // Конец if.
        __syncthreads();                                     // Синхронизация после шага.
    }                                                        // Конец цикла.

    if (tid == 0) {                                          // Первый поток блока
        d_out[blockIdx.x] = s[0];                             // записывает сумму блока.
    }                                                        // Конец if.
}                                                            // Конец ядра.

// Хост-функция: многопроходная редукция shared до одного числа.
float gpu_reduce_shared(const float* d_in, int n, int threads) { // Вход: указатель на GPU и размер.
    int blocks = (n + threads * 2 - 1) / (threads * 2);          // Считаем количество блоков.
    float* d_tmp = nullptr;                                      // Временный буфер на GPU.
    CUDA_CHECK(cudaMalloc(&d_tmp, blocks * sizeof(float)));      // Выделяем память под частичные суммы.

    int cur_n = n;                                               // Текущий размер.
    const float* cur_in = d_in;                                  // Текущий вход.
    float* cur_out = d_tmp;                                      // Текущий выход.

    while (true) {                                               // Повторяем до одного блока.
        blocks = (cur_n + threads * 2 - 1) / (threads * 2);       // Пересчитываем блоки.
        reduce_shared<<<blocks, threads, threads * sizeof(float)>>>(cur_in, cur_out, cur_n); // Запускаем ядро.
        CUDA_CHECK(cudaGetLastError());                           // Проверка запуска.
        CUDA_CHECK(cudaDeviceSynchronize());                      // Ждём завершения.

        if (blocks == 1) break;                                   // Если один блок — результат готов.

        cur_n = blocks;                                           // Новый размер = число частичных сумм.
        cur_in = cur_out;                                         // Новый вход = текущий выход.

        CUDA_CHECK(cudaFree(d_tmp));                               // Освобождаем старый буфер.
        blocks = (cur_n + threads * 2 - 1) / (threads * 2);        // Считаем новый размер буфера.
        CUDA_CHECK(cudaMalloc(&d_tmp, blocks * sizeof(float)));    // Выделяем новый буфер.
        cur_out = d_tmp;                                          // Обновляем указатель.
    }                                                             // Конец while.

    float h_sum = 0.0f;                                           // Результат на CPU.
    CUDA_CHECK(cudaMemcpy(&h_sum, cur_out, sizeof(float), cudaMemcpyDeviceToHost)); // Копируем 1 float.
    CUDA_CHECK(cudaFree(d_tmp));                                  // Освобождаем временную память.
    return h_sum;                                                 // Возвращаем сумму.
}                                                                 // Конец функции.

// ============================================================
// 2) SCAN (prefix sum): много-блочный inclusive scan
//    1) scan внутри блоков + blockSums
//    2) scan(blockSums) рекурсивно
//    3) add offsets
// ============================================================

// Ядро: inclusive scan (Hillis–Steele) внутри блока + запись суммы блока.
__global__ void block_inclusive_scan(const float* __restrict__ d_in, // Входной массив.
                                     float* __restrict__ d_out,      // Выходной scan.
                                     float* __restrict__ blockSums,  // Суммы блоков.
                                     int n)                          // Размер.
{
    extern __shared__ float s[];                                     // Shared память на блок.
    int tid = threadIdx.x;                                           // Индекс потока в блоке.
    int gid = blockIdx.x * blockDim.x + tid;                         // Глобальный индекс элемента.

    float x = (gid < n) ? d_in[gid] : 0.0f;                          // Читаем элемент или 0.
    s[tid] = x;                                                      // Записываем в shared.
    __syncthreads();                                                 // Синхронизация.

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {        // offset = 1,2,4,...
        float val = s[tid];                                          // Берём текущее значение.
        if (tid >= offset) val += s[tid - offset];                   // Добавляем значение слева.
        __syncthreads();                                             // Ждём все потоки перед записью.
        s[tid] = val;                                                // Пишем обновление.
        __syncthreads();                                             // Ждём все потоки после записи.
    }                                                                // Конец цикла.

    if (gid < n) d_out[gid] = s[tid];                                // Записываем scan для валидных элементов.

    if (tid == blockDim.x - 1) {                                     // Последний поток блока
        int last = min(n - 1, (blockIdx.x + 1) * blockDim.x - 1);     // Находим последний валидный индекс блока.
        blockSums[blockIdx.x] = d_out[last];                         // Записываем сумму блока.
    }                                                                // Конец if.
}                                                                    // Конец ядра.

// Ядро: добавляем оффсет (сумму предыдущих блоков) к каждому элементу блока.
__global__ void add_block_offsets(float* d_out,                        // Массив scan на GPU.
                                  const float* __restrict__ scannedBlockSums, // Просканированные суммы блоков.
                                  int n)                               // Размер массива.
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;                   // Глобальный индекс.
    if (gid >= n) return;                                              // За границей — выходим.
    if (blockIdx.x > 0) {                                              // Для блоков кроме первого
        d_out[gid] += scannedBlockSums[blockIdx.x - 1];                // добавляем сумму предыдущих блоков.
    }                                                                  // Конец if.
}                                                                      // Конец ядра.

// Хост-функция: много-блочный inclusive scan.
void gpu_scan_inclusive(float* d_in, float* d_out, int n, int threads) { // Вход/выход на GPU.
    int blocks = (n + threads - 1) / threads;                            // Число блоков.
    float* d_blockSums = nullptr;                                        // Суммы блоков.
    CUDA_CHECK(cudaMalloc(&d_blockSums, blocks * sizeof(float)));        // Выделяем память под суммы блоков.

    block_inclusive_scan<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, d_blockSums, n); // Scan по блокам.
    CUDA_CHECK(cudaGetLastError());                                      // Проверяем запуск.
    CUDA_CHECK(cudaDeviceSynchronize());                                 // Ждём завершения.

    if (blocks == 1) {                                                   // Если блок один —
        CUDA_CHECK(cudaFree(d_blockSums));                               // освобождаем память,
        return;                                                          // и выходим.
    }                                                                    // Конец if.

    float* d_scannedBlockSums = nullptr;                                 // Просканированные суммы блоков.
    CUDA_CHECK(cudaMalloc(&d_scannedBlockSums, blocks * sizeof(float))); // Память под scan(blockSums).

    gpu_scan_inclusive(d_blockSums, d_scannedBlockSums, blocks, threads); // Рекурсивно сканируем blockSums.

    add_block_offsets<<<blocks, threads>>>(d_out, d_scannedBlockSums, n); // Добавляем оффсеты к каждому блоку.
    CUDA_CHECK(cudaGetLastError());                                       // Проверяем запуск.
    CUDA_CHECK(cudaDeviceSynchronize());                                  // Ждём завершения.

    CUDA_CHECK(cudaFree(d_blockSums));                                    // Освобождаем blockSums.
    CUDA_CHECK(cudaFree(d_scannedBlockSums));                             // Освобождаем scannedBlockSums.
}                                                                         // Конец scan.

// ============================================================
// 3) CPU reference implementations (для сравнения)
// ============================================================

float cpu_reduce(const std::vector<float>& a) {                           // CPU редукция.
    return std::accumulate(a.begin(), a.end(), 0.0f);                     // Последовательная сумма.
}                                                                         // Конец.

void cpu_scan_inclusive(const std::vector<float>& in, std::vector<float>& out) { // CPU scan.
    out.resize(in.size());                                                // Задаём размер.
    float run = 0.0f;                                                     // Накопленная сумма.
    for (size_t i = 0; i < in.size(); ++i) {                              // Проход по массиву.
        run += in[i];                                                     // Добавляем элемент.
        out[i] = run;                                                     // Записываем префикс.
    }                                                                     // Конец цикла.
}                                                                         // Конец.

// ============================================================
// MAIN: замеры и CSV-результаты
// ============================================================

int main() {                                                              // Точка входа.
    std::vector<int> sizes = { 1<<10, 1<<15, 1<<18, 1<<20, 1<<22 };        // Размеры массивов для теста.
    int threads = 256;                                                    // Размер блока (можно 128/256/512).
    GpuTimer gt;                                                          // Таймер GPU.

    std::cout                                                             // Печатаем заголовок CSV.
        << "n,"                                                           // Размер массива.
        << "cpu_reduce_ms,"                                               // Время CPU reduce.
        << "cpu_scan_ms,"                                                 // Время CPU scan.
        << "h2d_pageable_ms,"                                             // H2D pageable.
        << "h2d_pinned_ms,"                                               // H2D pinned.
        << "d2h_pageable_ms,"                                             // D2H pageable.
        << "d2h_pinned_ms,"                                               // D2H pinned.
        << "gpu_reduce_atomic_ms,"                                        // GPU reduce atomic.
        << "gpu_reduce_shared_ms,"                                        // GPU reduce shared.
        << "gpu_scan_ms,"                                                 // GPU scan.
        << "reduce_abs_diff,"                                             // |CPU sum - GPU sum|.
        << "scan_max_abs_diff"                                            // max |CPU scan - GPU scan|.
        << "\n";                                                          // Перевод строки.

    for (int n : sizes) {                                                 // Цикл по размерам.
        std::vector<float> h_in(n);                                       // Входной массив на CPU.
        for (int i = 0; i < n; ++i) {                                     // Заполняем массив.
            h_in[i] = 0.1f + (i % 7) * 0.0001f;                            // Дробные значения (чтобы проявлялась погрешность).
        }                                                                  // Конец заполнения.

        auto t0 = std::chrono::high_resolution_clock::now();              // Начало замера CPU reduce.
        float cpu_sum = cpu_reduce(h_in);                                 // CPU редукция.
        auto t1 = std::chrono::high_resolution_clock::now();              // Конец замера.
        double cpu_reduce_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); // Время в мс.

        std::vector<float> h_scan_cpu;                                    // Результат CPU scan.
        t0 = std::chrono::high_resolution_clock::now();                   // Начало замера CPU scan.
        cpu_scan_inclusive(h_in, h_scan_cpu);                             // CPU scan.
        t1 = std::chrono::high_resolution_clock::now();                   // Конец замера.
        double cpu_scan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();   // Время в мс.

        float* d_in = nullptr;                                            // Вход на GPU.
        float* d_out = nullptr;                                           // Выход scan на GPU.
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));                 // Выделяем d_in.
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));                // Выделяем d_out.

        float* d_sum_atomic = nullptr;                                    // GPU сумма для atomic.
        CUDA_CHECK(cudaMalloc(&d_sum_atomic, sizeof(float)));             // Выделяем 1 float.

        // -------- H2D pageable --------
        gt.tic();                                                         // Старт таймера.
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // Копируем pageable -> GPU.
        float h2d_pageable_ms = gt.toc();                                  // Время H2D pageable.

        // -------- H2D pinned --------
        float* h_pinned = nullptr;                                        // Указатель pinned на хосте.
        CUDA_CHECK(cudaMallocHost(&h_pinned, n * sizeof(float)));         // Выделяем pinned память.
        std::memcpy(h_pinned, h_in.data(), n * sizeof(float));            // Копируем данные в pinned буфер.

        gt.tic();                                                         // Старт таймера.
        CUDA_CHECK(cudaMemcpy(d_in, h_pinned, n * sizeof(float), cudaMemcpyHostToDevice)); // Копируем pinned -> GPU.
        float h2d_pinned_ms = gt.toc();                                    // Время H2D pinned.

        // -------- GPU reduce atomic --------
        CUDA_CHECK(cudaMemset(d_sum_atomic, 0, sizeof(float)));           // Обнуляем сумму.
        int blocks_atomic = (n + threads - 1) / threads;                  // Блоки для atomic.
        gt.tic();                                                         // Старт таймера.
        reduce_atomic_global<<<blocks_atomic, threads>>>(d_in, d_sum_atomic, n); // Запуск atomic reduce.
        CUDA_CHECK(cudaGetLastError());                                    // Проверяем ошибки.
        CUDA_CHECK(cudaDeviceSynchronize());                               // Ждём завершения.
        float gpu_reduce_atomic_ms = gt.toc();                             // Время atomic reduce.

        // -------- GPU reduce shared --------
        gt.tic();                                                         // Старт таймера.
        float gpu_sum_shared = gpu_reduce_shared(d_in, n, threads);        // Shared reduce.
        float gpu_reduce_shared_ms = gt.toc();                             // Время shared reduce.

        // -------- GPU scan (multi-block) --------
        gt.tic();                                                         // Старт таймера.
        gpu_scan_inclusive(d_in, d_out, n, threads);                       // GPU scan.
        float gpu_scan_ms = gt.toc();                                      // Время scan.

        // -------- D2H pageable --------
        std::vector<float> h_scan_gpu(n);                                  // Буфер scan на CPU.
        gt.tic();                                                         // Старт таймера.
        CUDA_CHECK(cudaMemcpy(h_scan_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // D2H pageable.
        float d2h_pageable_ms = gt.toc();                                  // Время D2H pageable.

        // -------- D2H pinned --------
        gt.tic();                                                         // Старт таймера.
        CUDA_CHECK(cudaMemcpy(h_pinned, d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // D2H pinned.
        float d2h_pinned_ms = gt.toc();                                    // Время D2H pinned.
        std::memcpy(h_scan_gpu.data(), h_pinned, n * sizeof(float));       // Копируем pinned -> vector для проверки.

        // -------- Проверка корректности --------
        float reduce_abs_diff = std::fabs(cpu_sum - gpu_sum_shared);       // Разница сумм.
        float scan_max_abs_diff = 0.0f;                                    // Максимальная ошибка scan.
        for (int i = 0; i < n; ++i) {                                      // Сравнение scan поэлементно.
            float diff = std::fabs(h_scan_cpu[i] - h_scan_gpu[i]);         // Разница на позиции i.
            if (diff > scan_max_abs_diff) scan_max_abs_diff = diff;        // Обновляем максимум.
        }                                                                  // Конец проверки.

        // -------- Печать CSV строки --------
        std::cout
            << n << ","                                                   // n.
            << cpu_reduce_ms << ","                                       // CPU reduce.
            << cpu_scan_ms << ","                                         // CPU scan.
            << h2d_pageable_ms << ","                                     // H2D pageable.
            << h2d_pinned_ms << ","                                       // H2D pinned.
            << d2h_pageable_ms << ","                                     // D2H pageable.
            << d2h_pinned_ms << ","                                       // D2H pinned.
            << gpu_reduce_atomic_ms << ","                                // GPU atomic reduce.
            << gpu_reduce_shared_ms << ","                                // GPU shared reduce.
            << gpu_scan_ms << ","                                         // GPU scan.
            << reduce_abs_diff << ","                                     // diff reduce.
            << scan_max_abs_diff                                          // diff scan.
            << "\n";                                                      // newline.

        // -------- Освобождение памяти --------
        CUDA_CHECK(cudaFree(d_in));                                       // Освобождаем d_in.
        CUDA_CHECK(cudaFree(d_out));                                      // Освобождаем d_out.
        CUDA_CHECK(cudaFree(d_sum_atomic));                               // Освобождаем d_sum_atomic.
        CUDA_CHECK(cudaFreeHost(h_pinned));                               // Освобождаем pinned host memory.
    }                                                                      // Конец цикла.

    return 0;                                                              // Успешный выход.
}                                                                          // Конец main.
