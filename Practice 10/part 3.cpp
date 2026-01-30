%%writefile main.cu
#include <cuda_runtime.h>      // CUDA runtime API
#include <omp.h>               // omp_get_wtime() для CPU тайминга (можно заменить на std::chrono)
#include <iostream>            // вывод
#include <vector>              // CPU контейнер (для сравнения/референса)
#include <iomanip>             // форматирование
#include <cmath>               // fabs
#include <algorithm>           // std::max

// ----------------------------- Проверка ошибок CUDA -----------------------------
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

// ----------------------------- GPU kernel: простая "тяжёленькая" обработка -----------------------------
// Здесь делаем несколько операций на элемент, чтобы ядро занимало заметное время.
// В реальном мире это могла бы быть фильтрация, нормализация, feature engineering и т.п.
__global__ void transform_kernel(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // глобальный индекс
    if (idx < n) {                                     // защита границ
        float x = in[idx];                             // чтение
        // Несколько арифм. операций (чтобы не было "слишком пустого" memory-bound ядра)
        x = x * 1.000123f + 0.000321f;                 // линейное преобразование
        x = x * x + 0.5f * x + 0.25f;                  // полином
        out[idx] = x;                                  // запись результата
    }
}

// ----------------------------- CPU часть гибридного алгоритма -----------------------------
// Например: "подготовка данных" или "пост-обработка".
// Тут: считаем сумму результатов на CPU (как пример работы CPU после GPU).
double cpu_reduce_sum(const float* data, int n) {
    double s = 0.0;                                    // аккумулятор
    for (int i = 0; i < n; ++i) {                       // цикл по данным
        s += data[i];                                   // суммирование
    }
    return s;                                          // возвращаем сумму
}

// ----------------------------- Вспомогательная функция: тайминг CUDA events -----------------------------
// Возвращает миллисекунды между start и stop.
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;                                   // сюда время
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // считаем
    return ms;                                         // возвращаем
}

int main() {
    // ----------------------------- Параметры эксперимента -----------------------------
    const int N = 1 << 26;                              // ~67 млн элементов (большой массив)
    const int BLOCK = 256;                              // threads per block
    const int STREAMS = 4;                              // количество CUDA streams
    const int CHUNK = (N + STREAMS - 1) / STREAMS;      // размер чанка на один stream (приближенно)
    const int REPEATS = 5;                              // повторов всего пайплайна для усреднения

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Hybrid CPU+GPU profiling\n";
    std::cout << "N=" << N << ", BLOCK=" << BLOCK << ", STREAMS=" << STREAMS
              << ", approx CHUNK=" << CHUNK << ", REPEATS=" << REPEATS << "\n\n";

    // - Обычная host-память может копироваться через staging/пейджинг -> дороже.
    // - pinned память позволяет DMA и обычно ускоряет cudaMemcpyAsync и overlap.

    // ----------------------------- Выделяем pinned memory на host -----------------------------
    float* h_in  = nullptr;                             // вход на CPU (pinned)
    float* h_out = nullptr;                             // выход на CPU (pinned)
    CUDA_CHECK(cudaMallocHost(&h_in,  N * sizeof(float)));  // page-locked host input
    CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));  // page-locked host output

    // ----------------------------- Инициализация данных на CPU (последовательная часть) -----------------------------
    double t_cpu0 = omp_get_wtime();                    // старт CPU таймера
    for (int i = 0; i < N; ++i) {                       // генерируем вход
        h_in[i] = (i % 1024) * 0.001f;                  // простой паттерн
    }
    double t_cpu1 = omp_get_wtime();                    // конец CPU таймера
    double cpu_prepare_s = (t_cpu1 - t_cpu0);           // время подготовки

    // ----------------------------- Выделяем device память -----------------------------
    float* d_in  = nullptr;                             // вход на GPU
    float* d_out = nullptr;                             // выход на GPU
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));   // device input
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));   // device output

    // ----------------------------- Создаём streams -----------------------------
    std::vector<cudaStream_t> streams(STREAMS);          // контейнер потоков
    for (int s = 0; s < STREAMS; ++s) {                  // создаём каждый stream
        CUDA_CHECK(cudaStreamCreate(&streams[s]));       // создание stream
    }

    // ----------------------------- События для профилирования по стадиям -----------------------------
    // Мы будем измерять отдельно:
    // - H2D (host->device)
    // - Kernel
    // - D2H (device->host)
    // и общий "end-to-end".
    cudaEvent_t ev_start, ev_stop;                       // общие события
    CUDA_CHECK(cudaEventCreate(&ev_start));              // создаём start
    CUDA_CHECK(cudaEventCreate(&ev_stop));               // создаём stop

    // Для более точной детализации сделаем массив событий по stream-ам и стадиям
    // (start/end для H2D, kernel, D2H).
    struct StageEvents {
        cudaEvent_t h2d_start, h2d_end;                  // события H2D
        cudaEvent_t k_start,   k_end;                    // события kernel
        cudaEvent_t d2h_start, d2h_end;                  // события D2H
    };

    std::vector<StageEvents> sev(STREAMS);               // события на каждый stream
    for (int s = 0; s < STREAMS; ++s) {                  // создаём события
        CUDA_CHECK(cudaEventCreate(&sev[s].h2d_start));
        CUDA_CHECK(cudaEventCreate(&sev[s].h2d_end));
        CUDA_CHECK(cudaEventCreate(&sev[s].k_start));
        CUDA_CHECK(cudaEventCreate(&sev[s].k_end));
        CUDA_CHECK(cudaEventCreate(&sev[s].d2h_start));
        CUDA_CHECK(cudaEventCreate(&sev[s].d2h_end));
    }

    // ----------------------------- Агрегаторы времени (мс) -----------------------------
    double avg_total_ms = 0.0;                           // среднее end-to-end
    double avg_h2d_ms = 0.0;                             // среднее H2D
    double avg_kernel_ms = 0.0;                          // среднее kernel
    double avg_d2h_ms = 0.0;                             // среднее D2H
    double avg_cpu_post_s = 0.0;                         // среднее CPU пост-обработка (сек)

    // =====================================================================================
    // Основной цикл измерений: pipeline с overlap (асинхронные копии + kernel в разных streams)
    // =====================================================================================
    for (int rep = 0; rep < REPEATS; ++rep) {
        // Очистка выходного буфера на host (не обязательна, но удобно)
        for (int i = 0; i < N; ++i) {                    // простой обнулитель
            h_out[i] = 0.0f;                             // сбрасываем
        }

        CUDA_CHECK(cudaEventRecord(ev_start));           // общий старт (в default stream)

        // ----------------------------- Запускаем работу по чанкам -----------------------------
        for (int s = 0; s < STREAMS; ++s) {              // для каждого stream
            int offset = s * CHUNK;                      // смещение чанка
            int size = std::min(CHUNK, N - offset);      // размер чанка (последний может быть меньше)
            if (size <= 0) continue;                     // если вышли за пределы — пропускаем

            // ---- H2D async ----
            CUDA_CHECK(cudaEventRecord(sev[s].h2d_start, streams[s]));  // отметка начала H2D в этом stream
            CUDA_CHECK(cudaMemcpyAsync(d_in + offset,                   // dst на GPU
                                       h_in + offset,                   // src на CPU
                                       size * sizeof(float),           // байты
                                       cudaMemcpyHostToDevice,         // направление
                                       streams[s]));                   // stream
            CUDA_CHECK(cudaEventRecord(sev[s].h2d_end, streams[s]));    // отметка конца H2D

            // ---- Kernel ----
            int grid = (size + BLOCK - 1) / BLOCK;       // сколько блоков для этого чанка
            CUDA_CHECK(cudaEventRecord(sev[s].k_start, streams[s]));    // начало kernel
            transform_kernel<<<grid, BLOCK, 0, streams[s]>>>(d_in + offset, d_out + offset, size); // kernel
            CUDA_CHECK(cudaGetLastError());              // проверка ошибки запуска kernel
            CUDA_CHECK(cudaEventRecord(sev[s].k_end, streams[s]));      // конец kernel

            // ---- D2H async ----
            CUDA_CHECK(cudaEventRecord(sev[s].d2h_start, streams[s]));  // начало D2H
            CUDA_CHECK(cudaMemcpyAsync(h_out + offset,                  // dst на CPU
                                       d_out + offset,                  // src на GPU
                                       size * sizeof(float),           // байты
                                       cudaMemcpyDeviceToHost,         // направление
                                       streams[s]));                   // stream
            CUDA_CHECK(cudaEventRecord(sev[s].d2h_end, streams[s]));    // конец D2H
        }

        // ----------------------------- Ждём завершения всех streams -----------------------------
        for (int s = 0; s < STREAMS; ++s) {              // по каждому stream
            CUDA_CHECK(cudaStreamSynchronize(streams[s])); // ждём окончания H2D+kernel+D2H в stream
        }

        CUDA_CHECK(cudaEventRecord(ev_stop));            // общий стоп
        CUDA_CHECK(cudaEventSynchronize(ev_stop));       // ждём, пока stop зафиксируется

        // ----------------------------- Считаем времена стадий (суммируя по streams) -----------------------------
        double h2d_ms = 0.0;                             // накопитель H2D
        double k_ms   = 0.0;                             // накопитель kernel
        double d2h_ms = 0.0;                             // накопитель D2H

        for (int s = 0; s < STREAMS; ++s) {              // суммируем по stream-ам
            h2d_ms += elapsed_ms(sev[s].h2d_start, sev[s].h2d_end); // время H2D в stream
            k_ms   += elapsed_ms(sev[s].k_start,   sev[s].k_end);   // время kernel в stream
            d2h_ms += elapsed_ms(sev[s].d2h_start, sev[s].d2h_end); // время D2H в stream
        }

        float total_ms = elapsed_ms(ev_start, ev_stop);  // общий end-to-end по GPU событиям

        // ----------------------------- CPU пост-обработка: редукция суммы -----------------------------
        // Это часть "гибридности": GPU сделал transform, CPU делает агрегацию.
        double cpu_post0 = omp_get_wtime();              // старт CPU пост-обработки
        double sum = cpu_reduce_sum(h_out, N);           // суммируем выход на CPU
        double cpu_post1 = omp_get_wtime();              // конец CPU пост-обработки
        double cpu_post_s = (cpu_post1 - cpu_post0);     // время CPU пост-обработки

        // Чтобы sum не выкинули оптимизаторы и было видно, что реально считали:
        if (rep == 0) {                                  // печатаем только один раз
            std::cout << "Sample CPU post-sum = " << sum << "\n\n";
        }

        // ----------------------------- Агрегируем для среднего -----------------------------
        avg_total_ms   += total_ms;                      // общий
        avg_h2d_ms     += h2d_ms;                        // H2D (сумма по streams)
        avg_kernel_ms  += k_ms;                          // kernel (сумма по streams)
        avg_d2h_ms     += d2h_ms;                        // D2H (сумма по streams)
        avg_cpu_post_s += cpu_post_s;                    // CPU post

        // ----------------------------- Печатаем репорт по одному прогону -----------------------------
        std::cout << "Run " << rep + 1 << ":\n";
        std::cout << "  total (GPU timeline)    = " << total_ms << " ms\n";
        std::cout << "  H2D sum over streams    = " << h2d_ms   << " ms\n";
        std::cout << "  kernel sum over streams = " << k_ms     << " ms\n";
        std::cout << "  D2H sum over streams    = " << d2h_ms   << " ms\n";
        std::cout << "  CPU post (reduce)       = " << (cpu_post_s * 1000.0) << " ms\n\n";
    }

    // ----------------------------- Усреднение -----------------------------
    avg_total_ms   /= REPEATS;                            // среднее total
    avg_h2d_ms     /= REPEATS;                            // среднее H2D
    avg_kernel_ms  /= REPEATS;                            // среднее kernel
    avg_d2h_ms     /= REPEATS;                            // среднее D2H
    avg_cpu_post_s /= REPEATS;                            // среднее CPU post

    // ----------------------------- Анализ накладных расходов -----------------------------
    // Накладные расходы передачи данных можно оценить как (H2D + D2H) относительно total.
    // Но важно: из-за overlap total может быть меньше суммы стадий.
    double transfer_ms = avg_h2d_ms + avg_d2h_ms;         // суммарное среднее время копирований (по streams)
    double transfer_share = transfer_ms / avg_total_ms;   // "доля" относительно общего времени (может быть >1 при overlap)

    // ----------------------------- Печать итогов -----------------------------
    std::cout << "==================== AVERAGE OVER RUNS ====================\n";
    std::cout << "CPU prepare (init input)           = " << (cpu_prepare_s * 1000.0) << " ms\n";
    std::cout << "AVG total (GPU timeline)           = " << avg_total_ms << " ms\n";
    std::cout << "AVG H2D sum over streams           = " << avg_h2d_ms << " ms\n";
    std::cout << "AVG kernel sum over streams        = " << avg_kernel_ms << " ms\n";
    std::cout << "AVG D2H sum over streams           = " << avg_d2h_ms << " ms\n";
    std::cout << "AVG CPU post (reduce)              = " << (avg_cpu_post_s * 1000.0) << " ms\n";
    std::cout << "Transfer overhead (H2D+D2H) / total = " << transfer_share << " (note: overlap can inflate this ratio)\n\n";

    // ----------------------------- Выявление узких мест -----------------------------
    // Простая логика: смотрим, что крупнее по времени.
    std::cout << "Bottleneck hints:\n";
    if (avg_kernel_ms > avg_h2d_ms && avg_kernel_ms > avg_d2h_ms) {
        std::cout << "  - Likely kernel-bound (kernel dominates).\n";
    } else if (avg_h2d_ms + avg_d2h_ms > avg_kernel_ms) {
        std::cout << "  - Likely transfer-bound (copies dominate).\n";
    } else {
        std::cout << "  - Mixed/overlapped; check per-stage and overlap efficiency.\n";
    }
    if (avg_cpu_post_s * 1000.0 > avg_total_ms) {
        std::cout << "  - CPU post-processing is significant; consider parallelizing it or moving reduction to GPU.\n";
    }
    std::cout << "\n";

    // Реализовано pinned host memory (cudaMallocHost) + cudaMemcpyAsync + streams (overlap)
    // Идея : Сравнить с вариантом без pinned (обычный new/malloc) и/или без streams (один stream, синхронные memcpy)
    

    // ----------------------------- Cleanup -----------------------------
    for (int s = 0; s < STREAMS; ++s) {                   // уничтожаем streams
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    for (int s = 0; s < STREAMS; ++s) {                   // уничтожаем events
        CUDA_CHECK(cudaEventDestroy(sev[s].h2d_start));
        CUDA_CHECK(cudaEventDestroy(sev[s].h2d_end));
        CUDA_CHECK(cudaEventDestroy(sev[s].k_start));
        CUDA_CHECK(cudaEventDestroy(sev[s].k_end));
        CUDA_CHECK(cudaEventDestroy(sev[s].d2h_start));
        CUDA_CHECK(cudaEventDestroy(sev[s].d2h_end));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));               // уничтожаем общий start
    CUDA_CHECK(cudaEventDestroy(ev_stop));                // уничтожаем общий stop

    CUDA_CHECK(cudaFree(d_in));                           // освобождаем device память
    CUDA_CHECK(cudaFree(d_out));                          // освобождаем device память

    CUDA_CHECK(cudaFreeHost(h_in));                       // освобождаем pinned host память
    CUDA_CHECK(cudaFreeHost(h_out));                      // освобождаем pinned host память

    return 0;                                             // успех
}
