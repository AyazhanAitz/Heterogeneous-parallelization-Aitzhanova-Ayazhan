%%writefile reduction.cu
#include <cuda_runtime.h>                      // Подключаем CUDA Runtime API.
#include <iostream>                            // Для вывода в консоль.
#include <vector>                              // Для удобного хранения массивов на CPU.
#include <numeric>                             // Для std::accumulate (сумма на CPU).
#include <cstdlib>                             // Для rand().
#include <cmath>                               // Для fabs() при проверке.

#define CUDA_CHECK(call) do {                  /* Макрос для проверки ошибок CUDA. */ \
    cudaError_t err = (call);                  /* Выполняем вызов CUDA и сохраняем код ошибки. */ \
    if (err != cudaSuccess) {                  /* Если ошибка есть — выводим и завершаем. */ \
        std::cerr << "CUDA error: "            /* Печатаем префикс. */ \
                  << cudaGetErrorString(err)   /* Печатаем текст ошибки. */ \
                  << " at " << __FILE__        /* Печатаем имя файла. */ \
                  << ":" << __LINE__           /* Печатаем номер строки. */ \
                  << std::endl;                /* Перевод строки. */ \
        std::exit(1);                          /* Завершаем программу с ошибкой. */ \
    }                                          /* Конец проверки. */ \
} while(0)                                     /* Удобная форма макроса. */

// Ядро редукции: суммирует элементы входного массива и пишет частичные суммы по блокам.
__global__ void reduce_sum_shared(const float* __restrict__ d_in,  // Указатель на входной массив на GPU.
                                  float* __restrict__ d_out,       // Указатель на массив частичных сумм (по блокам).
                                  int n)                           // Размер входного массива.
{
    extern __shared__ float sdata[];                                 // Разделяемая память: буфер для редукции внутри блока.

    unsigned int tid = threadIdx.x;                                  // Локальный индекс потока в блоке.
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;  // Глобальный индекс: читаем по 2 элемента на поток.

    float sum = 0.0f;                                                // Локальная сумма потока.

    if (idx < (unsigned)n) sum += d_in[idx];                         // Добавляем первый элемент, если в границах.
    if (idx + blockDim.x < (unsigned)n) sum += d_in[idx + blockDim.x]; // Добавляем второй элемент, если в границах.

    sdata[tid] = sum;                                                // Кладём сумму потока в shared memory.
    __syncthreads();                                                 // Ждём, пока все потоки запишут данные.

    // Редукция внутри блока (деревом): каждый шаг делит число активных потоков пополам.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {          // s — текущее смещение (stride).
        if (tid < s) {                                               // Активны только потоки в первой половине.
            sdata[tid] += sdata[tid + s];                            // Складываем пару элементов.
        }
        __syncthreads();                                             // Синхронизация перед следующим шагом.
    }

    if (tid == 0) {                                                  // Первый поток блока записывает результат блока.
        d_out[blockIdx.x] = sdata[0];                                // Частичная сумма всего блока.
    }
}

// Вспомогательная функция: делает редукцию на GPU до одного числа, вызывая reduce_sum_shared несколько раз.
float gpu_reduce_sum(const std::vector<float>& h_in)                 // Принимаем входной массив на CPU.
{
    int n = (int)h_in.size();                                        // Размер массива.
    if (n == 0) return 0.0f;                                         // Если массив пустой — сумма 0.

    float* d_in = nullptr;                                           // Указатель на вход на GPU.
    float* d_tmp = nullptr;                                          // Указатель на временный буфер частичных сумм.

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));                // Выделяем память под входной массив.
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // Копируем данные на GPU.

    int threads = 256;                                               // Число потоков в блоке (обычно 128/256/512).
    int blocks = (n + threads * 2 - 1) / (threads * 2);              // Число блоков с учётом чтения по 2 элемента на поток.

    CUDA_CHECK(cudaMalloc(&d_tmp, blocks * sizeof(float)));          // Выделяем память под частичные суммы блоков.

    const int shared_bytes = threads * sizeof(float);                // Размер shared memory на блок.

    int cur_n = n;                                                   // Текущий размер массива для редукции.
    const float* cur_in = d_in;                                      // Текущий вход (сначала это исходный массив).
    float* cur_out = d_tmp;                                          // Текущий выход (частичные суммы).

    while (true) {                                                   // Повторяем, пока не останется один элемент.
        blocks = (cur_n + threads * 2 - 1) / (threads * 2);           // Пересчитываем количество блоков для текущего размера.

        reduce_sum_shared<<<blocks, threads, shared_bytes>>>(cur_in, cur_out, cur_n); // Запускаем ядро редукции.
        CUDA_CHECK(cudaGetLastError());                               // Проверяем ошибку запуска ядра.
        CUDA_CHECK(cudaDeviceSynchronize());                          // Ждём завершения ядра для корректной проверки.

        if (blocks == 1) break;                                       // Если остался один блок — результат в cur_out[0].

        cur_n = blocks;                                               // Новый размер = число частичных сумм (по блокам).
        cur_in = cur_out;                                             // Следующий вход — это текущий выход (частичные суммы).

        CUDA_CHECK(cudaFree(d_tmp));                                  // Освобождаем старый временный буфер.
        CUDA_CHECK(cudaMalloc(&d_tmp, ((cur_n + threads * 2 - 1) / (threads * 2)) * sizeof(float))); // Выделяем новый буфер.
        cur_out = d_tmp;                                              // Обновляем указатель на выход.
    }

    float h_sum = 0.0f;                                               // Результат на CPU.
    CUDA_CHECK(cudaMemcpy(&h_sum, cur_out, sizeof(float), cudaMemcpyDeviceToHost)); // Копируем итоговую сумму на CPU.

    CUDA_CHECK(cudaFree(d_in));                                       // Освобождаем память входа.
    CUDA_CHECK(cudaFree(d_tmp));                                      // Освобождаем временную память.

    return h_sum;                                                     // Возвращаем сумму.
}

int main()                                                           // Точка входа.
{
    const int n = 1024 + 123;                                         // Тестовый размер.
    std::vector<float> h(n);                                          // Создаём тестовый массив на CPU.

    for (int i = 0; i < n; ++i) {                                     // Заполняем массив значениями.
        h[i] = 1.0f + (i % 5) * 0.25f;                                // Дет. паттерн: 1.0,1.25,1.5,1.75,2.0...
    }

    float cpu_sum = std::accumulate(h.begin(), h.end(), 0.0f);        // Считаем сумму на CPU.
    float gpu_sum = gpu_reduce_sum(h);                                // Считаем сумму на GPU через редукцию.

    float diff = std::fabs(cpu_sum - gpu_sum);                        // Считаем абсолютную разницу.
    float eps = 1e-3f;                                                // Допуск для float.

    std::cout << "n       = " << n << "\n";                           // Печатаем размер.
    std::cout << "CPU sum = " << cpu_sum << "\n";                     // Печатаем сумму CPU.
    std::cout << "GPU sum = " << gpu_sum << "\n";                     // Печатаем сумму GPU.
    std::cout << "diff    = " << diff << "\n";                        // Печатаем разницу.

    if (diff < eps) {                                                 // Проверяем корректность в пределах допуска.
        std::cout << "OK: result matches (eps=" << eps << ")\n";      // Успех.
        return 0;                                                     // Нормальный выход.
    } else {                                                          // Иначе ошибка.
        std::cout << "FAIL: mismatch!\n";                             // Сообщаем о несоответствии.
        return 1;                                                     // Выход с ошибкой.
    }
}
