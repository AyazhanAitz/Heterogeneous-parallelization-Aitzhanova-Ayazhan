%%writefile main.cu
#include <cuda_runtime.h>     
#include <iostream>           
#include <vector>             
#include <iomanip>            
#include <cmath>              

// ----------------------------- Макрос для проверки ошибок CUDA -----------------------------
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

// ----------------------------- Параметр "плохого доступа" -----------------------------
// STRIDE влияет на то, насколько "разбросанно" потоки читают память.
// Чем STRIDE больше, тем хуже коалесцирование (обычно).
constexpr int STRIDE = 32;

// ----------------------------- Kernel 1: коалесцированный доступ -----------------------------
// Поток с глобальным индексом idx обрабатывает элемент idx.
// Потоки warp-а читают подряд лежащие элементы => коалесцированные транзакции памяти.
__global__ void kernel_coalesced(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // глобальный индекс элемента
    if (idx < n) {                                      // защита от выхода за границы
        float x = in[idx];                              // чтение подряд -> эффективно
        out[idx] = x * 2.0f + 1.0f;                     // простая "обработка" (чтобы компилятор не выкинул)
    }
}

// ----------------------------- Kernel 2: некоалесцированный доступ -----------------------------
// Поток idx читает элемент j = (idx * STRIDE) % n.
// Внутри warp потоки обращаются к адресам далеко друг от друга => больше транзакций памяти => медленнее.
__global__ void kernel_uncoalesced_stride(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    // глобальный индекс потока
    if (idx < n) {                                      // защита от выхода за границы
        int j = (idx * STRIDE) % n;                     // "плохое" отображение индексов на память
        float x = in[j];                                // чтение со страйдом -> плохо
        out[j] = x * 2.0f + 1.0f;                       // пишем туда же (тут нет гонки, потому что j уникален при n кратном, но для простоты — ок)
    }
}

// ----------------------------- Kernel 3: оптимизация через shared memory -----------------------------
// Идея: в shared memory грузим блок данных коалесцированно (каждый поток берёт соседний элемент),
// затем делаем вычисления из shared (быстро), потом записываем обратно коалесцированно.
__global__ void kernel_shared_tiled(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float tile[];                     // динамический shared memory (размер задаём при запуске)
    int tid = threadIdx.x;                              // индекс потока внутри блока
    int base = blockIdx.x * blockDim.x;                 // стартовый индекс блока в глобальном массиве
    int idx = base + tid;                               // глобальный индекс элемента для данного потока

    if (idx < n) {                                      // если в пределах массива
        tile[tid] = in[idx];                            // коалесцированная загрузка в shared
    } else {
        tile[tid] = 0.0f;                               // безопасное значение, если вышли за границу
    }

    __syncthreads();                                    // синхронизация: все потоки должны загрузить tile

    if (idx < n) {                                      // снова проверяем границы
        float x = tile[tid];                            // чтение из shared memory (быстрее, чем global)
        // маленькое вычисление + зависимость от соседнего элемента, чтобы показать пользу shared
        float left = (tid > 0) ? tile[tid - 1] : x;     // сосед слева (внутри блока)
        out[idx] = (x + left) * 1.5f;                   // результат
    }
}

// ----------------------------- Kernel 4: изменение организации потоков -----------------------------
// Вместо "1 поток -> 1 элемент" делаем "1 поток -> 2 элемента".
// Это снижает overhead планирования/индексации и увеличивает работу на поток.
// При коалесцированном доступе это часто помогает (не всегда, но полезный приём).
__global__ void kernel_coalesced_two_per_thread(const float* __restrict__ in, float* __restrict__ out, int n) {
    int globalThread = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока
    int i0 = globalThread * 2;                                 // первый индекс элемента для потока
    int i1 = i0 + 1;                                           // второй индекс элемента для потока

    if (i0 < n) {                                              // проверка границы для первого
        float x0 = in[i0];                                     // коалесцированное чтение
        out[i0] = x0 * 2.0f + 1.0f;                            // запись результата
    }
    if (i1 < n) {                                              // проверка границы для второго
        float x1 = in[i1];                                     // коалесцированное чтение
        out[i1] = x1 * 2.0f + 1.0f;                            // запись результата
    }
}

// ----------------------------- Функция измерения времени cudaEvent -----------------------------
// Возвращает среднее время ядра (в миллисекундах) по repeats запускам.
template <typename KernelFunc, typename... Args>
float measure_kernel_ms(KernelFunc kernel, dim3 grid, dim3 block, std::size_t sharedBytes, int repeats, Args... args) {
    cudaEvent_t start, stop;                                   // события CUDA для тайминга
    CUDA_CHECK(cudaEventCreate(&start));                        // создаём event start
    CUDA_CHECK(cudaEventCreate(&stop));                         // создаём event stop

    // Прогрев: один запуск, чтобы убрать "первый запуск" из измерений
    kernel<<<grid, block, sharedBytes>>>(args...);              // запуск ядра
    CUDA_CHECK(cudaGetLastError());                             // проверяем ошибки запуска
    CUDA_CHECK(cudaDeviceSynchronize());                        // ждём завершения

    CUDA_CHECK(cudaEventRecord(start));                         // ставим отметку времени "start" в поток
    for (int r = 0; r < repeats; ++r) {                         // несколько повторов для усреднения
        kernel<<<grid, block, sharedBytes>>>(args...);          // запускаем ядро
    }
    CUDA_CHECK(cudaEventRecord(stop));                          // ставим отметку времени "stop"
    CUDA_CHECK(cudaEventSynchronize(stop));                     // ждём, пока stop действительно наступит

    float ms = 0.0f;                                            // сюда запишем время
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));         // ms = время между start и stop
    ms /= static_cast<float>(repeats);                          // делим на число повторов

    CUDA_CHECK(cudaEventDestroy(start));                        // уничтожаем event start
    CUDA_CHECK(cudaEventDestroy(stop));                         // уничтожаем event stop

    return ms;                                                  // возвращаем среднее время одного запуска
}

// ----------------------------- Host проверка корректности -----------------------------
// Сравниваем несколько значений (не весь массив, чтобы не тратить время).
bool check_some_values(const std::vector<float>& ref, const std::vector<float>& got, int step) {
    for (std::size_t i = 0; i < ref.size(); i += step) {        // проверяем каждый step-ый элемент
        float a = ref[i];                                       // эталон
        float b = got[i];                                       // полученное
        if (std::fabs(a - b) > 1e-3f) {                         // допуск для float
            std::cerr << "Mismatch at i=" << i
                      << " ref=" << a << " got=" << b << "\n";  // печатаем ошибку
            return false;                                       // возвращаем false
        }
    }
    return true;                                                // всё ок
}

int main() {
    // ----------------------------- Настройки эксперимента -----------------------------
    const int n = 1 << 26;                                      // ~67 млн элементов (большой массив для заметного времени)
    const int repeats = 20;                                     // сколько раз повторять ядро для среднего времени
    const int blockSize = 256;                                  // размер блока (часто хорошее значение)
    const int gridSize = (n + blockSize - 1) / blockSize;       // количество блоков для 1 элемент/поток

    // Для "2 элемента на поток" нужно меньше блоков (потоков в 2 раза меньше):
    const int gridSize2 = ( (n + 2 * blockSize - 1) / (2 * blockSize) );

    // ----------------------------- Инициализация host данных -----------------------------
    std::vector<float> h_in(n);                                 // входной массив на CPU
    std::vector<float> h_out(n, 0.0f);                          // выходной массив на CPU
    std::vector<float> h_ref(n, 0.0f);                          // эталон для проверки

    for (int i = 0; i < n; ++i) {                               // заполняем вход
        h_in[i] = static_cast<float>(i % 1024) * 0.001f;        // простой паттерн (детерминированный)
    }

    // Эталон для coalesced (out[i] = in[i] * 2 + 1)
    for (int i = 0; i < n; ++i) {                               // считаем эталон на CPU
        h_ref[i] = h_in[i] * 2.0f + 1.0f;                       // формула как в kernel_coalesced
    }

    // ----------------------------- Выделение памяти на GPU -----------------------------
    float* d_in = nullptr;                                      // указатель на GPU вход
    float* d_out = nullptr;                                     // указатель на GPU выход
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));           // выделяем память под вход
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));          // выделяем память под выход

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // копируем вход на GPU

    // ----------------------------- Печать параметров -----------------------------
    std::cout << std::fixed << std::setprecision(3);            // формат вывода
    std::cout << "N elements = " << n << "\n";                  // печатаем размер
    std::cout << "Block size = " << blockSize << "\n";          // печатаем block
    std::cout << "Grid  size (1 el/thread) = " << gridSize << "\n";
    std::cout << "Grid  size (2 el/thread) = " << gridSize2 << "\n";
    std::cout << "STRIDE = " << STRIDE << "\n\n";

    // ----------------------------- 1) Coalesced kernel -----------------------------
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));        // очищаем выход
    float t_coal = measure_kernel_ms(kernel_coalesced, dim3(gridSize), dim3(blockSize), 0, repeats, d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат на CPU
    bool ok1 = check_some_values(h_ref, h_out, 1 << 20);        // проверка по редким точкам

    // ----------------------------- 2) Uncoalesced stride kernel -----------------------------
    // Внимание: это ядро пишет out[j], где j = (idx*STRIDE)%n. Оно корректно как "демонстрация",
    // но эталон для него другой (мы тут сравним только скорость и факт, что результат не NaN).
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));        // очищаем выход
    float t_uncoal = measure_kernel_ms(kernel_uncoalesced_stride, dim3(gridSize), dim3(blockSize), 0, repeats, d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат

    // Простая sanity-проверка: убедимся, что в массиве есть ненулевые значения
    bool ok2 = false;                                           // флаг "похоже, что ядро что-то сделало"
    for (int i = 0; i < n; i += (1 << 22)) {                    // редкие проверки
        if (h_out[i] != 0.0f) {                                 // если нашли ненулевой элемент
            ok2 = true;                                         // значит ядро работало
            break;                                              // выходим
        }
    }

    // ----------------------------- 3) Shared memory tiled kernel -----------------------------
    // Shared bytes = blockSize * sizeof(float), потому что tile[blockSize]
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));        // очищаем выход
    float t_shared = measure_kernel_ms(kernel_shared_tiled, dim3(gridSize), dim3(blockSize),
                                       blockSize * sizeof(float), repeats, d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат

    // Эталон для shared kernel другой (там out[i] = (in[i] + left)*1.5), поэтому просто sanity:
    bool ok3 = false;                                           // флаг sanity
    for (int i = 1; i < n; i += (1 << 22)) {                    // редкие проверки
        if (h_out[i] != 0.0f) {                                 // если не ноль
            ok3 = true;                                         // ок
            break;                                              // выходим
        }
    }

    // ----------------------------- 4) Thread organization: 2 elements per thread -----------------------------
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));        // очищаем выход
    float t_two = measure_kernel_ms(kernel_coalesced_two_per_thread, dim3(gridSize2), dim3(blockSize), 0, repeats, d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // копируем результат
    bool ok4 = check_some_values(h_ref, h_out, 1 << 20);        // сверяем с эталоном (формула совпадает)

    // ----------------------------- Печать результатов -----------------------------
    std::cout << "Kernel timings (ms, lower is better):\n";
    std::cout << "  1) Coalesced                 : " << t_coal   << " ms  (correct=" << (ok1 ? "yes" : "no") << ")\n";
    std::cout << "  2) Uncoalesced (stride)      : " << t_uncoal << " ms  (sanity="  << (ok2 ? "yes" : "no") << ")\n";
    std::cout << "  3) Shared memory (tiled)     : " << t_shared << " ms  (sanity="  << (ok3 ? "yes" : "no") << ")\n";
    std::cout << "  4) Coalesced (2 el/thread)   : " << t_two    << " ms  (correct=" << (ok4 ? "yes" : "no") << ")\n\n";

    // ----------------------------- Скорости в GB/s (приближённо) -----------------------------
    // Для 1) и 4): читаем 4 байта и пишем 4 байта на элемент => ~8*N байт трафика.
    // Для 2) тоже ~8*N, но из-за паттерна памяти эффективность ниже.
    // Это грубая оценка: реальный трафик зависит от кешей/транзакций, но для отчёта подходит.
    auto gbps = [&](float ms, double bytesMoved) -> double {    // лямбда для перевода ms->GB/s
        double sec = ms / 1000.0;                               // переводим миллисекунды в секунды
        double gb = bytesMoved / 1e9;                           // переводим байты в гигабайты (10^9)
        return gb / sec;                                        // GB/s
    };

    const double bytes_per_element = 8.0;                       // 4 байта чтение + 4 байта запись
    const double bytes_total = bytes_per_element * static_cast<double>(n); // общий трафик в байтах (приближённо)

    std::cout << "Approx effective throughput (GB/s):\n";
    std::cout << "  Coalesced               : " << gbps(t_coal, bytes_total) << "\n";
    std::cout << "  Uncoalesced (stride)    : " << gbps(t_uncoal, bytes_total) << "\n";
    std::cout << "  Shared (tiled)          : " << gbps(t_shared, bytes_total) << "  (note: extra shared ops)\n";
    std::cout << "  2 el/thread (coalesced) : " << gbps(t_two, bytes_total) << "\n\n";

    // ----------------------------- Освобождение памяти -----------------------------
    CUDA_CHECK(cudaFree(d_in));                                 // освобождаем вход на GPU
    CUDA_CHECK(cudaFree(d_out));                                // освобождаем выход на GPU

    return 0;                                                   // успешный выход
}
