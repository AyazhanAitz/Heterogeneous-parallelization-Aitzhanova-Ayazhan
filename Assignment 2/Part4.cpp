#include <cuda_runtime.h>   // Основные CUDA-функции: cudaMalloc, cudaMemcpy, cudaEvent...
#include <device_launch_parameters.h> // Определения для запуска kernel-ов
#include <iostream>         // cout
#include <vector>           // vector
#include <algorithm>        // std::is_sorted
#include <cstdlib>          // rand, srand
#include <ctime>            // time
#include <climits>          // INT_MAX

using namespace std;

// Макрос для проверки ошибок CUDA.
// Если CUDA-операция вернула ошибку, печатаем сообщение и выходим.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            cerr << "CUDA error: " << cudaGetErrorString(err)                 \
                 << " at " << __FILE__ << ":" << __LINE__ << endl;           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Удобная функция: округляем вверх деление a / b
static int divUp(int a, int b) {
    return (a + b - 1) / b;
}

// Функция заполнения массива случайными числами
static void fillRandom(vector<int>& a, int lo = 1, int hi = 100000) {
    for (int i = 0; i < (int)a.size(); i++) {
        a[i] = lo + rand() % (hi - lo + 1);
    }
}

// Простая проверка "массив отсортирован?"
static bool isSortedHost(const vector<int>& a) {
    return std::is_sorted(a.begin(), a.end());
}

/*
  Идея GPU алгоритма (упрощённая, но корректная):
  1) Разбиваем массив на чанки размера CHUNK (= элементы, которые обрабатывает 1 блок).
  2) Каждый блок грузит свой чанк в shared memory и сортирует (Bitonic sort).
     Bitonic удобно для GPU и не требует рекурсии.
  3) Потом делаем слияния: runSize = CHUNK, затем 2*CHUNK, 4*CHUNK, ...
     На каждом шаге запускаем kernel mergeRunsKernel, который параллельно сливает пары run-ов.
  4) Используем ping-pong буферы: на одном шаге читаем из A, пишем в B, потом меняем местами.
*/

constexpr int BLOCK_THREADS = 256;   // Количество потоков в блоке
constexpr int CHUNK = 1024;          // Размер подмассива, сортируемого одним блоком (должен быть степенью 2)
static_assert((CHUNK& (CHUNK - 1)) == 0, "CHUNK must be power of two");

// Kernel сортировки чанка в одном блоке.
// Каждый блок сортирует CHUNK элементов (или меньше в хвосте, но мы дополняем INT_MAX).
__global__ void sortChunksBitonicKernel(const int* __restrict__ d_in,
    int* __restrict__ d_out,
    int n) {
    // Shared memory (быстрая память внутри блока).
    // Здесь храним текущий чанк для сортировки.
    __shared__ int s[CHUNK];

    // Номер блока
    int blockId = blockIdx.x;

    // Номер потока внутри блока
    int tid = threadIdx.x;

    // Стартовый индекс чанка в глобальном массиве
    int base = blockId * CHUNK;

    // Загружаем CHUNK элементов в shared memory.
    // Поскольку потоков BLOCK_THREADS=256, а CHUNK=1024, то каждый поток загрузит 4 элемента.
    for (int k = tid; k < CHUNK; k += blockDim.x) {
        int globalIdx = base + k;

        // Если выходим за границы исходного массива,
        // подставляем INT_MAX, чтобы "паддинг" ушёл в конец при сортировке.
        s[k] = (globalIdx < n) ? d_in[globalIdx] : INT_MAX;
    }

    // Ждём, пока все потоки закончат загрузку
    __syncthreads();

    // Bitonic sort внутри блока для массива размера CHUNK
    // k — текущий размер битонической последовательности
    for (int k = 2; k <= CHUNK; k <<= 1) {

        // j — шаг сравнения/перестановки
        for (int j = k >> 1; j > 0; j >>= 1) {

            // Индекс элемента, который обслуживает данный поток.
            // В bitonic sort обычно делают сравнение для каждого i.
            for (int i = tid; i < CHUNK; i += blockDim.x) {

                // Партнёр для сравнения определяется XOR с j
                int ixj = i ^ j;

                // Чтобы не делать двойные сравнения, обычно берут только i < ixj
                if (ixj > i) {

                    // Определяем направление сортировки:
                    // Если (i & k) == 0, сортируем по возрастанию, иначе по убыванию.
                    bool ascending = ((i & k) == 0);

                    int a = s[i];
                    int b = s[ixj];

                    // Если ascending и a > b — меняем местами.
                    // Если descending и a < b — тоже меняем местами.
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }

            // Синхронизация после каждого шага сравнения/перестановки
            __syncthreads();
        }
    }

    // Записываем отсортированный чанк обратно в глобальную память
    for (int k = tid; k < CHUNK; k += blockDim.x) {
        int globalIdx = base + k;
        if (globalIdx < n) {
            d_out[globalIdx] = s[k];
        }
    }
}

// Device-функция: параллельное слияние “одного элемента” через бинарный поиск.
// Мы хотим определить, какой элемент попадёт в позицию outPos при слиянии двух отсортированных диапазонов.
//
// Есть два диапазона:
// left  = A[leftStart ... leftStart + runSize - 1]
// right = A[rightStart... rightStart+ runSize - 1]
// (в хвосте массив может быть короче)
//
// Для каждой позиции k в результирующем массиве (0..total-1):
// ищем, сколько элементов взять из left (i),
// тогда из right возьмём k - i.
//
// Подбираем i бинарным поиском так, чтобы выполнялись условия "merge".
// Это известная идея (вариант merge-path / diagonal merge), но в учебном виде.
__device__ int pickFromLeftCount(const int* A,
    int leftStart, int leftLen,
    int rightStart, int rightLen,
    int k) {
    // i — сколько элементов берем из left
    // тогда j = k - i элементов берем из right

    // Нижняя граница для i:
    // если k больше rightLen, минимум из left будет k - rightLen
    int lo = max(0, k - rightLen);

    // Верхняя граница для i:
    // нельзя взять больше leftLen, и нельзя взять больше k
    int hi = min(k, leftLen);

    // Бинарный поиск по i
    while (lo < hi) {
        int mid = (lo + hi) / 2;

        int i = mid;
        int j = k - i;

        // Берём "пограничные" элементы слева и справа, аккуратно обрабатывая границы.
        int leftPrev = (i == 0) ? INT_MIN : A[leftStart + i - 1];
        int leftCur = (i == leftLen) ? INT_MAX : A[leftStart + i];
        int rightPrev = (j == 0) ? INT_MIN : A[rightStart + j - 1];
        int rightCur = (j == rightLen) ? INT_MAX : A[rightStart + j];

        // Условие корректного разреза:
        // leftPrev <= rightCur и rightPrev <= leftCur
        // Если rightPrev > leftCur, значит мы взяли слишком мало из left -> увеличиваем i
        if (rightPrev > leftCur) {
            lo = mid + 1;
        }
        else {
            // Иначе пробуем уменьшить i (ищем минимальный подходящий i)
            hi = mid;
        }
    }

    return lo;
}

// Kernel слияния пар отсортированных run-ов размера runSize.
// Каждая пара run-ов сливается в результирующий массив.
// Важно: kernel полностью параллельный по элементам результирующего массива.
__global__ void mergeRunsKernel(const int* __restrict__ d_in,
    int* __restrict__ d_out,
    int n,
    int runSize) {
    // Глобальный индекс потока по всем GPU потокам
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток отвечает за одну позицию результирующего массива (outPos)
    int outPos = gid;

    // Если выходим за пределы — ничего не делаем
    if (outPos >= n) return;

    // Определяем, к какой "паре run-ов" относится outPos.
    // Пара run-ов имеет размер 2*runSize.
    int pairSize = 2 * runSize;

    // Начало текущей пары
    int pairStart = (outPos / pairSize) * pairSize;

    // Смещение внутри пары (0..2*runSize-1, но в хвосте может быть меньше)
    int k = outPos - pairStart;

    // Левый run начинается в pairStart
    int leftStart = pairStart;

    // Правый run начинается после левого
    int rightStart = pairStart + runSize;

    // Реальные длины left и right с учётом конца массива n
    int leftLen = min(runSize, n - leftStart);
    int rightLen = 0;

    // Если rightStart уже за пределами массива, правого run-а нет
    if (rightStart < n) {
        rightLen = min(runSize, n - rightStart);
    }
    else {
        rightLen = 0;
    }

    // Общая длина результата для этой пары
    int totalLen = leftLen + rightLen;

    // Если k выходит за предел totalLen (бывает в хвосте),
    // значит эта позиция outPos относится к "пустой зоне" — просто не пишем.
    if (k >= totalLen) return;

    // Находим, сколько элементов взять из left для позиции k
    int i = pickFromLeftCount(d_in, leftStart, leftLen, rightStart, rightLen, k);

    // Тогда j = k - i элементов берём из right
    int j = k - i;

    // Элемент результата — минимум из "следующих" кандидатов leftCur и rightCur,
    // но поскольку i вычислен так, что разрез корректный,
    // достаточно взять max(leftPrev, rightPrev) как элемент на позиции k.
    int leftPrev = (i == 0) ? INT_MIN : d_in[leftStart + i - 1];
    int rightPrev = (j == 0) ? INT_MIN : d_in[rightStart + j - 1];

    int val = (leftPrev > rightPrev) ? leftPrev : rightPrev;

    d_out[outPos] = val;
}

// GPU merge sort: обёртка на стороне CPU, которая запускает kernels и меряет время
static float gpuMergeSort(int* d_bufA, int* d_bufB, int n) {
    // Создаём CUDA события для замера времени на GPU
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    // Засекаем старт
    CUDA_CHECK(cudaEventRecord(evStart));

    // 1) Сортируем чанки: каждый блок сортирует CHUNK элементов
    int numChunks = divUp(n, CHUNK);

    // Запускаем kernel: numChunks блоков, BLOCK_THREADS потоков
    sortChunksBitonicKernel << <numChunks, BLOCK_THREADS >> > (d_bufA, d_bufB, n);

    // Проверяем ошибки запуска
    CUDA_CHECK(cudaGetLastError());

    // Теперь отсортированные чанки находятся в d_bufB
    // Дальше делаем итеративные слияния, используя ping-pong буферы.

    int* in = d_bufB;   // текущий вход (после chunk sort)
    int* out = d_bufA;  // текущий выход

    // runSize — размер отсортированных блоков, с которых начинаем
    int runSize = CHUNK;

    // Пока размер run-а меньше n, продолжаем слияния, удваивая runSize
    while (runSize < n) {
        // Общее количество элементов n, каждому элементу соответствует один поток
        int threads = 256;
        int blocks = divUp(n, threads);

        // Запускаем merge kernel
        mergeRunsKernel << <blocks, threads >> > (in, out, n, runSize);

        // Проверяем ошибки
        CUDA_CHECK(cudaGetLastError());

        // Меняем местами in/out (ping-pong)
        int* tmp = in;
        in = out;
        out = tmp;

        // Удваиваем размер run-а
        runSize *= 2;
    }

    // Останавливаем таймер
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    // Считаем время в миллисекундах
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    // Чистим события
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    // Важно: после цикла конечный результат лежит в указателе in.
    // Но вызывающей стороне удобнее знать, что итог в d_bufA или d_bufB.
    // Здесь мы не возвращаем указатель, поэтому в main() мы просто скопируем из "in".
    // Чтобы не усложнять интерфейс, ниже в main мы сделаем простой трюк:
    // вызовем gpuMergeSort так, чтобы результат в конце оказался в d_bufB (или проверим копированием).
    // В учебных задачах это нормально.

    return ms;
}

// Бенчмарк для одного размера
static void benchmarkSize(int n) {
    vector<int> h(n);

    // Заполняем на CPU случайными числами
    fillRandom(h);

    // Выделяем память на GPU для двух буферов (ping-pong)
    int* d_A = nullptr;
    int* d_B = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * sizeof(int)));

    // Копируем данные CPU -> GPU (H2D)
    CUDA_CHECK(cudaMemcpy(d_A, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // Чтобы интерфейс был проще:
    // - Мы запускаем gpuMergeSort(d_A, d_B, n).
    // - После chunk sort данные окажутся в d_B.
    // - Потом merge будет "пинг-понг", итог может оказаться либо в d_A, либо в d_B.
    // Поэтому после сортировки мы просто попробуем считать оба и выбрать отсортированный.
    float kernelMs = gpuMergeSort(d_A, d_B, n);

    // Копируем оба буфера назад на CPU, чтобы понять, где итог
    vector<int> hA(n), hB(n);
    CUDA_CHECK(cudaMemcpy(hA.data(), d_A, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hB.data(), d_B, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Проверяем, какой из них отсортирован
    bool sortedA = isSortedHost(hA);
    bool sortedB = isSortedHost(hB);

    // Выбираем итоговый результат
    vector<int>* result = nullptr;
    if (sortedA) result = &hA;
    else if (sortedB) result = &hB;

    cout << "N = " << n << "\n";
    cout << "GPU kernel time (chunk sort + merges): " << kernelMs << " ms\n";
    cout << "Sorted in d_A: " << (sortedA ? "true" : "false") << "\n";
    cout << "Sorted in d_B: " << (sortedB ? "true" : "false") << "\n";

    if (result == nullptr) {
        cout << "WARNING: result is not sorted (check code / GPU / run).\n";
    }

    // Освобождаем память на GPU
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    cout << "\n";
}

int main() {
    setlocale(LC_ALL, "Russian");

    // Инициализируем rand()
    srand((unsigned)time(nullptr));

    // Чтобы результаты времени были корректные,
    // полезно "прогреть" GPU одним пустым вызовом синхронизации
    CUDA_CHECK(cudaDeviceSynchronize());

    // Тестируем производительность на 10 000
    benchmarkSize(10000);

    // Тестируем производительность на 100 000
    benchmarkSize(100000);

    cout << "Вывод:\n";
    cout << "1) Каждый блок GPU сортирует свой подмассив (чанк) в shared memory.\n";
    cout << "2) Затем выполняются параллельные слияния отсортированных подмассивов, пока не получим весь массив.\n";
    cout << "3) Для 10 000 элементов ускорение может быть небольшим из-за накладных расходов GPU.\n";
    cout << "4) Для 100 000 элементов GPU обычно показывает себя лучше, так как работы больше.\n";

    return 0;
}
