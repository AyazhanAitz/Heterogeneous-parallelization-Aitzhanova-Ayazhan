%%writefile main.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define CUDA_CHECK(call) do {                                                \
  cudaError_t err = (call);                                                  \
  if (err != cudaSuccess) {                                                  \
    std::cerr << "CUDA error: " << cudaGetErrorString(err)                   \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
    std::exit(1);                                                            \
  }                                                                          \
} while (0)

// Стек в глобальной памяти: atomics работают по адресу поля top в device-памяти. 
struct Stack {                                                     // Описываем структуру стека для GPU.
  int* data;                                                       // Указатель на буфер данных стека в device-памяти.
  int  top;                                                        // Индекс вершины (последний занятый индекс).
  int  capacity;                                                   // Максимальная ёмкость стека (кол-во элементов).

  __device__ void init(int* buffer, int size) {                    // Инициализация: привязываем буфер и задаём параметры.
    data = buffer;                                                 // Сохраняем адрес буфера данных.
    top = -1;                                                      // Ставим вершину на -1 (стек пуст).
    capacity = size;                                               // Сохраняем ёмкость стека.
  }

  __device__ bool push(int value) {                                // Push: добавляет элемент атомарно, предотвращая гонки.
    int oldTop = atomicAdd(&top, 1);                               // Атомарно увеличиваем top, получаем старое значение.
    int pos = oldTop + 1;                                          // Реальная позиция вставки — старое + 1.
    if (pos < capacity) {                                          // Проверяем, не вышли ли за ёмкость.
      data[pos] = value;                                           // Записываем значение в стек.
      return true;                                                 // Успешно добавили элемент.
    }
    atomicSub(&top, 1);                                            // Откатываем top обратно, если переполнение.
    return false;                                                  // Сообщаем, что push не прошёл.
  }

  __device__ bool pop(int* value) {                                // Pop: снимает элемент атомарно, предотвращая гонки.
    int oldTop = atomicSub(&top, 1);                               // Атомарно уменьшаем top, получаем старую вершину.
    int pos = oldTop;                                              // Реальная позиция чтения — старая вершина.
    if (pos >= 0) {                                                // Если стек не был пустым.
      *value = data[pos];                                          // Забираем значение с вершины.
      return true;                                                 // Успешно сняли элемент.
    }
    atomicAdd(&top, 1);                                            // Откатываем top, если стек был пуст (underflow).
    return false;                                                  // Сообщаем, что pop не прошёл.
  }
};

__global__ void initStackKernel(Stack* s, int* buffer, int cap) {  // Ядро инициализации стека на GPU.
  if (threadIdx.x == 0 && blockIdx.x == 0) {                       // Выполняем инициализацию только одним потоком.
    s->init(buffer, cap);                                          // Вызываем device-метод init.
  }
}

__global__ void pushKernel(Stack* s, int nPush, int* pushOk) {      // Ядро: много потоков параллельно делают push.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;                 // Считаем глобальный индекс потока.
  if (tid < nPush) {                                               // Ограничиваемся числом push-операций.
    bool ok = s->push(tid);                                        // Пытаемся запушить уникальное значение = tid.
    pushOk[tid] = ok ? 1 : 0;                                      // Сохраняем успех/неуспех для проверки на CPU.
  }
}

__global__ void popKernel(Stack* s, int nPop, int* popped, int* popOk) { // Ядро: много потоков параллельно делают pop.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;                      // Считаем глобальный индекс потока.
  if (tid < nPop) {                                                     // Ограничиваемся числом pop-операций.
    int val = -1;                                                       // Заготовка для снятого значения.
    bool ok = s->pop(&val);                                             // Пытаемся снять элемент.
    popped[tid] = val;                                                  // Пишем снятое значение (или -1 при неуспехе).
    popOk[tid] = ok ? 1 : 0;                                            // Пишем флаг успеха pop.
  }
}

int main() {                                                        // Точка входа: выделяем память, запускаем ядра, проверяем результат.
  const int CAPACITY = 1024;                                       // Задаём фиксированную ёмкость стека.
  const int N_THREADS = 2048;                                      // Задаём число потоков (push/pop попыток) больше capacity.
  const int BLOCK = 256;                                           // Выбираем размер блока для запуска CUDA-ядер.
  const int GRID = (N_THREADS + BLOCK - 1) / BLOCK;                // Считаем число блоков по формуле ceiling.

  Stack* d_stack = nullptr;                                        // Указатель на стек в device-памяти.
  int* d_buffer = nullptr;                                         // Указатель на буфер данных стека в device-памяти.
  int* d_pushOk = nullptr;                                         // Массив флагов успеха push в device-памяти.
  int* d_popOk = nullptr;                                          // Массив флагов успеха pop в device-памяти.
  int* d_popped = nullptr;                                         // Массив снятых значений (результат pop) в device-памяти.

  CUDA_CHECK(cudaMalloc(&d_stack, sizeof(Stack)));                 // Выделяем память под структуру Stack на GPU.
  CUDA_CHECK(cudaMalloc(&d_buffer, CAPACITY * sizeof(int)));       // Выделяем память под данные стека на GPU.
  CUDA_CHECK(cudaMalloc(&d_pushOk, N_THREADS * sizeof(int)));      // Выделяем память под push-статусы.
  CUDA_CHECK(cudaMalloc(&d_popOk, N_THREADS * sizeof(int)));       // Выделяем память под pop-статусы.
  CUDA_CHECK(cudaMalloc(&d_popped, N_THREADS * sizeof(int)));      // Выделяем память под массив popped.

  initStackKernel<<<1, 1>>>(d_stack, d_buffer, CAPACITY);          // Инициализируем стек на GPU (одним потоком).
  CUDA_CHECK(cudaGetLastError());                                  // Проверяем ошибки запуска ядра.
  CUDA_CHECK(cudaDeviceSynchronize());                             // Дожидаемся завершения инициализации.

  pushKernel<<<GRID, BLOCK>>>(d_stack, N_THREADS, d_pushOk);       // Параллельно выполняем push из множества потоков.
  CUDA_CHECK(cudaGetLastError());                                  // Проверяем ошибки запуска push-ядра.
  CUDA_CHECK(cudaDeviceSynchronize());                             // Дожидаемся завершения всех push.

  popKernel<<<GRID, BLOCK>>>(d_stack, N_THREADS, d_popped, d_popOk);// Параллельно выполняем pop из множества потоков.
  CUDA_CHECK(cudaGetLastError());                                  // Проверяем ошибки запуска pop-ядра.
  CUDA_CHECK(cudaDeviceSynchronize());                             // Дожидаемся завершения всех pop.

  Stack h_stack{};                                                 // Хост-копия стека для чтения финального top.
  std::vector<int> pushOk(N_THREADS);                              // Хост-массив статусов push.
  std::vector<int> popOk(N_THREADS);                               // Хост-массив статусов pop.
  std::vector<int> popped(N_THREADS);                              // Хост-массив снятых значений.

  CUDA_CHECK(cudaMemcpy(&h_stack, d_stack, sizeof(Stack), cudaMemcpyDeviceToHost)); // Копируем структуру стека на CPU.
  CUDA_CHECK(cudaMemcpy(pushOk.data(), d_pushOk, N_THREADS * sizeof(int), cudaMemcpyDeviceToHost)); // Копируем pushOk.
  CUDA_CHECK(cudaMemcpy(popOk.data(), d_popOk, N_THREADS * sizeof(int), cudaMemcpyDeviceToHost));   // Копируем popOk.
  CUDA_CHECK(cudaMemcpy(popped.data(), d_popped, N_THREADS * sizeof(int), cudaMemcpyDeviceToHost)); // Копируем popped.

  int pushed = 0;                                                  // Счётчик успешных push.
  for (int i = 0; i < N_THREADS; ++i) {                            // Проходим по всем попыткам push.
    pushed += pushOk[i];                                           // Суммируем флаги успеха.
  }

  int poppedCount = 0;                                             // Счётчик успешных pop.
  for (int i = 0; i < N_THREADS; ++i) {                            // Проходим по всем попыткам pop.
    poppedCount += popOk[i];                                       // Суммируем флаги успеха.
  }

  int expected = (N_THREADS < CAPACITY) ? N_THREADS : CAPACITY;    // Ожидаемое число успешных push (не больше capacity).

  std::vector<int> seen(N_THREADS, 0);                             // Массив для проверки уникальности снятых значений.
  int duplicates = 0;                                              // Счётчик дублей среди popped значений.
  int outOfRange = 0;                                              // Счётчик значений вне диапазона.
  for (int i = 0; i < N_THREADS; ++i) {                            // Проходим по всем снятым значениям.
    if (popOk[i] == 1) {                                           // Учитываем только успешные pop.
      int v = popped[i];                                           // Берём снятое значение.
      if (v < 0 || v >= N_THREADS) {                               // Проверяем, что v в допустимом диапазоне [0..N_THREADS-1].
        outOfRange += 1;                                           // Считаем ошибку диапазона.
      } else {                                                     // Если значение корректно по диапазону.
        seen[v] += 1;                                              // Отмечаем, что значение встретилось.
        if (seen[v] > 1) {                                         // Если значение встретилось больше 1 раза.
          duplicates += 1;                                         // Считаем дубль.
        }
      }
    }
  }

  std::cout << "CAPACITY           = " << CAPACITY << "\n";        // Печатаем ёмкость.
  std::cout << "N_THREADS (ops)    = " << N_THREADS << "\n";       // Печатаем число потоков/операций.
  std::cout << "PUSH success       = " << pushed << " (expected " << expected << ")\n"; // Печатаем успешные push.
  std::cout << "POP success        = " << poppedCount << " (expected " << expected << ")\n"; // Печатаем успешные pop.
  std::cout << "Final top          = " << h_stack.top << " (expected -1)\n"; // Печатаем финальный top.
  std::cout << "Popped duplicates  = " << duplicates << " (expected 0)\n";  // Печатаем число дублей.
  std::cout << "Popped outOfRange  = " << outOfRange << " (expected 0)\n";  // Печатаем число значений вне диапазона.

  bool okAll = true;                                               // Флаг общей корректности.
  okAll = okAll && (pushed == expected);                           // Проверяем, что push прошёл ровно expected раз.
  okAll = okAll && (poppedCount == expected);                      // Проверяем, что pop прошёл ровно expected раз.
  okAll = okAll && (h_stack.top == -1);                            // Проверяем, что стек пуст в конце.
  okAll = okAll && (duplicates == 0);                              // Проверяем отсутствие дублей.
  okAll = okAll && (outOfRange == 0);                              // Проверяем отсутствие значений вне диапазона.

  std::cout << "CORRECTNESS        = " << (okAll ? "PASS" : "FAIL") << "\n"; // Итоговая оценка.

  CUDA_CHECK(cudaFree(d_popped));                                  // Освобождаем popped на GPU.
  CUDA_CHECK(cudaFree(d_popOk));                                   // Освобождаем popOk на GPU.
  CUDA_CHECK(cudaFree(d_pushOk));                                  // Освобождаем pushOk на GPU.
  CUDA_CHECK(cudaFree(d_buffer));                                  // Освобождаем буфер стека на GPU.
  CUDA_CHECK(cudaFree(d_stack));                                   // Освобождаем структуру стека на GPU.

  return okAll ? 0 : 1;                                            // Возвращаем 0 при успехе, 1 при провале проверок.
}
