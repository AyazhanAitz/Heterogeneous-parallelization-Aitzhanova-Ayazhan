%%writefile queue.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                              \
  cudaError_t err = (call);                                                \
  if (err != cudaSuccess) {                                                \
    std::printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err),       \
                __FILE__, __LINE__);                                       \
    std::exit(1);                                                          \
  }                                                                        \
} while (0)

// Очередь с атомарными head/tail (без кольцевого буфера)
struct Queue {                                                             // Структура очереди в device-памяти.
  int* data;                                                               // Буфер данных очереди.
  int  head;                                                               // Индекс головы (dequeue).
  int  tail;                                                               // Индекс хвоста (enqueue).
  int  capacity;                                                           // Максимальная ёмкость очереди.

  __device__ void init(int* buffer, int size) {                            // Инициализация очереди.
    data = buffer;                                                         // Привязываем буфер.
    head = 0;                                                              // Голова начинается с 0.
    tail = 0;                                                              // Хвост начинается с 0.
    capacity = size;                                                       // Сохраняем ёмкость.
  }

  __device__ bool enqueue(int value) {                                     // Добавление элемента в очередь.
    int pos = atomicAdd(&tail, 1);                                         // Атомарно резервируем позицию.
    if (pos < capacity) {                                                  // Проверяем переполнение.
      data[pos] = value;                                                   // Записываем элемент.
      return true;                                                         // Успех.
    }
    atomicSub(&tail, 1);                                                   // Откат tail при переполнении.
    return false;                                                          // Неуспех.
  }

  __device__ bool dequeue(int* value) {                                    // Удаление элемента из очереди.
    int pos = atomicAdd(&head, 1);                                         // Атомарно резервируем позицию головы.
    int t = tail;                                                          // Читаем текущий tail.
    if (pos < t) {                                                         // Если очередь не пуста.
      *value = data[pos];                                                  // Считываем элемент.
      return true;                                                         // Успех.
    }
    atomicSub(&head, 1);                                                   // Откат head при пустой очереди.
    return false;                                                          // Неуспех.
  }
};

__global__ void initQueueKernel(Queue* q, int* buffer, int cap) {          // Ядро инициализации очереди.
  if (blockIdx.x == 0 && threadIdx.x == 0) {                               // Выполняем одним потоком.
    q->init(buffer, cap);                                                  // Инициализируем очередь.
  }
}

__global__ void enqueueKernel(Queue* q, int nOps, int* ok) {               // Ядро параллельного enqueue.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;                         // Глобальный индекс потока.
  if (tid < nOps) {                                                        // Ограничение по числу операций.
    ok[tid] = q->enqueue(tid) ? 1 : 0;                                     // Пытаемся добавить элемент.
  }
}

__global__ void dequeueKernel(Queue* q, int nOps, int* vals, int* ok) {    // Ядро параллельного dequeue.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;                         // Глобальный индекс потока.
  if (tid < nOps) {                                                        // Ограничение по числу операций.
    int v = -1;                                                            // Значение по умолчанию.
    ok[tid] = q->dequeue(&v) ? 1 : 0;                                      // Пытаемся извлечь элемент.
    vals[tid] = v;                                                        // Сохраняем результат.
  }
}

int main() {                                                                // Главная функция программы.
  const int CAPACITY = 1024;                                                // Ёмкость очереди.
  const int N_OPS = 2048;                                                   // Количество операций.
  const int BLOCK = 256;                                                    // Размер блока.
  const int GRID = (N_OPS + BLOCK - 1) / BLOCK;                             // Количество блоков.

  Queue* d_q;                                                              // Очередь в device-памяти.
  int* d_buf;                                                              // Буфер очереди в device-памяти.
  int* d_enqOk;                                                            // Результаты enqueue.
  int* d_deqOk;                                                            // Результаты dequeue.
  int* d_vals;                                                             // Извлечённые значения.

  CUDA_CHECK(cudaMalloc(&d_q, sizeof(Queue)));                             // Выделяем память под Queue.
  CUDA_CHECK(cudaMalloc(&d_buf, CAPACITY * sizeof(int)));                  // Выделяем буфер данных.
  CUDA_CHECK(cudaMalloc(&d_enqOk, N_OPS * sizeof(int)));                   // Выделяем enqOk.
  CUDA_CHECK(cudaMalloc(&d_deqOk, N_OPS * sizeof(int)));                   // Выделяем deqOk.
  CUDA_CHECK(cudaMalloc(&d_vals, N_OPS * sizeof(int)));                    // Выделяем vals.

  initQueueKernel<<<1,1>>>(d_q, d_buf, CAPACITY);                          // Инициализация очереди.
  CUDA_CHECK(cudaDeviceSynchronize());                                     // Ждём завершения.

  enqueueKernel<<<GRID,BLOCK>>>(d_q, N_OPS, d_enqOk);                      // Параллельный enqueue.
  CUDA_CHECK(cudaDeviceSynchronize());                                     // Ждём завершения.

  dequeueKernel<<<GRID,BLOCK>>>(d_q, N_OPS, d_vals, d_deqOk);              // Параллельный dequeue.
  CUDA_CHECK(cudaDeviceSynchronize());                                     // Ждём завершения.

  int* enqOk = (int*)std::malloc(N_OPS * sizeof(int));                    // Host enqOk.
  int* deqOk = (int*)std::malloc(N_OPS * sizeof(int));                    // Host deqOk.
  int* vals  = (int*)std::malloc(N_OPS * sizeof(int));                    // Host vals.
  int* seen  = (int*)std::calloc(N_OPS, sizeof(int));                     // Массив для проверки дублей.

  CUDA_CHECK(cudaMemcpy(enqOk, d_enqOk, N_OPS * sizeof(int), cudaMemcpyDeviceToHost)); // Копируем enqOk.
  CUDA_CHECK(cudaMemcpy(deqOk, d_deqOk, N_OPS * sizeof(int), cudaMemcpyDeviceToHost)); // Копируем deqOk.
  CUDA_CHECK(cudaMemcpy(vals,  d_vals,  N_OPS * sizeof(int), cudaMemcpyDeviceToHost)); // Копируем vals.

  int enqSuccess = 0, deqSuccess = 0;                                     // Счётчики успешных операций.
  for (int i = 0; i < N_OPS; ++i) enqSuccess += enqOk[i];                // Считаем enqueue.
  for (int i = 0; i < N_OPS; ++i) deqSuccess += deqOk[i];                // Считаем dequeue.

  int duplicates = 0, outOfRange = 0;                                     // Счётчики ошибок.
  for (int i = 0; i < N_OPS; ++i) {                                      // Проверка корректности.
    if (deqOk[i]) {
      int v = vals[i];
      if (v < 0 || v >= N_OPS) outOfRange++;
      else if (++seen[v] > 1) duplicates++;
    }
  }

  std::printf("ENQUEUE success = %d\n", enqSuccess);                      // Вывод enqueue.
  std::printf("DEQUEUE success = %d\n", deqSuccess);                      // Вывод dequeue.
  std::printf("Duplicates     = %d\n", duplicates);                       // Вывод дублей.
  std::printf("Out of range   = %d\n", outOfRange);                       // Вывод ошибок диапазона.

  CUDA_CHECK(cudaFree(d_vals));                                           // Освобождаем device vals.
  CUDA_CHECK(cudaFree(d_deqOk));                                          // Освобождаем device deqOk.
  CUDA_CHECK(cudaFree(d_enqOk));                                          // Освобождаем device enqOk.
  CUDA_CHECK(cudaFree(d_buf));                                            // Освобождаем device буфер.
  CUDA_CHECK(cudaFree(d_q));                                              // Освобождаем device очередь.

  std::free(seen);                                                        // Освобождаем host seen.
  std::free(vals);                                                        // Освобождаем host vals.
  std::free(deqOk);                                                       // Освобождаем host deqOk.
  std::free(enqOk);                                                       // Освобождаем host enqOk.

  return 0;                                                               // Успешное завершение.
}
