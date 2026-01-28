%%writefile kernel.cl
__kernel void matmul(                                           // Объявляем OpenCL-ядро для матричного умножения
    __global const float* A,                                     // Указатель на матрицу A в глобальной памяти (размер N×M)
    __global const float* B,                                     // Указатель на матрицу B в глобальной памяти (размер M×K)
    __global float* C,                                           // Указатель на матрицу C в глобальной памяти (размер N×K)
    const int N,                                                 // Число строк матрицы A (и C)
    const int M,                                                 // Число столбцов A (и строк B)
    const int K                                                  // Число столбцов матрицы B (и C)
) {
    int row = get_global_id(0);                                  // Получаем индекс строки результирующей матрицы C
    int col = get_global_id(1);                                  // Получаем индекс столбца результирующей матрицы C

    if (row < N && col < K) {                                    // Проверяем границы, чтобы не выйти за размеры матрицы
        float sum = 0.0f;                                        // Переменная для накопления суммы произведений
        for (int i = 0; i < M; i++) {                            // Проходим по общей размерности M
            sum += A[row * M + i] * B[i * K + col];              // Считаем sum += A(row,i) * B(i,col)
        }
        C[row * K + col] = sum;                                  // Записываем результат в C(row,col)
    }
}
%%writefile main.cpp
#define CL_TARGET_OPENCL_VERSION 120                              // Устанавливаем версию OpenCL 1.2 (совместимо с Colab заголовками)
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cmath>

#define CHECK_CL(err, msg)                                        \
if ((err) != CL_SUCCESS) {                                        \
    std::cerr << (msg) << " Error: " << (err) << std::endl;       \
    std::exit(EXIT_FAILURE);                                      \
}

std::string loadKernel(const char* filename) {                    // Функция загружает текст ядра из файла kernel.cl
    std::ifstream file(filename);                                 // Открываем файл ядра
    return std::string(                                           // Возвращаем содержимое файла как строку
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

std::string getDeviceName(cl_device_id device) {                  // Функция возвращает имя OpenCL-устройства
    size_t size = 0;                                              // Переменная для размера строки имени
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size);   // Запрашиваем размер строки имени
    std::vector<char> name(size);                                 // Создаём буфер нужной длины
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr); // Считываем имя в буфер
    return std::string(name.data());                              // Возвращаем имя как std::string
}

std::string getDeviceType(cl_device_id device) {                  // Функция возвращает тип устройства (CPU/GPU/OTHER)
    cl_device_type type = 0;                                      // Переменная для типа устройства
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr); // Получаем тип устройства
    if (type & CL_DEVICE_TYPE_GPU) return "GPU";                  // Если есть флаг GPU — возвращаем "GPU"
    if (type & CL_DEVICE_TYPE_CPU) return "CPU";                  // Если есть флаг CPU — возвращаем "CPU"
    return "OTHER";                                               // Иначе возвращаем "OTHER"
}

void matmulCPU(const std::vector<float>& A,                       // CPU-функция матричного умножения: C = A×B
               const std::vector<float>& B,                       // B хранится в виде массива M×K
               std::vector<float>& C,                             // C хранится в виде массива N×K
               int N, int M, int K) {                             // Размеры матриц
    for (int r = 0; r < N; r++) {                                 // Идём по строкам C
        for (int c = 0; c < K; c++) {                             // Идём по столбцам C
            float sum = 0.0f;                                     // Накопитель суммы произведений
            for (int i = 0; i < M; i++) {                         // Идём по общей размерности M
                sum += A[r * M + i] * B[i * K + c];               // sum += A(r,i) * B(i,c)
            }
            C[r * K + c] = sum;                                   // Записываем C(r,c)
        }
    }
}

int main() {                                                      // Главная функция программы
    const int N = 256;                                             // Количество строк A и C
    const int M = 256;                                             // Количество столбцов A и строк B
    const int K = 256;                                             // Количество столбцов B и C

    const size_t bytesA = (size_t)N * M * sizeof(float);           // Размер буфера A в байтах
    const size_t bytesB = (size_t)M * K * sizeof(float);           // Размер буфера B в байтах
    const size_t bytesC = (size_t)N * K * sizeof(float);           // Размер буфера C в байтах

    std::vector<float> A(N * M);                                   // Создаём матрицу A (N×M) на хосте
    std::vector<float> B(M * K);                                   // Создаём матрицу B (M×K) на хосте
    std::vector<float> C(N * K, 0.0f);                             // Создаём матрицу C для OpenCL (N×K)
    std::vector<float> C_cpu(N * K, 0.0f);                         // Создаём матрицу C_cpu для CPU (N×K)

    for (int i = 0; i < N * M; i++) A[i] = (float)(i % 100) / 10.0f; // Заполняем A простыми значениями
    for (int i = 0; i < M * K; i++) B[i] = (float)(i % 100) / 20.0f; // Заполняем B простыми значениями

    auto cpu_start = std::chrono::high_resolution_clock::now();    // Старт замера CPU
    matmulCPU(A, B, C_cpu, N, M, K);                                // Запускаем матричное умножение на CPU
    auto cpu_end = std::chrono::high_resolution_clock::now();      // Конец замера CPU
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // CPU время (мс)

    cl_int err;                                                    // Переменная для ошибок OpenCL
    cl_platform_id platform = nullptr;                             // OpenCL платформа
    cl_device_id device = nullptr;                                 // OpenCL устройство

    err = clGetPlatformIDs(1, &platform, nullptr);                 // Получаем первую платформу
    CHECK_CL(err, "Platform error");                               // Проверяем ошибки

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr); // Получаем первое устройство
    CHECK_CL(err, "Device error");                                 // Проверяем ошибки

    std::cout << "OpenCL device type: " << getDeviceType(device) << std::endl; // Печатаем тип устройства
    std::cout << "OpenCL device name: " << getDeviceName(device) << std::endl; // Печатаем имя устройства

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); // Создаём контекст
    CHECK_CL(err, "Context error");                                // Проверяем ошибки

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); // Создаём очередь 1.2
    CHECK_CL(err, "Queue error");                                  // Проверяем ошибки

    std::string kernelSource = loadKernel("kernel.cl");            // Загружаем исходник ядра
    const char* src = kernelSource.c_str();                        // Преобразуем в C-строку
    size_t length = kernelSource.size();                           // Длина исходника

    cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &err); // Создаём OpenCL программу
    CHECK_CL(err, "Program error");                                // Проверяем ошибки

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // Компилируем
    if (err != CL_SUCCESS) {                                       // Если сборка не удалась
        char log[4096];                                            // Буфер под лог компиляции
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr); // Получаем build log
        std::cerr << "Build log:\n" << log << std::endl;           // Печатаем лог
        return 1;                                                  // Выходим с ошибкой
    }

    cl_kernel kernel = clCreateKernel(program, "matmul", &err);    // Создаём kernel matmul
    CHECK_CL(err, "Kernel error");                                 // Проверяем ошибки

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytesA, nullptr, &err); // Буфер A на устройстве
    CHECK_CL(err, "Buffer A error");                               // Проверяем
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytesB, nullptr, &err); // Буфер B на устройстве
    CHECK_CL(err, "Buffer B error");                               // Проверяем
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytesC, nullptr, &err); // Буфер C на устройстве
    CHECK_CL(err, "Buffer C error");                               // Проверяем

    auto ocl_total_start = std::chrono::high_resolution_clock::now(); // Старт полного времени OpenCL (копирование+ядро+чтение)

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytesA, A.data(), 0, nullptr, nullptr); // Копируем A на устройство
    CHECK_CL(err, "Write A error");                                // Проверяем
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytesB, B.data(), 0, nullptr, nullptr); // Копируем B на устройство
    CHECK_CL(err, "Write B error");                                // Проверяем

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);         // Аргумент 0: A
    CHECK_CL(err, "Set arg0 error");                                // Проверяем
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);         // Аргумент 1: B
    CHECK_CL(err, "Set arg1 error");                                // Проверяем
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);         // Аргумент 2: C
    CHECK_CL(err, "Set arg2 error");                                // Проверяем
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);               // Аргумент 3: N
    CHECK_CL(err, "Set arg3 error");                                // Проверяем
    err = clSetKernelArg(kernel, 4, sizeof(int), &M);               // Аргумент 4: M
    CHECK_CL(err, "Set arg4 error");                                // Проверяем
    err = clSetKernelArg(kernel, 5, sizeof(int), &K);               // Аргумент 5: K
    CHECK_CL(err, "Set arg5 error");                                // Проверяем

    size_t globalSize[2] = { (size_t)N, (size_t)K };                // Глобальная сетка: N×K work-items (по одному на каждый C[row,col])

    size_t localSize[2] = { 16, 16 };                               // Локальный размер группы (можно менять для эксперимента)

    cl_event event;                                                 // Событие для профилирования времени ядра
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, &event); // Запуск ядра 2D
    CHECK_CL(err, "Kernel launch error");                           // Проверяем

    clWaitForEvents(1, &event);                                     // Ждём завершения ядра

    cl_ulong start = 0, end = 0;                                    // Переменные для времени ядра
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr); // Старт
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);       // Конец
    double kernel_time = (end - start) * 1e-6;                      // Время ядра в мс

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytesC, C.data(), 0, nullptr, nullptr); // Читаем C на хост
    CHECK_CL(err, "Read C error");                                  // Проверяем

    auto ocl_total_end = std::chrono::high_resolution_clock::now(); // Конец полного времени OpenCL
    double ocl_total_time = std::chrono::duration<double, std::milli>(ocl_total_end - ocl_total_start).count(); // Полное время

    int mismatches = 0;                                             // Счётчик ошибок сравнения
    for (int i = 0; i < N * K; i++) {                               // Сравниваем каждый элемент результата
        if (std::fabs(C[i] - C_cpu[i]) > 1e-3f) {                   // Допуск чуть больше из-за float
            mismatches++;                                           // Увеличиваем счётчик ошибок
            if (mismatches < 5)                                     // Печатаем первые 4 несовпадения
                std::cout << "Mismatch at " << i << ": " << C[i] << " vs " << C_cpu[i] << std::endl; // Печать
        }
    }

    std::cout << "CPU time (ms): " << cpu_time << std::endl;        // Время CPU
    std::cout << "OpenCL kernel time (ms): " << kernel_time << std::endl; // Время только ядра на GPU/устройстве OpenCL
    std::cout << "OpenCL total time (ms): " << ocl_total_time << std::endl; // Полное время (копирование+ядро+чтение)
    std::cout << "Mismatches: " << mismatches << std::endl;         // Корректность

    double speedup_kernel = cpu_time / kernel_time;                 // Ускорение по чистому ядру
    double speedup_total  = cpu_time / ocl_total_time;              // Ускорение с учётом копирования

    std::cout << "Speedup (CPU / OpenCL kernel): " << speedup_kernel << "x" << std::endl; // Вывод ускорения kernel
    std::cout << "Speedup (CPU / OpenCL total):  " << speedup_total  << "x" << std::endl; // Вывод ускорения total

    std::ofstream csv("results.csv");                               // Сохраняем данные для графика
    csv << "Metric,Time_ms\n";                                      // Заголовок CSV
    csv << "CPU," << cpu_time << "\n";                              // CPU время
    csv << "OpenCL_kernel," << kernel_time << "\n";                 // OpenCL kernel время
    csv << "OpenCL_total," << ocl_total_time << "\n";               // OpenCL total время
    csv.close();                                                    // Закрываем CSV

    clReleaseEvent(event);                                          // Освобождаем event
    clReleaseMemObject(bufC);                                       // Освобождаем bufC
    clReleaseMemObject(bufB);                                       // Освобождаем bufB
    clReleaseMemObject(bufA);                                       // Освобождаем bufA
    clReleaseKernel(kernel);                                        // Освобождаем kernel
    clReleaseProgram(program);                                      // Освобождаем program
    clReleaseCommandQueue(queue);                                   // Освобождаем queue
    clReleaseContext(context);                                      // Освобождаем context

    return 0;                                                       // Успешно завершаем программу
}
