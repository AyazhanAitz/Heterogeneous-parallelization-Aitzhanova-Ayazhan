%%writefile main.cpp
#define CL_TARGET_OPENCL_VERSION 120                     
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cmath>

#define CHECK_CL(err, msg)                                \
if ((err) != CL_SUCCESS) {                                \
    std::cerr << (msg) << " Error: " << (err) << std::endl; \
    std::exit(EXIT_FAILURE);                              \
}

std::string loadKernel(const char* filename) {            // Функция загружает текст OpenCL-ядра из файла
    std::ifstream file(filename);                         // Открываем файл с исходником ядра
    return std::string(                                   // Возвращаем файл как строку
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
}

std::string getDeviceName(cl_device_id device) {          // Функция возвращает имя OpenCL-устройства
    size_t size = 0;                                      // Переменная для размера строки
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size); // Узнаём размер имени устройства
    std::vector<char> name(size);                         // Создаём буфер нужного размера
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr); // Читаем имя устройства
    return std::string(name.data());                      // Возвращаем имя как строку
}

std::string getDeviceType(cl_device_id device) {          // Функция возвращает тип устройства: CPU/GPU/OTHER
    cl_device_type type = 0;                              // Переменная под тип
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr); // Запрашиваем тип устройства
    if (type & CL_DEVICE_TYPE_GPU) return "GPU";          // Если GPU — возвращаем GPU
    if (type & CL_DEVICE_TYPE_CPU) return "CPU";          // Если CPU — возвращаем CPU
    return "OTHER";                                      // Иначе OTHER
}

int main() {                                              // Главная функция программы
    const int N = 1 << 20;                                // Размер массива (≈ 1 млн элементов)
    const size_t bytes = (size_t)N * sizeof(float);       // Размер массива в байтах

    std::vector<float> A(N), B(N), C(N), C_cpu(N);        // Векторы для данных на хосте

    for (int i = 0; i < N; i++) {                         // Инициализируем входные данные
        A[i] = i * 1.0f;                                  // Заполняем A
        B[i] = i * 2.0f;                                  // Заполняем B
    }

    // ---------------- CPU измерение ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Начало измерения CPU
    for (int i = 0; i < N; i++)                           // Последовательный цикл по массиву
        C_cpu[i] = A[i] + B[i];                           // Сложение на CPU
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Конец измерения CPU
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // CPU-время (мс)

    // ---------------- OpenCL: платформа и устройство ----------------
    cl_int err;                                           // Переменная для ошибок OpenCL
    cl_platform_id platform = nullptr;                    // Идентификатор платформы
    cl_device_id device = nullptr;                        // Идентификатор устройства

    err = clGetPlatformIDs(1, &platform, nullptr);        // Получаем первую доступную платформу
    CHECK_CL(err, "Platform error");                      // Проверяем ошибки

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr); // Получаем первое доступное устройство
    CHECK_CL(err, "Device error");                        // Проверяем ошибки

    std::cout << "OpenCL device type: " << getDeviceType(device) << std::endl; // Печать типа устройства
    std::cout << "OpenCL device name: " << getDeviceName(device) << std::endl; // Печать имени устройства

    // ---------------- OpenCL: контекст и очередь ----------------
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); // Создаём контекст
    CHECK_CL(err, "Context error");                       // Проверяем ошибки

    cl_command_queue queue = clCreateCommandQueue(        // Создаём командную очередь (OpenCL 1.2 вариант)
        context,                                          // Контекст
        device,                                           // Устройство
        CL_QUEUE_PROFILING_ENABLE,                        // Включаем профилирование
        &err                                              // Код ошибки
    );
    CHECK_CL(err, "Queue error");                         // Проверяем ошибки

    // ---------------- OpenCL: программа и ядро ----------------
    std::string kernelSource = loadKernel("kernel.cl");   // Загружаем исходник ядра из файла
    const char* src = kernelSource.c_str();               // Указатель на C-строку
    size_t length = kernelSource.size();                  // Длина исходника

    cl_program program = clCreateProgramWithSource(context, 1, &src, &length, &err); // Создаём программу
    CHECK_CL(err, "Program error");                       // Проверяем ошибки

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr); // Компилируем программу
    if (err != CL_SUCCESS) {                              // Если сборка не удалась
        char log[4096];                                   // Буфер для build log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr); // Получаем лог
        std::cerr << "Build log:\n" << log << std::endl;  // Печатаем лог
        return 1;                                         // Выходим с ошибкой
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err); // Создаём объект ядра
    CHECK_CL(err, "Kernel error");                        // Проверяем ошибки

    // ---------------- OpenCL: буферы ----------------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err); // Буфер A
    CHECK_CL(err, "Buffer A error");                      // Проверяем
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err); // Буфер B
    CHECK_CL(err, "Buffer B error");                      // Проверяем
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); // Буфер C
    CHECK_CL(err, "Buffer C error");                      // Проверяем

    // ---------------- OpenCL: total time (копирование + ядро + чтение) ----------------
    auto ocl_total_start = std::chrono::high_resolution_clock::now(); // Старт полного OpenCL времени

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, A.data(), 0, nullptr, nullptr); // Копируем A
    CHECK_CL(err, "Write A error");                       // Проверяем
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, B.data(), 0, nullptr, nullptr); // Копируем B
    CHECK_CL(err, "Write B error");                       // Проверяем

    // ---------------- OpenCL: аргументы ядра ----------------
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA); // Устанавливаем аргумент A
    CHECK_CL(err, "Set arg0 error");                      // Проверяем
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB); // Устанавливаем аргумент B
    CHECK_CL(err, "Set arg1 error");                      // Проверяем
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC); // Устанавливаем аргумент C
    CHECK_CL(err, "Set arg2 error");                      // Проверяем

    size_t globalSize = (size_t)N;                        // Количество work-items

    cl_event event;                                       // Событие для профилирования ядра
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, &event); // Запуск
    CHECK_CL(err, "Kernel launch error");                 // Проверяем

    clWaitForEvents(1, &event);                           // Ждём завершения ядра

    cl_ulong start = 0, end = 0;                          // Переменные для времени ядра
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr); // Старт ядра
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);       // Конец ядра
    double kernel_time = (end - start) * 1e-6;            // Время ядра в мс

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C.data(), 0, nullptr, nullptr); // Читаем C
    CHECK_CL(err, "Read C error");                        // Проверяем

    auto ocl_total_end = std::chrono::high_resolution_clock::now(); // Конец полного OpenCL времени
    double ocl_total_time = std::chrono::duration<double, std::milli>(ocl_total_end - ocl_total_start).count(); // Полное время

    // ---------------- Проверка корректности ----------------
    int mismatches = 0;                                   // Счётчик несовпадений
    for (int i = 0; i < N; i++) {                         // Проходим по всем элементам
        if (std::fabs(C[i] - C_cpu[i]) > 1e-6f) {         // Проверяем разницу
            mismatches++;                                 // Увеличиваем счётчик
            if (mismatches < 5)                           // Печатаем только первые 4 ошибки
                std::cout << "Mismatch at " << i << ": " << C[i] << " vs " << C_cpu[i] << std::endl; // Печать
        }
    }

    // ---------------- Вывод результатов ----------------
    std::cout << "CPU time (ms): " << cpu_time << std::endl;                 // Вывод времени CPU
    std::cout << "OpenCL kernel time (ms): " << kernel_time << std::endl;    // Вывод времени ядра
    std::cout << "OpenCL total time (ms): " << ocl_total_time << std::endl;  // Вывод полного времени OpenCL
    std::cout << "Mismatches: " << mismatches << std::endl;                  // Вывод числа ошибок

    std::ofstream csv("results.csv");                    // Создаём CSV для графика
    csv << "Metric,Time_ms\n";                           // Заголовок CSV
    csv << "CPU," << cpu_time << "\n";                   // CPU время
    csv << "OpenCL_kernel," << kernel_time << "\n";      // Время только ядра
    csv << "OpenCL_total," << ocl_total_time << "\n";    // Полное время OpenCL
    csv.close();                                         // Закрываем CSV

    // ---------------- Освобождение ресурсов ----------------
    clReleaseEvent(event);                                // Освобождаем event
    clReleaseMemObject(bufC);                             // Освобождаем bufC
    clReleaseMemObject(bufB);                             // Освобождаем bufB
    clReleaseMemObject(bufA);                             // Освобождаем bufA
    clReleaseKernel(kernel);                              // Освобождаем kernel
    clReleaseProgram(program);                            // Освобождаем program
    clReleaseCommandQueue(queue);                         // Освобождаем queue
    clReleaseContext(context);                            // Освобождаем context

    return 0;                                             // Успешный выход
}
