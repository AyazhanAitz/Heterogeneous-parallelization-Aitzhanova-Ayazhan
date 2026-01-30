%%writefile mpi_mean_std.cpp

#include <mpi.h>            
#include <iostream>         
#include <vector>           
#include <random>           
#include <cmath>            
#include <cstdlib>          
#include <algorithm>        

int main(int argc, char** argv) {
    // Инициализируем MPI-среду; после этого можно вызывать MPI-функции.
    MPI_Init(&argc, &argv);

    int rank = 0;           // Переменная для номера текущего процесса (rank).
    int size = 1;           // Переменная для общего количества процессов.

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Узнаём rank текущего процесса в коммуникаторе MPI_COMM_WORLD.
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Узнаём, сколько всего процессов запущено.

    // Задаём размер массива N; по умолчанию используем 10^6, как в примере на скрине.
    long long N = 1000000;

    // Если пользователь передал N в командной строке (например, ./a.out 2000000), читаем его.
    if (argc >= 2) {
        N = std::stoll(argv[1]);          // Преобразуем argv[1] (строку) в число типа long long.
        if (N < 0) N = 0;                 // На всякий случай: если ввели отрицательное — делаем 0.
    }

    // -------------------------------
    // 1) Распределение элементов по процессам (учёт остатка)
    // -------------------------------
    // Базовый размер блока (сколько элементов получит каждый процесс минимум).
    long long base = (size > 0) ? (N / size) : 0;

    // Остаток — сколько элементов "не поместилось" при равном делении.
    long long rem = (size > 0) ? (N % size) : 0;

    // Каждый процесс сам может вычислить, сколько элементов он должен получить:
    // первые rem процессов получают на 1 элемент больше.
    long long local_n = base + ((rank < rem) ? 1 : 0);

    // Локальный буфер для данных этого процесса (сюда MPI_Scatterv положит кусок массива).
    std::vector<double> local_data(static_cast<size_t>(local_n));

    // На rank 0 нам нужны:
    //  - полный исходный массив data размера N
    //  - массивы counts и displs для MPI_Scatterv
    std::vector<double> data;             // Полный массив (только на rank 0 будет заполнен).
    std::vector<int> counts;              // counts[i] = сколько элементов отправить процессу i.
    std::vector<int> displs;              // displs[i] = смещение (с какого индекса) для процесса i.

    if (rank == 0) {
        // Создаём полный массив на нулевом процессе.
        data.resize(static_cast<size_t>(N));

        // Создаём counts и displs размером size (по числу процессов).
        counts.resize(static_cast<size_t>(size));
        displs.resize(static_cast<size_t>(size));

        // Генератор случайных чисел:
        //  - seed делаем фиксированным для воспроизводимости (одинаковый результат при одинаковом N).
        std::mt19937 rng(42);

        // Распределение (например, равномерное) на отрезке [0, 1).
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Заполняем исходный массив случайными числами.
        for (long long i = 0; i < N; ++i) {
            data[static_cast<size_t>(i)] = dist(rng);  // Записываем очередное случайное число.
        }

        // Заполняем counts и displs для MPI_Scatterv.
        long long offset = 0;             // offset показывает текущий индекс начала блока.
        for (int p = 0; p < size; ++p) {  // Идём по всем процессам p = 0..size-1
            long long n_p = base + ((p < rem) ? 1 : 0); // Сколько элементов должен получить процесс p.
            counts[static_cast<size_t>(p)] = static_cast<int>(n_p); // counts в MPI обычно int.
            displs[static_cast<size_t>(p)] = static_cast<int>(offset); // Смещение для процесса p.
            offset += n_p;                 // Сдвигаем offset на размер выданного блока.
        }
    }

    // -------------------------------
    // 2) Scatter: раздаём части массива всем процессам (с учётом остатка)
    // -------------------------------
    // MPI_Scatterv позволяет раздать разное число элементов разным процессам.
    //  - sendbuf: исходные данные (только на rank 0, у остальных можно nullptr)
    //  - sendcounts: counts (только на rank 0)
    //  - displs: displacements (только на rank 0)
    //  - sendtype: MPI_DOUBLE (тип отправляемых элементов)
    //  - recvbuf: куда принимать на каждом процессе
    //  - recvcount: сколько элементов принимает текущий процесс (local_n)
    //  - recvtype: MPI_DOUBLE
    //  - root: 0 (кто раздаёт)
    //  - comm: MPI_COMM_WORLD
    MPI_Scatterv(
        (rank == 0 ? data.data() : nullptr),                 // Указатель на исходный массив (только root).
        (rank == 0 ? counts.data() : nullptr),               // Массив количеств (только root).
        (rank == 0 ? displs.data() : nullptr),               // Массив смещений (только root).
        MPI_DOUBLE,                                          // Тип исходных данных.
        local_data.data(),                                   // Куда принять локальный кусок.
        static_cast<int>(local_n),                            // Сколько элементов принять.
        MPI_DOUBLE,                                          // Тип принимаемых данных.
        0,                                                   // root процесс.
        MPI_COMM_WORLD                                       // Коммуникатор.
    );

    // -------------------------------
    // 3) Локальные вычисления: сумма и сумма квадратов
    // -------------------------------
    double local_sum = 0.0;        // Локальная сумма элементов.
    double local_sum_sq = 0.0;     // Локальная сумма квадратов элементов.

    for (long long i = 0; i < local_n; ++i) {                // Проходим по локальной части.
        double x = local_data[static_cast<size_t>(i)];       // Берём очередной элемент.
        local_sum += x;                                      // Добавляем в сумму.
        local_sum_sq += x * x;                               // Добавляем квадрат в сумму квадратов.
    }

    // -------------------------------
    // 4) Reduce: собираем локальные суммы на rank 0
    // -------------------------------
    double global_sum = 0.0;        // Глобальная сумма (итог на root).
    double global_sum_sq = 0.0;     // Глобальная сумма квадратов (итог на root).

    // Собираем суммы:
    MPI_Reduce(
        &local_sum,             // Отправляем адрес локальной суммы.
        &global_sum,            // Куда положить результат на root.
        1,                      // Количество элементов (одна сумма).
        MPI_DOUBLE,             // Тип данных.
        MPI_SUM,                // Операция редукции: суммирование.
        0,                      // root процесс.
        MPI_COMM_WORLD          // Коммуникатор.
    );

    // Собираем суммы квадратов:
    MPI_Reduce(
        &local_sum_sq,          // Отправляем адрес локальной суммы квадратов.
        &global_sum_sq,         // Куда положить результат на root.
        1,                      // Количество элементов.
        MPI_DOUBLE,             // Тип данных.
        MPI_SUM,                // Суммирование.
        0,                      // root процесс.
        MPI_COMM_WORLD          // Коммуникатор.
    );

    // -------------------------------
    // 5) rank 0 вычисляет среднее и стандартное отклонение
    // -------------------------------
    if (rank == 0) {
        // Если N = 0, то статистики не определены — обработаем аккуратно.
        if (N == 0) {
            std::cout << "N = 0, mean/stddev are undefined.\n"; // Сообщаем, что вычислять нечего.
        } else {
            // Среднее: mean = (1/N) * sum(x)
            double mean = global_sum / static_cast<double>(N);

            // По формуле со скрина:
            // sigma = sqrt( (1/N)*sum(x^2) - ( (1/N)*sum(x) )^2 )
            double ex2 = global_sum_sq / static_cast<double>(N);   // E[x^2]
            double ex = mean;                                      // E[x]
            double variance = ex2 - (ex * ex);                     // Var = E[x^2] - (E[x])^2

            // Из-за ошибок округления variance может получиться чуть отрицательной (например -1e-16),
            // поэтому подрезаем снизу до 0.
            variance = std::max(0.0, variance);

            double stddev = std::sqrt(variance);                   // Стандартное отклонение = sqrt(variance)

            // Выводим результаты.
            std::cout << "Processes: " << size << "\n";            // Сколько процессов участвовало.
            std::cout << "N: " << N << "\n";                       // Размер массива.
            std::cout << "Mean: " << mean << "\n";                 // Среднее значение.
            std::cout << "StdDev: " << stddev << "\n";             // Стандартное отклонение.
        }
    }

    // Завершаем работу MPI; после этого MPI-функции вызывать нельзя.
    MPI_Finalize();

    return 0; // Возвращаем код успешного завершения программы.
}
