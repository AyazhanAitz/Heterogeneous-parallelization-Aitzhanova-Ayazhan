#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

// Функция вычисления нового значения элемента массива
// Вызывается одинаково во всех MPI-процессах
static inline double processValue(double x) {
    return std::sqrt(x * x + 1.0) + x * 0.5;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);                               // старт MPI

    int rank = 0;                                         // номер процесса
    int size = 0;                                         // всего процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                 // получить rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);                 // получить size

    const int N = 1'000'000;                              // размер массива
    std::vector<double> global_in;                        // вход на root
    std::vector<double> global_out;                       // выход на root

    std::vector<int> counts(size, 0);                     // сколько элементов у каждого процесса
    std::vector<int> displs(size, 0);                     // смещения (начальные индексы) для Scatterv/Gatherv

    // распределение N по процессам максимально ровно
    int base = N / size;                                  // минимум на процесс
    int rem  = N % size;                                  // остаток
    for (int p = 0; p < size; ++p) {
        counts[p] = base + (p < rem ? 1 : 0);             // первые rem процессов получают +1 элемент
    }
    displs[0] = 0;                                        // смещение первого процесса
    for (int p = 1; p < size; ++p) {
        displs[p] = displs[p - 1] + counts[p - 1];        // накопительное смещение
    }

    // локальный буфер каждого процесса
    std::vector<double> local_in(counts[rank]);           // кусок входа
    std::vector<double> local_out(counts[rank]);          // кусок выхода

    // root генерирует данные один раз
    if (rank == 0) {
        global_in.resize(N);
        global_out.resize(N);

        std::mt19937_64 rng(42);                          // фиксированный seed
        std::uniform_real_distribution<double> dist(0.0, 10.0);
        for (int i = 0; i < N; ++i) {
            global_in[i] = dist(rng);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);                          // синхронизация перед замером
    double t0 = MPI_Wtime();                              // старт общего времени

    // раздаём массив по процессам
    MPI_Scatterv(
        rank == 0 ? global_in.data() : nullptr,           // sendbuf на root, иначе nullptr
        counts.data(),                                    // сколько отправлять каждому
        displs.data(),                                    // откуда начинать отправку для каждого
        MPI_DOUBLE,                                       // тип данных
        local_in.data(),                                  // receive buffer
        counts[rank],                                     // сколько принять этому процессу
        MPI_DOUBLE,                                       // тип данных
        0,                                                // root
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);                          // синхронизация перед локальными вычислениями
    double t_comp0 = MPI_Wtime();                         // старт локальных вычислений

    // локальная обработка
    for (int i = 0; i < counts[rank]; ++i) {
        local_out[i] = processValue(local_in[i]);
    }

    double t_comp1 = MPI_Wtime();                         // конец локальных вычислений

    // собираем результат обратно на root
    MPI_Gatherv(
        local_out.data(),                                 // sendbuf
        counts[rank],                                     // сколько отправить
        MPI_DOUBLE,                                       // тип
        rank == 0 ? global_out.data() : nullptr,          // recvbuf на root
        counts.data(),                                    // сколько принять от каждого
        displs.data(),                                    // куда класть от каждого
        MPI_DOUBLE,                                       // тип
        0,                                                // root
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);                          // синхронизация после сборки
    double t1 = MPI_Wtime();                              // конец общего времени

    // max по времени вычислений среди процессов — честная оценка, т.к. ждать будут самых медленных
    double local_comp_ms = (t_comp1 - t_comp0) * 1000.0;  // локальное время (мс)
    double max_comp_ms = 0.0;
    MPI_Reduce(&local_comp_ms, &max_comp_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total_ms = (t1 - t0) * 1000.0;             // полное время (scatter + compute + gather)
        std::cout.setf(std::ios::fixed);
        std::cout.precision(3);

        std::cout << "N = " << N << "\n";
        std::cout << "Processes = " << size << "\n";
        std::cout << "Total time (scatter+compute+gather) = " << total_ms << " ms\n";
        std::cout << "Max compute time across ranks       = " << max_comp_ms << " ms\n";

        // быстрая проверка: посчитать контрольную сумму результата
        // чтобы видеть, что при разных np результат одинаковый
        long double checksum = 0.0;
        for (int i = 0; i < N; ++i) checksum += global_out[i];
        std::cout << "Checksum = " << (double)checksum << "\n";
    }

    MPI_Finalize();                                       // завершение MPI
    return 0;
}
