%%writefile mpi_gauss.cpp

#include <mpi.h>        // MPI_Init, MPI_Comm_rank, MPI_Scatterv, MPI_Bcast, MPI_Gatherv, ...
#include <iostream>     // std::cout, std::cerr
#include <vector>       // std::vector
#include <random>       // генерация случайных чисел
#include <cmath>        // std::fabs
#include <algorithm>    // std::max
#include <cstdlib>      // std::stoi

// Функция для вычисления глобального индекса строки для локальной строки i.
// local_to_global = start_row + i
static inline int localToGlobalRow(int start_row, int i_local) {
    return start_row + i_local;
}

int main(int argc, char** argv) {
    // -------------------- MPI init --------------------
    MPI_Init(&argc, &argv);                         // Инициализация MPI-среды

    int rank = 0;                                   // Номер текущего процесса
    int size = 1;                                   // Кол-во процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);           // Получаем rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);           // Получаем size

    // -------------------- N from args --------------------
    // Размерность системы N задаётся параметром программы (требование задания).
    int N = 8;                                      // Значение по умолчанию (если аргумент не дали)
    if (argc >= 2) {                                // Если есть аргумент
        N = std::stoi(argv[1]);                     // Читаем N из командной строки
        if (N <= 0) N = 1;                          // Защита: N минимум 1
    }

    // -------------------- Распределение строк --------------------
    // Нужно распределить строки между процессами, учитывая, что N может не делиться на size.
    // Для этого вычислим countsRows[p] = сколько строк у процесса p
    // и displsRows[p] = с какой глобальной строки начинается блок процесса p.

    int base = N / size;                            // Минимум строк на процесс
    int rem  = N % size;                            // Остаток (первые rem процессов получат +1 строку)

    std::vector<int> countsRows(size);              // Сколько строк у каждого процесса
    std::vector<int> displsRows(size);              // Смещения (start row) для каждого процесса

    int offset = 0;                                 // Текущий стартовый индекс строки
    for (int p = 0; p < size; ++p) {                // Заполняем распределение
        countsRows[p] = base + ((p < rem) ? 1 : 0);  // Первые rem процессов получают на 1 строку больше
        displsRows[p] = offset;                     // Стартовый индекс строк для процесса p
        offset += countsRows[p];                    // Сдвигаем offset
    }

    int local_rows = countsRows[rank];              // Сколько строк у текущего процесса
    int start_row  = displsRows[rank];              // С какой глобальной строки начинается блок

    // Каждая строка расширенной матрицы имеет длину (N + 1): [A | b]
    // Мы будем хранить локальный блок как подряд идущий массив:
    // local_aug[ i*(N+1) + j ] — элемент j в локальной строке i.
    std::vector<double> local_aug(local_rows * (N + 1), 0.0);

    // -------------------- Данные на root --------------------
    // На rank=0 создаём полную расширенную матрицу aug размером N × (N+1).
    std::vector<double> aug;                        // Полная расширенная матрица (только root)
    if (rank == 0) {
        aug.resize(N * (N + 1));                    // Выделяем память под расширенную матрицу

        std::mt19937 rng(42);                       // Фиксируем seed для воспроизводимости
        std::uniform_real_distribution<double> dist(-1.0, 1.0); // Случайные числа [-1,1]

        // Генерируем матрицу A и вектор b.
        // Чтобы метод Гаусса работал устойчивее, делаем матрицу диагонально доминируемой:
        // A[i][i] будет заметно больше суммы модулей внедиагональных элементов.
        for (int i = 0; i < N; ++i) {               // Проходим по строкам
            double row_sum_abs = 0.0;               // Сумма модулей внедиагональных элементов строки
            for (int j = 0; j < N; ++j) {           // Проходим по столбцам A
                double val = dist(rng);             // Генерируем случайное значение
                if (i != j) {                       // Если элемент не диагональный
                    row_sum_abs += std::fabs(val);  // Учитываем в сумме модулей
                }
                aug[i * (N + 1) + j] = val;         // Записываем A[i][j] в расширенную матрицу
            }
            // Усиливаем диагональ: A[i][i] = row_sum_abs + 1.0 (гарантирует доминирование)
            aug[i * (N + 1) + i] = row_sum_abs + 1.0;

            // Генерируем правую часть b (последний столбец расширенной матрицы)
            aug[i * (N + 1) + N] = dist(rng);       // b[i]
        }
    }

    // -------------------- Scatterv строк расширенной матрицы --------------------
    // Нам нужно раздать строки: каждый процесс получит local_rows строк,
    // то есть local_rows * (N+1) чисел double.
    // Scatterv требует counts/displs в единицах элементов, поэтому умножаем на (N+1).

    std::vector<int> sendCounts(size);              // Кол-во элементов (double) на процесс
    std::vector<int> sendDispls(size);              // Смещения (в double) на процесс
    for (int p = 0; p < size; ++p) {                // Пересчитываем из строк в элементы
        sendCounts[p] = countsRows[p] * (N + 1);    // Каждая строка даёт (N+1) double
        sendDispls[p] = displsRows[p] * (N + 1);    // Смещение по строкам → смещение по double
    }

    MPI_Scatterv(
        (rank == 0 ? aug.data() : nullptr),         // Откуда отправляем (только root)
        sendCounts.data(),                          // Сколько double отправить каждому
        sendDispls.data(),                          // С какого смещения (в double) отправить каждому
        MPI_DOUBLE,                                 // Тип данных
        local_aug.data(),                           // Куда принимаем локальную расширенную матрицу
        local_rows * (N + 1),                       // Сколько double принимает текущий процесс
        MPI_DOUBLE,                                 // Тип данных
        0,                                          // root
        MPI_COMM_WORLD                              // коммуникатор
    );

    // Буфер pivot-строки (рассылается всем процессам через MPI_Bcast).
    std::vector<double> pivot_row(N + 1, 0.0);      // Хранит текущую опорную строку [A[k,*] | b[k]]

    // -------------------- Прямой ход (Forward Elimination) --------------------
    // Для каждого k от 0 до N-1:
    // 1) root выбирает pivot-строку (частичный pivoting по столбцу k)
    // 2) root рассылает pivot_row всем процессам через MPI_Bcast
    // 3) каждый процесс обновляет свои локальные строки i > k:
    //    row_i = row_i - factor * pivot_row, где factor = row_i[k] / pivot_row[k]
    //
    // Важно: строки распределены, поэтому только тот процесс, который владеет строкой,
    // реально может её обновлять.

    for (int k = 0; k < N; ++k) {                   // Идём по каждому ведущему столбцу

        // --------- 1) root делает pivoting и готовит pivot_row ---------
        // Мы выполняем pivoting на root в полной матрице, но у нас сейчас данные распределены.
        // Поэтому на каждом шаге k:
        // - собираем текущий столбец k (частично) на root через MPI_Gatherv (упрощённый подход)
        // - root выбирает pivot index
        // - если pivot_row находится у другого процесса, root запрашивает у него строку
        //
        // Чтобы сделать код проще и понятнее для лабораторной:
        // Мы будем каждый шаг k собирать ВСЮ матрицу на root, делать pivoting,
        // затем снова Scatterv обновлённую матрицу обратно.
      
        // 1a) Сначала соберём локальные блоки в root, чтобы root имел актуальную матрицу.
        MPI_Gatherv(
            local_aug.data(),                       // Отправляем локальные данные
            local_rows * (N + 1),                   // Кол-во отправляемых double
            MPI_DOUBLE,                             // Тип
            (rank == 0 ? aug.data() : nullptr),      // Куда собирать (только root)
            sendCounts.data(),                      // Сколько принять от каждого процесса
            sendDispls.data(),                      // Смещения при приёме
            MPI_DOUBLE,                             // Тип
            0,                                      // root
            MPI_COMM_WORLD                          // коммуникатор
        );

        if (rank == 0) {                            // Только root выбирает pivot и формирует pivot_row
            // 1b) Ищем строку pivot_idx с максимальным |A[i][k]| на диапазоне i=k..N-1
            int pivot_idx = k;                      // Предположим pivot = k
            double best = std::fabs(aug[k * (N + 1) + k]); // Модуль текущего диагонального элемента

            for (int i = k + 1; i < N; ++i) {       // Ищем ниже по строкам
                double val = std::fabs(aug[i * (N + 1) + k]); // |A[i][k]|
                if (val > best) {                   // Если нашли лучше
                    best = val;                     // Обновляем лучшую величину
                    pivot_idx = i;                  // Запоминаем индекс строки
                }
            }

            // 1c) Если pivot_idx != k — меняем строки местами (pivoting)
            if (pivot_idx != k) {                   // Если нужно поменять
                for (int j = k; j <= N; ++j) {      // Меняем элементы от k до N (включая b)
                    std::swap(aug[k * (N + 1) + j], aug[pivot_idx * (N + 1) + j]); // swap
                }
            }

            // 1d) Копируем опорную строку k в pivot_row для рассылки
            for (int j = 0; j <= N; ++j) {          // j=0..N (N — это столбец b)
                pivot_row[j] = aug[k * (N + 1) + j];// pivot_row[j] = aug[k][j]
            }
        }

        // --------- 2) Рассылаем pivot_row всем процессам ---------
        // MPI_Bcast — обязательно по требованию задания.
        MPI_Bcast(
            pivot_row.data(),                       // Буфер pivot-строки
            N + 1,                                  // Длина pivot-строки (A + b)
            MPI_DOUBLE,                             // Тип данных
            0,                                      // root
            MPI_COMM_WORLD                          // коммуникатор
        );

        // Защита от деления на ноль: если pivot_row[k] почти 0, система вырождена/плохая.
        // Для лабораторной достаточно вывести сообщение.
        double pivot = pivot_row[k];                // Опорный элемент A[k][k] после pivoting
        if (std::fabs(pivot) < 1e-12) {             // Если слишком мал
            if (rank == 0) {                        // Сообщаем только на root
                std::cerr << "Pivot is too small at k=" << k << ". System may be singular.\n";
            }
            MPI_Finalize();                         // Корректно завершаем MPI
            return 1;                               // Завершаем программу с ошибкой
        }

        // --------- 3) Каждый процесс обновляет свои локальные строки i > k ---------
        // У каждого процесса есть local_rows строк, которые соответствуют глобальным индексам:
        // global_i = start_row + i_local
        for (int i_local = 0; i_local < local_rows; ++i_local) { // Перебираем локальные строки
            int global_i = localToGlobalRow(start_row, i_local);  // Находим глобальный индекс строки

            if (global_i <= k) {                    // Мы не трогаем строки выше/равные pivot строке
                continue;                           // Переходим к следующей строке
            }

            // factor = A[global_i][k] / pivot
            double factor = local_aug[i_local * (N + 1) + k] / pivot;

            // После вычитания в столбце k будет 0 (приблизительно), можно сразу присвоить 0
            local_aug[i_local * (N + 1) + k] = 0.0;

            // Обновляем оставшуюся часть строки: j = k+1 .. N (включая b)
            for (int j = k + 1; j <= N; ++j) {      // До N включительно (N — столбец b)
                local_aug[i_local * (N + 1) + j] -= factor * pivot_row[j]; // row -= factor*pivot_row
            }
        }

        // --------- 4) После обновления root должен снова раздать обновлённую матрицу ---------
        // Чтобы в следующей итерации k+1 root видел актуальную матрицу, мы:
        // - собираем обновлённую матрицу на root
        // - затем снова Scatterv, чтобы у всех процессов были свежие данные
        MPI_Gatherv(
            local_aug.data(),
            local_rows * (N + 1),
            MPI_DOUBLE,
            (rank == 0 ? aug.data() : nullptr),
            sendCounts.data(),
            sendDispls.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Scatterv(
            (rank == 0 ? aug.data() : nullptr),
            sendCounts.data(),
            sendDispls.data(),
            MPI_DOUBLE,
            local_aug.data(),
            local_rows * (N + 1),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );
    }

    // -------------------- Сбор результата на root --------------------
    // По заданию: "обратный ход: соберите результаты на rank=0 и завершите вычисления".
    // Поэтому собираем финальную верхнетреугольную матрицу и b на root.
    MPI_Gatherv(
        local_aug.data(),
        local_rows * (N + 1),
        MPI_DOUBLE,
        (rank == 0 ? aug.data() : nullptr),
        sendCounts.data(),
        sendDispls.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // -------------------- Обратный ход (Back Substitution) на root --------------------
    // На root есть верхнетреугольная система Ux = y (где y — обновлённый b).
    // Решаем:
    // x[N-1] = y[N-1]/U[N-1][N-1]
    // x[i] = (y[i] - sum_{j=i+1..N-1} U[i][j]*x[j]) / U[i][i]

    if (rank == 0) {
        std::vector<double> x(N, 0.0);              // Вектор решения

        for (int i = N - 1; i >= 0; --i) {          // Идём снизу вверх
            double rhs = aug[i * (N + 1) + N];      // Правая часть y[i]
            double diag = aug[i * (N + 1) + i];     // Диагональный элемент U[i][i]

            // Вычитаем вклад уже найденных x[j]
            for (int j = i + 1; j < N; ++j) {       // j справа от диагонали
                rhs -= aug[i * (N + 1) + j] * x[j]; // rhs -= U[i][j] * x[j]
            }

            if (std::fabs(diag) < 1e-12) {          // Если диагональ почти 0 — система плохая
                std::cerr << "Zero/near-zero diagonal at i=" << i << ". Cannot solve.\n";
                MPI_Finalize();
                return 1;
            }

            x[i] = rhs / diag;                      // x[i] = rhs / U[i][i]
        }

        // -------------------- Вывод решения --------------------
        std::cout << "N = " << N << "\n";           // Печатаем размерность
        std::cout << "Solution x:\n";               // Заголовок
        for (int i = 0; i < N; ++i) {               // Выводим каждый элемент решения
            std::cout << "x[" << i << "] = " << x[i] << "\n";
        }
    }

    // -------------------- MPI finalize --------------------
    MPI_Finalize();                                  // Завершаем MPI-среду
    return 0;                                        // Успешное завершение программы
}
