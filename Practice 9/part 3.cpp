%%writefile mpi_floyd.cpp

#include <mpi.h>         
#include <iostream>      
#include <vector>        
#include <random>        
#include <iomanip>       
#include <algorithm>     

int main(int argc, char** argv) {
    // -------------------- 1) Инициализация MPI --------------------
    MPI_Init(&argc, &argv);                           // Включаем MPI

    int rank = 0;                                     // Номер процесса
    int size = 1;                                     // Кол-во процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);             // Узнаём rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);             // Узнаём size

    // -------------------- 2) Считываем N из параметра --------------------
    // По требованию задания размер графа должен задаваться параметром программы.
    int N = 8;                                        // Значение по умолчанию
    if (argc >= 2) {                                  // Если пользователь передал N
        N = std::max(1, std::atoi(argv[1]));          // Читаем N, минимум 1
    }

    // -------------------- 3) Засекаем время выполнения --------------------
    // По инструкции: start_time в начале программы, end_time в конце.
    double start_time = MPI_Wtime();                  // Старт таймера

    // -------------------- 4) Распределяем строки матрицы по процессам --------------------
    // Матрица N×N. Делим по строкам.
    // base = минимум строк на процесс, rem = остаток (первые rem процессов получают +1 строку).
    int base = N / size;                              // База строк на процесс
    int rem  = N % size;                              // Остаток строк

    // countsRows[p] = число строк у процесса p
    // displsRows[p] = стартовая глобальная строка процесса p
    std::vector<int> countsRows(size);
    std::vector<int> displsRows(size);

    int offset = 0;                                   // Текущий старт строк
    for (int p = 0; p < size; ++p) {                  // Для каждого процесса
        countsRows[p] = base + ((p < rem) ? 1 : 0);    // У первых rem на 1 строку больше
        displsRows[p] = offset;                        // Стартовая строка
        offset += countsRows[p];                       // Смещаемся дальше
    }

    int local_rows = countsRows[rank];                // Число строк у текущего процесса
    int start_row  = displsRows[rank];                // Глобальный индекс первой строки у процесса

    // local_mat хранит local_rows строк по N элементов => local_rows*N элементов
    std::vector<double> local_mat(local_rows * N, 0.0);

    // full_mat будет хранить полную матрицу только на rank 0 (для генерации и финального вывода)
    std::vector<double> full_mat;

    // -------------------- 5) rank 0 создаёт матрицу смежности --------------------
    // Мы создаём взвешенный ориентированный граф без отрицательных весов.
    // INF обозначает отсутствие ребра.
    const double INF = 1e12;                          // Большое число вместо бесконечности

    if (rank == 0) {                                  // Только root создаёт матрицу
        full_mat.resize(N * N);                       // Выделяем память под N×N

        std::mt19937 rng(42);                         // Фиксированный seed
        std::uniform_real_distribution<double> wdist(1.0, 20.0); // Вес ребра 1..20
        std::uniform_real_distribution<double> pdist(0.0, 1.0);  // Для вероятности ребра

        double edge_prob = 0.35;                      // Вероятность существования ребра

        for (int i = 0; i < N; ++i) {                 // По строкам
            for (int j = 0; j < N; ++j) {             // По столбцам
                if (i == j) {
                    full_mat[i * N + j] = 0.0;        // Расстояние до себя = 0
                } else {
                    // С вероятностью edge_prob есть ребро, иначе INF
                    if (pdist(rng) < edge_prob) {
                        full_mat[i * N + j] = wdist(rng); // Вес ребра
                    } else {
                        full_mat[i * N + j] = INF;    // Ребра нет
                    }
                }
            }
        }
    }

    // -------------------- 6) Scatter строк матрицы по процессам --------------------
    // Требование задания: MPI_Scatter.
    // Чтобы корректно работать при любом числе процессов (и N может не делиться),
    // используем Scatterv (Scatter + учёт разных размеров).
    std::vector<int> sendCounts(size);                // Кол-во элементов (double) на процесс
    std::vector<int> sendDispls(size);                // Смещения в элементах
    for (int p = 0; p < size; ++p) {
        sendCounts[p] = countsRows[p] * N;            // строк * N элементов в строке
        sendDispls[p] = displsRows[p] * N;            // стартовая строка * N
    }

    MPI_Scatterv(
        (rank == 0 ? full_mat.data() : nullptr),      // Откуда (root)
        sendCounts.data(),                             // Сколько отправить каждому
        sendDispls.data(),                             // Смещения
        MPI_DOUBLE,                                    // Тип
        local_mat.data(),                               // Куда принять
        local_rows * N,                                 // Сколько принять
        MPI_DOUBLE,                                    // Тип
        0,                                             // root
        MPI_COMM_WORLD                                  // Коммуникатор
    );

    // -------------------- 7) Буфер для "глобальной" матрицы на каждом процессе --------------------
    // По требованию: обмен через MPI_Allgather (точнее Allgatherv, чтобы учесть остаток).
    // На каждой итерации k нам нужна текущая полная матрица расстояний, чтобы брать dist[k][j] и dist[i][k].
    // Мы будем поддерживать global_mat на каждом процессе.
    std::vector<double> global_mat(N * N, 0.0);       // У каждого процесса есть полная матрица

    // Сначала синхронизируем global_mat из текущих локальных блоков
    MPI_Allgatherv(
        local_mat.data(),                              // Отправляем локальный блок
        local_rows * N,                                // Его размер
        MPI_DOUBLE,                                    // Тип
        global_mat.data(),                             // Куда собираем полную матрицу
        sendCounts.data(),                             // Сколько принять от каждого
        sendDispls.data(),                             // Смещения
        MPI_DOUBLE,                                    // Тип
        MPI_COMM_WORLD                                 // Коммуникатор
    );

    // -------------------- 8) Алгоритм Флойда–Уоршелла --------------------
    // Формула:
    // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]) для всех k
    //
    // Здесь:
    // - каждый процесс обновляет ТОЛЬКО свои строки i (которые ему принадлежат)
    // - после обновления на итерации k, процессы обмениваются данными через MPI_Allgatherv
    for (int k = 0; k < N; ++k) {                     // k — промежуточная вершина
        // Для удобства получаем указатель на k-ю строку глобальной матрицы
        const double* row_k = &global_mat[k * N];     // row_k[j] = dist[k][j]

        for (int i_local = 0; i_local < local_rows; ++i_local) { // Перебор локальных строк
            int i_global = start_row + i_local;       // Глобальный индекс строки i
            double dik = global_mat[i_global * N + k];// dist[i][k] — нужно для всех j

            // Если dist[i][k] = INF, смысла обновлять нет (пути через k нет)
            if (dik >= INF / 2) {
                continue;                              // Пропускаем строку
            }

            // Обновляем все столбцы j
            for (int j = 0; j < N; ++j) {              // Перебор столбцов
                double dkj = row_k[j];                 // dist[k][j]
                if (dkj >= INF / 2) {                  // Если dist[k][j] = INF, тоже пропускаем
                    continue;
                }

                // Текущее значение dist[i][j] находится в local_mat для нашей строки
                double& dij_local = local_mat[i_local * N + j]; // Ссылка на элемент dist[i][j] локально

                // Кандидат через вершину k
                double candidate = dik + dkj;          // dist[i][k] + dist[k][j]

                // Если кандидат лучше — обновляем
                if (candidate < dij_local) {
                    dij_local = candidate;             // dist[i][j] = candidate
                }
            }
        }

        // После обновления локальных строк на итерации k,
        // нужно обменяться обновлёнными данными между процессами.
        // Требование: использовать MPI_Allgather.
        MPI_Allgatherv(
            local_mat.data(),                          // Отправляем обновлённый локальный блок
            local_rows * N,                            // Его размер
            MPI_DOUBLE,                                // Тип
            global_mat.data(),                         // Получаем обновлённую полную матрицу
            sendCounts.data(),                         // Сколько принять
            sendDispls.data(),                         // Смещения
            MPI_DOUBLE,                                // Тип
            MPI_COMM_WORLD                             // Коммуникатор
        );
    }

    // -------------------- 9) Собираем финальную матрицу на rank 0 --------------------
    // После всех итераций global_mat на каждом процессе содержит финальные расстояния.
    // Но по заданию нужно "соберите матрицу на rank=0 и выведите её".
    // Поэтому соберём локальные блоки обратно на root в full_mat.

    if (rank == 0) {
        full_mat.assign(N * N, 0.0);                  // Готовим буфер под полную матрицу
    }

    MPI_Gatherv(
        local_mat.data(),                              // Отправляем локальный блок
        local_rows * N,                                // Его размер
        MPI_DOUBLE,                                    // Тип
        (rank == 0 ? full_mat.data() : nullptr),       // Куда собирать (root)
        sendCounts.data(),                             // Сколько принять
        sendDispls.data(),                             // Смещения
        MPI_DOUBLE,                                    // Тип
        0,                                             // root
        MPI_COMM_WORLD                                 // Коммуникатор
    );

    // -------------------- 10) Засекаем конец времени --------------------
    double end_time = MPI_Wtime();                    // Конец таймера

    // -------------------- 11) Вывод результата и времени (только rank 0) --------------------
    if (rank == 0) {
        std::cout << "N = " << N << ", processes = " << size << "\n";
        std::cout << "Execution time: " << (end_time - start_time) << " seconds.\n\n";

        // Для больших N печатать всю матрицу не удобно, поэтому выводим только если N <= 20.
        if (N <= 20) {
            std::cout << "All-pairs shortest paths (Floyd-Warshall result):\n";
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double v = full_mat[i * N + j];   // dist[i][j]
                    if (v >= INF / 2) {
                        std::cout << std::setw(6) << "INF"; // Если нет пути
                    } else {
                        std::cout << std::setw(6) << std::fixed << std::setprecision(1) << v; // Иначе расстояние
                    }
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "Matrix is large (N > 20), skipping full print.\n";
        }
    }

    // -------------------- 12) Завершение MPI --------------------
    MPI_Finalize();                                   // Закрываем MPI
    return 0;                                         // Успех
}
