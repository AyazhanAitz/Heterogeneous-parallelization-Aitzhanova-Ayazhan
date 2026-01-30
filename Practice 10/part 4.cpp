%%writefile main.cpp

#include <mpi.h>                 
#include <iostream>              
#include <vector>                
#include <string>                
#include <cstring>               
#include <algorithm>             
#include <numeric>               
#include <iomanip>               
#include <cstdint>               
#include <cmath>                 

// ----------------------------- Парсинг аргументов командной строки -----------------------------
struct Args {
    std::string mode = "strong";     // "strong" или "weak"
    std::string op = "sum";          // "sum" | "min" | "max"
    std::string comm = "reduce";     // "reduce" | "allreduce"
    int64_t n_global = 100000000;    // глобальный размер массива для strong scaling
    int64_t n_per_rank = 25000000;   // размер на процесс для weak scaling
    int repeats = 3;                 // сколько повторов для усреднения
};

// парсер: ожидаем флаги вида --key value
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& k) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << k << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            return std::string(argv[++i]);
        };

        if (key == "--mode") a.mode = need_value(key);
        else if (key == "--op") a.op = need_value(key);
        else if (key == "--comm") a.comm = need_value(key);
        else if (key == "--n") a.n_global = std::stoll(need_value(key));
        else if (key == "--n_per_rank") a.n_per_rank = std::stoll(need_value(key));
        else if (key == "--repeats") a.repeats = std::stoi(need_value(key));
        else {
            std::cerr << "Unknown arg: " << key << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    return a;
}

// ----------------------------- Генерация данных локального массива -----------------------------
// Чтобы не тратить время на rand(), делаем детерминированную формулу.
// Это важно для “чистоты” измерения compute vs communication.
inline double value_at(int64_t global_index) {
    // Псевдоданные: небольшие колебания, чтобы min/max тоже имели смысл
    // (не константа и не строго монотонная последовательность).
    return (global_index % 1000) * 0.001 + ((global_index % 7) - 3) * 1e-4;
}

// ----------------------------- Локальные агрегаты -----------------------------
double local_sum(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s;
}

double local_min(const std::vector<double>& x) {
    double m = x.empty() ? 0.0 : x[0];
    for (double v : x) m = std::min(m, v);
    return m;
}

double local_max(const std::vector<double>& x) {
    double m = x.empty() ? 0.0 : x[0];
    for (double v : x) m = std::max(m, v);
    return m;
}

// ----------------------------- Главная программа -----------------------------
int main(int argc, char** argv) {
    // Инициализируем MPI (должно быть самым первым, до любых MPI_* вызовов).
    MPI_Init(&argc, &argv);

    // Узнаём ранг процесса (id) и общее число процессов.
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Парсим аргументы (одинаковые у всех процессов).
    Args args = parse_args(argc, argv);

    // ----------------------------- Определяем локальный размер задачи -----------------------------
    // Strong scaling:
    //  - фиксируем N_global
    //  - N_local = N_global / P (плюс остаток распределяем по первым rank)
    //
    // Weak scaling:
    //  - фиксируем N_per_rank
    //  - N_global = N_per_rank * P
    int64_t n_global = 0;
    int64_t n_local = 0;
    int64_t start = 0; // глобальный индекс начала данных этого процесса (для генерации)

    if (args.mode == "strong") {
        n_global = args.n_global;

        // Равномерное разбиение + распределение остатка
        int64_t base = n_global / size;
        int64_t rem  = n_global % size;

        // Первые rem процессов получают на 1 элемент больше
        n_local = base + (rank < rem ? 1 : 0);

        // Вычисляем стартовый индекс через префиксную сумму размеров предыдущих рангов
        // (простая формула: rank*base + min(rank, rem))
        start = static_cast<int64_t>(rank) * base + std::min<int64_t>(rank, rem);

    } else if (args.mode == "weak") {
        n_local = args.n_per_rank;
        n_global = n_local * size;
        start = static_cast<int64_t>(rank) * n_local;

    } else {
        if (rank == 0) std::cerr << "Unknown mode: " << args.mode << " (use strong|weak)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ----------------------------- Готовим локальный массив -----------------------------
    // Важный момент: в реальных задачах данные могут приходить из файла/сети,
    // но здесь мы их генерируем детерминированно, чтобы фокус был на scaling и MPI коммуникациях.
    std::vector<double> x;
    x.resize(static_cast<size_t>(n_local));

    // Генерацию данных считаем “compute” частью (последовательной внутри ранга).
    // Для чистоты эксперимента можно вынести генерацию вне замера, но в отчёте полезно видеть её вклад.
    double t_gen0 = MPI_Wtime();
    for (int64_t i = 0; i < n_local; ++i) {
        x[static_cast<size_t>(i)] = value_at(start + i);
    }
    double t_gen1 = MPI_Wtime();
    double gen_time = t_gen1 - t_gen0;

    // ----------------------------- Выбор операции и MPI-коллектива -----------------------------
    // Поддерживаем op: sum/min/max
    // comm: reduce/allreduce
    const bool use_allreduce = (args.comm == "allreduce");
    if (!(args.comm == "reduce" || args.comm == "allreduce")) {
        if (rank == 0) std::cerr << "Unknown comm: " << args.comm << " (use reduce|allreduce)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Op mpi_op = MPI_SUM;
    if (args.op == "sum") mpi_op = MPI_SUM;
    else if (args.op == "min") mpi_op = MPI_MIN;
    else if (args.op == "max") mpi_op = MPI_MAX;
    else {
        if (rank == 0) std::cerr << "Unknown op: " << args.op << " (use sum|min|max)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ----------------------------- Прогрев -----------------------------
    // Один “холостой” прогон, чтобы убрать шум первого вызова.
    double warm_local = 0.0;
    if (args.op == "sum") warm_local = local_sum(x);
    if (args.op == "min") warm_local = local_min(x);
    if (args.op == "max") warm_local = local_max(x);

    double warm_global = 0.0;
    if (use_allreduce) {
        MPI_Allreduce(&warm_local, &warm_global, 1, MPI_DOUBLE, mpi_op, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&warm_local, &warm_global, 1, MPI_DOUBLE, mpi_op, 0, MPI_COMM_WORLD);
    }

    // ----------------------------- Основные измерения -----------------------------
    // Мы хотим раздельно:
    // 1) compute_time: локальный подсчёт агрегата
    // 2) comm_time: время коллективной операции Reduce/Allreduce
    // 3) total_time: compute + comm (с barrier для честного сравнения)
    double compute_sum = 0.0;
    double comm_sum = 0.0;
    double total_sum = 0.0;

    // Важно: синхронизация перед каждым замером — иначе один процесс может стартовать раньше,
    // и времена “поплывут”.
    for (int r = 0; r < args.repeats; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);

        double t_total0 = MPI_Wtime();

        // ---- compute stage ----
        double t_comp0 = MPI_Wtime();

        double local_val = 0.0;
        if (args.op == "sum") local_val = local_sum(x);
        if (args.op == "min") local_val = local_min(x);
        if (args.op == "max") local_val = local_max(x);

        double t_comp1 = MPI_Wtime();
        double comp_time = t_comp1 - t_comp0;

        // ---- communication stage ----
        MPI_Barrier(MPI_COMM_WORLD); // отделяем compute от comm (полезно для анализа)

        double t_comm0 = MPI_Wtime();

        double global_val = 0.0;
        if (use_allreduce) {
            // Allreduce возвращает результат всем процессам (часто нужно, но дороже по коммуникациям).
            MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, mpi_op, MPI_COMM_WORLD);
        } else {
            // Reduce возвращает результат только на root=0 (часто дешевле, если результат нужен только там).
            MPI_Reduce(&local_val, &global_val, 1, MPI_DOUBLE, mpi_op, 0, MPI_COMM_WORLD);
        }

        double t_comm1 = MPI_Wtime();
        double comm_time = t_comm1 - t_comm0;

        MPI_Barrier(MPI_COMM_WORLD);

        double t_total1 = MPI_Wtime();
        double total_time = t_total1 - t_total0;

        // Мы хотим “время программы” как максимум по всем процессам (критический путь).
        // Поэтому делаем Allreduce по max для времён.
        double comp_max = 0.0, comm_max = 0.0, total_max = 0.0;
        MPI_Allreduce(&comp_time, &comp_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&total_time, &total_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        compute_sum += comp_max;
        comm_sum += comm_max;
        total_sum += total_max;
    }

    double compute_avg = compute_sum / args.repeats;
    double comm_avg   = comm_sum   / args.repeats;
    double total_avg  = total_sum  / args.repeats;

    // ----------------------------- Strong scaling метрики -----------------------------
    // Для strong scaling:
    //  Speedup(P) = T(1) / T(P)
    //  Efficiency(P) = Speedup(P) / P
    //
    // Чтобы посчитать speedup, нужно знать baseline T(1).
    // В Colab проще: запусти программу отдельно с -np 1 и вручную сравни.
    // Но мы также можем “передать” T(1) через аргумент, чтобы программа сама считала — это усложнит.
    // Здесь выводим времена; speedup/eff считаешь по таблице (в отчёте так обычно и делают).

    // ----------------------------- Вывод результатов -----------------------------
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "================ MPI Scalability Report ================\n";
        std::cout << "mode      : " << args.mode << "\n";
        std::cout << "op        : " << args.op << "\n";
        std::cout << "comm      : " << args.comm << "\n";
        std::cout << "processes : " << size << "\n";
        std::cout << "N_global  : " << n_global << "\n";
        std::cout << "N_local~  : " << n_local << " (rank0 example)\n";
        std::cout << "repeats   : " << args.repeats << "\n\n";

        std::cout << "Timing (seconds, avg critical-path across ranks):\n";
        std::cout << "  gen_time (rank0 only, data init) = " << gen_time << "\n";
        std::cout << "  compute_avg                      = " << compute_avg << "\n";
        std::cout << "  comm_avg                         = " << comm_avg << "\n";
        std::cout << "  total_avg                        = " << total_avg << "\n\n";

    }

    // ----------------------------- Завершение -----------------------------
    MPI_Finalize();
    return 0;
}
