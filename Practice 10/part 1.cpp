%%writefile main.cpp
#include <omp.h>                 
#include <iostream>              
#include <vector>                
#include <random>                
#include <numeric>               
#include <iomanip>               
#include <cmath>                 
#include <algorithm>             

int main() {
    const std::size_t N = 50'000'000;      // размер массива (50 млн) 
    const int WARMUP_RUNS = 1;             // прогрев (чтобы сгладить влияние первого запуска/кеша)
    const int MEASURE_RUNS = 3;            // сколько раз мерить и усреднять (меньше шума)
    const int MAX_THREADS_TO_TEST = 16;    // сколько потоков максимум тестируем (можно увеличить)

    // ----------------------------- Переменные для профилирования -----------------------------
    double t0 = 0.0;                       // сюда будем сохранять время начала сегмента
    double t1 = 0.0;                       // сюда будем сохранять время конца сегмента

    // ----------------------------- Выделение памяти под массив -----------------------------
    t0 = omp_get_wtime();                  // старт измерения времени инициализации/подготовки
    std::vector<double> a;                 // объявляем вектор для данных
    a.resize(N);                           // выделяем N элементов типа double
    t1 = omp_get_wtime();                  // конец измерения подготовки памяти
    const double alloc_time = (t1 - t0);   // время выделения памяти (условно последовательная часть)

    // ----------------------------- Заполнение массива данными -----------------------------
    // Важно: генератор случайных чисел в параллели делать сложнее; для честности сделаем заполнение последовательно.
    // Это также создаёт заметную "последовательную часть" для анализа закона Амдала.
    t0 = omp_get_wtime();                  // старт измерения заполнения
    std::mt19937_64 rng(123456);           // фиксированный seed для воспроизводимости
    std::uniform_real_distribution<double> dist(0.0, 1.0); // равномерное распределение [0..1]
    for (std::size_t i = 0; i < N; ++i) {  // обычный последовательный цикл по массиву
        a[i] = dist(rng);                  // записываем случайное число в массив
    }
    t1 = omp_get_wtime();                  // конец заполнения
    const double init_time = (t1 - t0);    // время инициализации данных (последовательная часть)

    // ----------------------------- Последовательный расчёт (baseline) -----------------------------
    // Считаем сумму, среднее и дисперсию последовательно — это базовая линия для speedup.
    t0 = omp_get_wtime();                  // старт измерения последовательных вычислений

    double sum_seq = 0.0;                  // сумма (последовательно)
    for (std::size_t i = 0; i < N; ++i) {  // идём по массиву
        sum_seq += a[i];                   // накапливаем сумму
    }

    const double mean_seq = sum_seq / static_cast<double>(N); // среднее значение

    double var_acc_seq = 0.0;              // аккумулятор для суммы квадратов отклонений
    for (std::size_t i = 0; i < N; ++i) {  // второй проход по массиву
        const double d = a[i] - mean_seq;  // отклонение от среднего
        var_acc_seq += d * d;              // добавляем квадрат отклонения
    }

    const double var_seq = var_acc_seq / static_cast<double>(N); // дисперсия (population variance)
    const double std_seq = std::sqrt(var_seq);                   // стандартное отклонение

    t1 = omp_get_wtime();                  // конец измерения
    const double compute_seq_time = (t1 - t0); // время последовательной вычислительной части

    // Полное "последовательное время программы" (если считать всё как один сценарий выполнения):
    const double total_seq_time = alloc_time + init_time + compute_seq_time;

    // ----------------------------- Вывод baseline -----------------------------
    std::cout << std::fixed << std::setprecision(6); // фиксированный формат и точность
    std::cout << "N = " << N << "\n";                // печатаем размер массива
    std::cout << "Sequential results:\n";            // заголовок
    std::cout << "  sum  = " << sum_seq << "\n";     // сумма
    std::cout << "  mean = " << mean_seq << "\n";    // среднее
    std::cout << "  var  = " << var_seq << "\n";     // дисперсия
    std::cout << "  std  = " << std_seq << "\n";     // стандартное отклонение

    std::cout << "\nSequential timing (seconds):\n";                 // заголовок времени
    std::cout << "  alloc_time        = " << alloc_time << "\n";     // выделение памяти
    std::cout << "  init_time         = " << init_time << "\n";      // заполнение массива
    std::cout << "  compute_seq_time  = " << compute_seq_time << "\n"; // вычисления sum/mean/var
    std::cout << "  total_seq_time    = " << total_seq_time << "\n"; // суммарно

    // ----------------------------- Таблица экспериментов по числу потоков -----------------------------
    std::cout << "\nOpenMP experiments:\n"; // заголовок
    std::cout << "threads"
              << std::setw(16) << "compute_par_s"
              << std::setw(12) << "speedup"
              << std::setw(12) << "eff"
              << std::setw(16) << "p_amdahl"
              << std::setw(18) << "S_amdahl_pred"
              << "\n"; // шапка таблицы

    // Будем тестировать 1, 2, 4, 8, ... и также все значения до MAX_THREADS_TO_TEST (можно проще, но так наглядно).
    // Сделаем набор: 1..MAX_THREADS_TO_TEST.
    for (int threads = 1; threads <= MAX_THREADS_TO_TEST; ++threads) {
        omp_set_num_threads(threads); // задаём, сколько потоков использовать в следующем OpenMP регионе

        // ----------------------------- Прогрев (warm-up) -----------------------------
        for (int w = 0; w < WARMUP_RUNS; ++w) {        // несколько прогревочных прогонов
            double warm_sum = 0.0;                     // сумма для прогрева
            #pragma omp parallel for reduction(+:warm_sum) // параллельный цикл с редукцией суммы
            for (std::size_t i = 0; i < N; ++i) {      // каждый поток берёт часть индексов
                warm_sum += a[i];                      // накапливаем сумму в редукцию
            }
            (void)warm_sum;                            // подавляем предупреждение о неиспользуемой переменной
        }

        // ----------------------------- Измерения (усреднение по нескольким прогонам) -----------------------------
        double compute_par_sum_time = 0.0;             // сюда накопим время параллельной вычислительной части

        double sum_par = 0.0;                          // итоговая сумма (параллельная)
        double mean_par = 0.0;                         // итоговое среднее (параллельное)
        double var_par = 0.0;                          // итоговая дисперсия (параллельная)

        for (int r = 0; r < MEASURE_RUNS; ++r) {       // несколько измерительных прогонов
            t0 = omp_get_wtime();                      // старт измерения

            // 1) Параллельная сумма
            sum_par = 0.0;                             // обнуляем сумму перед редукцией
            #pragma omp parallel for reduction(+:sum_par) // редукция суммы по потокам
            for (std::size_t i = 0; i < N; ++i) {      // параллельный проход по массиву
                sum_par += a[i];                       // каждый поток добавляет свою часть
            }

            // 2) Среднее (после редукции — это уже одно значение)
            mean_par = sum_par / static_cast<double>(N); // считаем среднее

            // 3) Параллельная дисперсия (второй проход)
            double var_acc_par = 0.0;                  // аккумулятор квадратов отклонений
            #pragma omp parallel for reduction(+:var_acc_par) // редукция суммы квадратов
            for (std::size_t i = 0; i < N; ++i) {      // второй параллельный проход
                const double d = a[i] - mean_par;      // отклонение от среднего
                var_acc_par += d * d;                  // добавляем квадрат отклонения
            }
            var_par = var_acc_par / static_cast<double>(N); // дисперсия

            t1 = omp_get_wtime();                      // конец измерения
            compute_par_sum_time += (t1 - t0);         // добавляем время этого прогона в сумму
        }

        const double compute_par_time = compute_par_sum_time / static_cast<double>(MEASURE_RUNS); // среднее время

        // ----------------------------- Валидация результата -----------------------------
        // Проверим, что параллельные результаты близки к последовательным (иначе измерения бессмысленны).
        const double eps = 1e-8;                       // допуск на сравнение (для double)
        const bool ok_sum = std::fabs(sum_par - sum_seq) <= eps * std::max(1.0, std::fabs(sum_seq)); // сравнение суммы
        const bool ok_var = std::fabs(var_par - var_seq) <= 1e-7 * std::max(1.0, std::fabs(var_seq)); // сравнение дисперсии

        if (!ok_sum || !ok_var) {                      // если расхождение слишком большое
            std::cout << "WARNING: mismatch at threads=" << threads
                      << " ok_sum=" << ok_sum
                      << " ok_var=" << ok_var << "\n"; // предупреждаем
        }

        // ----------------------------- Speedup / Efficiency -----------------------------
        // Speedup считаем по вычислительной части: baseline = compute_seq_time, параллельное = compute_par_time.
        const double speedup = compute_seq_time / compute_par_time; // ускорение
        const double efficiency = speedup / static_cast<double>(threads); // эффективность

        // ----------------------------- Оценка доли параллельной части (Amdahl) -----------------------------
        // Закон Амдала: S(N) = 1 / ((1-p) + p/N).
        // Выразим p через измеренный S и N: p = (1 - 1/S) / (1 - 1/N).
        double p_amdahl = 0.0;                         // оценка доли параллельной части
        if (threads > 1) {                             // формула корректна для N>1
            const double invS = 1.0 / speedup;         // 1/S
            const double denom = (1.0 - 1.0 / static_cast<double>(threads)); // (1 - 1/N)
            p_amdahl = (1.0 - invS) / denom;           // p по Амдалу
            if (p_amdahl < 0.0) p_amdahl = 0.0;        // ограничим снизу из-за шума измерений
            if (p_amdahl > 1.0) p_amdahl = 1.0;        // ограничим сверху
        } else {
            p_amdahl = 0.0;                            // при 1 потоке "оценка p" не нужна
        }

        // Предсказанное ускорение по Амдалу для данного N и оцененного p:
        const double S_pred = (threads > 1)
            ? (1.0 / ((1.0 - p_amdahl) + (p_amdahl / static_cast<double>(threads))))
            : 1.0;                                     // для 1 потока ожидаем 1

        // ----------------------------- Печать строки таблицы -----------------------------
        std::cout << std::setw(7) << threads
                  << std::setw(16) << compute_par_time
                  << std::setw(12) << speedup
                  << std::setw(12) << efficiency
                  << std::setw(16) << p_amdahl
                  << std::setw(18) << S_pred
                  << "\n";
    }

    // ----------------------------- Итоговый анализ последовательной/параллельной доли -----------------------------
    // Если считать "всю программу" как alloc + init + compute, то доля параллельной части примерно:
    // parallel_fraction_total ~ compute_seq_time / total_seq_time
    // (потому что alloc/init мы сделали последовательно).
    const double parallel_fraction_total = compute_seq_time / total_seq_time;  // грубая оценка доли "параллелизуемого"
    const double serial_fraction_total = 1.0 - parallel_fraction_total;        // грубая оценка доли "непараллелизуемого"

    std::cout << "\nWhole-program fraction estimate (very rough):\n";
    std::cout << "  parallel_fraction_total ~= " << parallel_fraction_total << "\n";
    std::cout << "  serial_fraction_total   ~= " << serial_fraction_total << "\n";

    std::cout << "\nNotes for report:\n";
    std::cout << "  1) Speedup grows with threads but eventually saturates (memory bandwidth + overhead).\n";
    std::cout << "  2) Amdahl: even small serial parts (init/alloc/overheads) limit max speedup.\n";
    std::cout << "  3) Compare measured speedup vs S_amdahl_pred to discuss deviations.\n";

    return 0; // успешное завершение программы
}
