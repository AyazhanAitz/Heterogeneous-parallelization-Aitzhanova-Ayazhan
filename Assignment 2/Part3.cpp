#include <iostream>   // Подключаем библиотеку для ввода и вывода (cout)
#include <cstdlib>    // Подключаем библиотеку для rand() и srand()
#include <ctime>      // Подключаем библиотеку для time()
#include <vector>     // Подключаем контейнер vector
#include <algorithm>  // Для функции is_sorted (проверка сортировки)
#include <omp.h>      // Подключаем OpenMP для параллельных вычислений

using namespace std;

// Функция заполнения массива случайными числами
// vector<int>& a — ссылка на массив
// lo и hi — нижняя и верхняя границы случайных чисел
static void fillRandom(vector<int>& a, int lo = 1, int hi = 100000) {
    // Проходим по каждому элементу массива
    for (int& x : a) {
        // Генерируем случайное число в диапазоне [lo; hi]
        x = lo + rand() % (hi - lo + 1);
    }
}

// Последовательная сортировка выбором
// Алгоритм имеет сложность O(n^2)
static void selectionSortSequential(vector<int>& a) {
    // Получаем размер массива
    int n = (int)a.size();

    // Внешний цикл — указывает текущую позицию,
    // куда будет поставлен минимальный элемент
    for (int i = 0; i < n - 1; i++) {

        // Предполагаем, что минимальный элемент — текущий
        int minIdx = i;

        // Внутренний цикл — ищем минимальный элемент
        // среди оставшейся части массива
        for (int j = i + 1; j < n; j++) {

            // Если найден элемент меньше текущего минимума
            if (a[j] < a[minIdx]) {
                // Обновляем индекс минимального элемента
                minIdx = j;
            }
        }

        // Если минимальный элемент найден не на позиции i
        if (minIdx != i) {
            // Меняем местами текущий элемент и минимальный
            int tmp = a[i];
            a[i] = a[minIdx];
            a[minIdx] = tmp;
        }
    }
}

// Параллельная версия сортировки выбором с использованием OpenMP
static void selectionSortParallel(vector<int>& a) {
    // Получаем размер массива
    int n = (int)a.size();

    // Внешний цикл остаётся последовательным,
    // так как каждая итерация зависит от предыдущей
    for (int i = 0; i < n - 1; i++) {

        // Глобальный минимум для текущей итерации
        int globalMinVal = a[i];

        // Индекс глобального минимума
        int globalMinIdx = i;

        // Начало параллельной области
#pragma omp parallel
        {
            // Локальный минимум для каждого потока
            int localMinVal = globalMinVal;

            // Локальный индекс минимума для потока
            int localMinIdx = globalMinIdx;

            // Распараллеливаем цикл поиска минимума
            // Каждый поток получает свою часть диапазона
#pragma omp for nowait
            for (int j = i + 1; j < n; j++) {

                // Если найден элемент меньше локального минимума
                if (a[j] < localMinVal) {
                    // Обновляем локальный минимум
                    localMinVal = a[j];
                    localMinIdx = j;
                }
            }

            // Критическая секция —
            // только один поток за раз обновляет глобальный минимум
#pragma omp critical
            {
                // Если локальный минимум меньше глобального
                if (localMinVal < globalMinVal) {
                    // Обновляем глобальный минимум
                    globalMinVal = localMinVal;
                    globalMinIdx = localMinIdx;
                }
            }
        }

        // После завершения всех потоков
        // меняем элементы местами, если нужно
        if (globalMinIdx != i) {
            int tmp = a[i];
            a[i] = a[globalMinIdx];
            a[globalMinIdx] = tmp;
        }
    }
}

// Функция тестирования производительности для одного размера массива
static void benchmarkOneSize(int n) {
    // Создаём исходный массив
    vector<int> base(n);

    // Заполняем его случайными числами
    fillRandom(base);

    // Копия для последовательной версии
    vector<int> seqArr = base;

    // Копия для параллельной версии
    vector<int> parArr = base;

    // Засекаем время начала последовательной сортировки
    double seqStart = omp_get_wtime();

    // Запускаем последовательную сортировку
    selectionSortSequential(seqArr);

    // Засекаем время окончания
    double seqEnd = omp_get_wtime();

    // Засекаем время начала параллельной сортировки
    double parStart = omp_get_wtime();

    // Запускаем параллельную сортировку
    selectionSortParallel(parArr);

    // Засекаем время окончания
    double parEnd = omp_get_wtime();

    // Проверяем, отсортирован ли массив
    bool seqSorted = is_sorted(seqArr.begin(), seqArr.end());
    bool parSorted = is_sorted(parArr.begin(), parArr.end());

    // Проверяем, одинаковы ли результаты
    bool same = (seqArr == parArr);

    // Вычисляем время работы в миллисекундах
    double seqTime = (seqEnd - seqStart) * 1000.0;
    double parTime = (parEnd - parStart) * 1000.0;

    // Выводим результаты
    cout << "Размер массива: " << n << "\n";
    cout << "Последовательная версия: " << seqTime << " мс | sorted=" << seqSorted << "\n";
    cout << "Параллельная версия:     " << parTime << " мс | sorted=" << parSorted << "\n";
    cout << "Результаты совпадают:    " << same << "\n";

    // Если параллельное время не нулевое — считаем ускорение
    if (parTime > 0.0) {
        cout << "Ускорение (seq/par):     " << (seqTime / parTime) << "x\n";
    }

    cout << "------------------------------------------\n";
}

int main() {
    // Устанавливаем русскую локаль
    setlocale(LC_ALL, "Russian");

    // Инициализируем генератор случайных чисел
    srand((unsigned)time(nullptr));

    // Выводим количество доступных потоков OpenMP
    cout << "Количество потоков OpenMP: " << omp_get_max_threads() << "\n";
    cout << "------------------------------------------\n";

    // Тестируем массив из 1000 элементов
    benchmarkOneSize(1000);

    // Тестируем массив из 10000 элементов
    benchmarkOneSize(10000);

    // Итоговый вывод
    cout << "Вывод:\n";
    cout << "- Сортировка выбором имеет квадратичную сложность O(n^2).\n";
    cout << "- В параллельной версии распараллелен только поиск минимума.\n";
    cout << "- Для небольших массивов накладные расходы OpenMP могут превышать выигрыш.\n";

    return 0;
}
