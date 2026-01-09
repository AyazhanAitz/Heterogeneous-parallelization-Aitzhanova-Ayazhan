#include <iostream>      // подключаем ввод и вывод
#include <vector>        // подключаем контейнер vector
#include <random>        // генерация случайных чисел
#include <algorithm>     // swap, inplace_merge
#include <chrono>        // измерение времени
#include <omp.h>         // библиотека OpenMP

using namespace std;     // чтобы не писать std:: перед каждым именем

// Функция проверяет, отсортирован ли массив по возрастанию
bool isSorted(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++) {      // идём по массиву
        if (a[i - 1] > a[i]) return false;       // если порядок нарушен
    }
    return true;                                 // массив отсортирован
}

// Функция создаёт массив случайных чисел
vector<int> makeRandomArray(int n, int lo = 1, int hi = 100) {
    vector<int> a(n);                            // создаём массив размера n
    mt19937 rng(12345);                          // генератор случайных чисел
    uniform_int_distribution<int> dist(lo, hi); // диапазон значений
    for (int i = 0; i < n; i++) {                // заполняем массив
        a[i] = dist(rng);                        // случайным числом
    }
    return a;                                    // возвращаем массив
}

// Последовательная пузырьковая сортировка
void bubbleSortSequential(vector<int>& a) {
    int n = (int)a.size();                       // размер массива
    for (int pass = 0; pass < n - 1; pass++) {   // количество проходов
        bool swapped = false;                    // был ли обмен
        for (int j = 0; j < n - 1 - pass; j++) { // проходим по массиву
            if (a[j] > a[j + 1]) {               // если элементы не по порядку
                swap(a[j], a[j + 1]);            // меняем местами
                swapped = true;                  // фиксируем обмен
            }
        }
        if (!swapped) break;                     // если обменов не было — выходим
    }
}

// Последовательная сортировка выбором
void selectionSortSequential(vector<int>& a) {
    int n = (int)a.size();                       // размер массива
    for (int i = 0; i < n - 1; i++) {             // текущая позиция
        int minIdx = i;                          // индекс минимума
        for (int j = i + 1; j < n; j++) {         // ищем минимум справа
            if (a[j] < a[minIdx]) {               // если нашли меньше
                minIdx = j;                      // обновляем минимум
            }
        }
        if (minIdx != i) {                       // если минимум не на месте
            swap(a[i], a[minIdx]);               // меняем элементы
        }
    }
}

// Последовательная сортировка вставками
void insertionSortSequential(vector<int>& a) {
    int n = (int)a.size();                       // размер массива
    for (int i = 1; i < n; i++) {                 // начинаем со второго элемента
        int key = a[i];                          // текущий элемент
        int j = i - 1;                           // индекс слева
        while (j >= 0 && a[j] > key) {            // сдвигаем элементы вправо
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;                          // вставляем элемент
    }
}

// Параллельная пузырьковая сортировка (odd-even)
void bubbleSortParallelOddEven(vector<int>& a) {
    int n = (int)a.size();                       // размер массива
    for (int phase = 0; phase < n; phase++) {    // количество фаз
        int start = (phase % 2 == 0) ? 0 : 1;     // чётная или нечётная фаза

#pragma omp parallel for                 // распараллеливаем цикл
        for (int j = start; j < n - 1; j += 2) { // сравниваем пары
            if (a[j] > a[j + 1]) {               // если элементы не по порядку
                int tmp = a[j];                  // временная переменная
                a[j] = a[j + 1];                 // меняем местами
                a[j + 1] = tmp;
            }
        }
    }
}

// Параллельная сортировка выбором
void selectionSortParallel(vector<int>& a) {
    int n = (int)a.size();                       // размер массива

    for (int i = 0; i < n - 1; i++) {             // текущая позиция
        int globalMinVal = a[i];                  // глобальный минимум
        int globalMinIdx = i;                     // индекс минимума

#pragma omp parallel                     // параллельная область
        {
            int localMinVal = globalMinVal;       // локальный минимум
            int localMinIdx = globalMinIdx;

#pragma omp for nowait
            for (int j = i + 1; j < n; j++) {     // поиск минимума
                if (a[j] < localMinVal) {
                    localMinVal = a[j];
                    localMinIdx = j;
                }
            }

#pragma omp critical                 // критическая секция
            {
                if (localMinVal < globalMinVal) {
                    globalMinVal = localMinVal;
                    globalMinIdx = localMinIdx;
                }
            }
        }

        if (globalMinIdx != i) {                  // если минимум не на месте
            swap(a[i], a[globalMinIdx]);          // меняем элементы
        }
    }
}

// Вставки на участке массива
void insertionSortRange(vector<int>& a, int l, int r) {
    for (int i = l + 1; i < r; i++) {              // сортируем подмассив
        int key = a[i];
        int j = i - 1;
        while (j >= l && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// Параллельная сортировка вставками (блочная)
void insertionSortParallelBlocked(vector<int>& a) {
    int n = (int)a.size();                        // размер массива
    if (n <= 1) return;                           // если мало элементов — выходим

    int threads = omp_get_max_threads();           // число потоков
    int blocks = max(1, threads);                 // количество блоков
    int blockSize = (n + blocks - 1) / blocks;    // размер блока

#pragma omp parallel for
    for (int b = 0; b < blocks; b++) {             // сортируем каждый блок
        int l = b * blockSize;
        int r = min(n, l + blockSize);
        if (l < r) insertionSortRange(a, l, r);
    }

    for (int width = blockSize; width < n; width *= 2) {
#pragma omp parallel for
        for (int left = 0; left < n; left += 2 * width) {
            int mid = min(n, left + width);
            int right = min(n, left + 2 * width);
            if (mid < right) {
                inplace_merge(a.begin() + left,
                    a.begin() + mid,
                    a.begin() + right);
            }
        }
    }
}

// Функция измеряет время выполнения сортировки
template <typename Func>
long long measureMs(Func f, vector<int>& a) {
    auto t1 = chrono::high_resolution_clock::now(); // старт времени
    f(a);                                           // вызываем сортировку
    auto t2 = chrono::high_resolution_clock::now(); // конец времени
    return chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
}

int main() {
    setlocale(LC_ALL, "Russian");   // установка русской локали
    vector<int> sizes = { 1000, 10000, 100000 };     // размеры массивов

    cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";

    for (int n : sizes) {
        cout << "Размер массива n = " << n << "\n";

        vector<int> base = makeRandomArray(n);     // исходный массив

        vector<int> a1 = base, a2 = base, a3 = base;
        vector<int> p1 = base, p2 = base, p3 = base;

        long long t1 = measureMs(bubbleSortSequential, a1);
        long long t2 = measureMs(selectionSortSequential, a2);
        long long t3 = measureMs(insertionSortSequential, a3);

        long long t4 = measureMs(bubbleSortParallelOddEven, p1);
        long long t5 = measureMs(selectionSortParallel, p2);
        long long t6 = measureMs(insertionSortParallelBlocked, p3);

        cout << "Bubble   seq=" << t1 << " ms | par=" << t4 << " ms\n";
        cout << "Select   seq=" << t2 << " ms | par=" << t5 << " ms\n";
        cout << "Insert   seq=" << t3 << " ms | par=" << t6 << " ms\n\n";
    }

    return 0;                                    
}
