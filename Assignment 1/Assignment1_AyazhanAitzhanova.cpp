#include <iostream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <chrono>
#include <omp.h>

using namespace std;

// Функция fillRandom заполняет массив случайными числами от 1 до 100
// arr - указатель на массив
// n - размер массива
void fillRandom(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1 + rand() % 100;
    }
}

// Функция averageSequential считает среднее значение массива последовательно
// const int* arr - массив, который не изменяем
// n - размер массива
double averageSequential(const int* arr, int n) {
    // Используем long long, потому что сумма может быть большой (чтобы не было переполнения int)
    long long sum = 0;

    // Цикл по всем элементам массива
    for (int i = 0; i < n; i++) {
        // Добавляем текущий элемент к сумме
        sum += arr[i];
    }

    // Возвращаем среднее и приводим к double, чтобы было дробное деление
    return (double)sum / n;
}

// Функция averageParallelOMP считает среднее значение массива параллельно (OpenMP)
// Используем reduction(+:sum) для корректного сложения из разных потоков
double averageParallelOMP(const int* arr, int n) {
    // Общая сумма (будет собрана из частичных сумм потоков)
    long long sum = 0;

    // pragma omp parallel for - распараллелить цикл for
    // reduction(+:sum) - каждый поток считает свою sum,
    // потом OpenMP складывает их в общую sum
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        // Каждый поток добавляет элементы в свою локальную сумму
        sum += arr[i];
    }

    // возвращаем среднее значение 
    return (double)sum / n;
}

// Функция minMaxSequential находит минимум и максимум в массиве последовательно
// mn и mx передаются по ссылке, чтобы функция могла их изменить
void minMaxSequential(const int* arr, int n, int& mn, int& mx) {
    // Начальный минимум ставим максимально возможным int
    mn = INT_MAX;

    // Начальный максимум ставим минимально возможным int
    mx = INT_MIN;

    // Цикл по всем элементам массива
    for (int i = 0; i < n; i++) {
        // Если текущий элемент меньше mn - обновляем минимум
        if (arr[i] < mn) mn = arr[i];

        // Если текущий элемент больше mx - обновляем максимум
        if (arr[i] > mx) mx = arr[i];
    }
}

// Функция minMaxParallelOMP находит минимум и максимум параллельно (OpenMP)
// каждый поток считает localMin/localMax, затем объединяем результаты в критической секции.

void minMaxParallelOMP(const int* arr, int n, int& mn, int& mx) {
    // Глобальный минимум/максимум (результат)
    mn = INT_MAX;
    mx = INT_MIN;

    // pragma omp parallel - создаём параллельную область
#pragma omp parallel
    {
        // Локальный минимум для конкретного потока
        int localMin = INT_MAX;

        // Локальный максимум для конкретного потока
        int localMax = INT_MIN;

        // pragma omp for - делим цикл for между потоками
        // nowait - не заставляем потоки ждать друг друга после цикла
#pragma omp for nowait
        for (int i = 0; i < n; i++) {
            // Обновляем локальный минимум
            if (arr[i] < localMin) localMin = arr[i];

            // Обновляем локальный максимум
            if (arr[i] > localMax) localMax = arr[i];
        }

        // critical - критическая секция
        // сюда заходит только один поток одновременно
        // чтобы безопасно обновить общие mn и mx
#pragma omp critical
        {
            // Если локальный минимум меньше общего - обновляем
            if (localMin < mn) mn = localMin;

            // Если локальный максимум больше общего - обновляем
            if (localMax > mx) mx = localMax;
        }
    }
}

// Функция usNow возвращает текущее время в микросекундах, чтобы посчитать время выполнения последовательных и параллельных функций
long long usNow() {
    // Берём текущее время high_resolution_clock и переводим в microseconds
    // count() возвращает число микросекунд
    return chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

int main() {
    // Устанавливаем русскую локаль
    setlocale(LC_ALL, "Russian");

    // Инициализируем генератор случайных чисел текущим временем
    srand((unsigned)time(nullptr));

    // ЗАДАНИЕ 1

    cout << "Задание 1: массив 50 000, среднее\n";

    // Размер массива
    int n1 = 50000;

    // Динамически выделяем память под массив из 50 000 int
    int* a1 = new int[n1];

    // Заполняем массив случайными числами
    fillRandom(a1, n1);

    // Считаем среднее значение последовательно
    double avg1 = averageSequential(a1, n1);

    // Выводим результат
    cout << "Среднее = " << avg1 << "\n";

    // Освобождаем память 
    delete[] a1;

    cout << "Память освобождена.\n\n";

    // ЗАДАНИЕ 2

    cout << "Задание 2: массив 1 000 000, min/max последовательно + время\n";

    // Размер массива
    int n2 = 1000000;

    // Выделяем память под массив
    int* a2 = new int[n2];

    // Заполняем случайными числами
    fillRandom(a2, n2);

    // Переменные под минимум и максимум
    int mnSeq, mxSeq;

    // Засекаем время начала (микросекунды)
    long long t1 = usNow();

    // Ищем min/max последовательно
    minMaxSequential(a2, n2, mnSeq, mxSeq);

    // Засекаем время конца
    long long t2 = usNow();

    // Выводим min/max
    cout << "min = " << mnSeq << ", max = " << mxSeq << "\n";

    // Выводим время выполнения в микросекундах
    cout << "Время (последовательно) = " << (t2 - t1) << " us\n\n";


    // ЗАДАНИЕ 3

    cout << "Задание 3: тот же массив 1 000 000, min/max OpenMP + сравнение времени\n";

    // Переменные под минимум и максимум (параллельный результат)
    int mnPar, mxPar;

    // Время начала
    long long t3 = usNow();

    // Ищем min/max параллельно (OpenMP)
    minMaxParallelOMP(a2, n2, mnPar, mxPar);

    // Время конца
    long long t4 = usNow();

    // Выводим результат min/max
    cout << "min = " << mnPar << ", max = " << mxPar << "\n";

    // Выводим время OpenMP версии
    cout << "Время (OpenMP) = " << (t4 - t3) << " us\n";

    // Вычисляем ускорение (если OpenMP время не 0)
    if ((t4 - t3) > 0) {
        cout << "Ускорение ~ " << (double)(t2 - t1) / (double)(t4 - t3) << "x\n\n";
    }

    // Освобождаем память массива задания 2/3
    delete[] a2;

    // ЗАДАНИЕ 4

    cout << "Задание 4: массив 5 000 000, среднее seq vs OpenMP(reduction)\n";

    // Размер массива
    int n4 = 5000000;

    // Выделяем память под массив
    int* a4 = new int[n4];

    // Заполняем массив
    fillRandom(a4, n4);

    // Запрещаем OpenMP автоматически менять число потоков
    omp_set_dynamic(0);

    // Ставим максимальное число потоков, которое доступно системе
    omp_set_num_threads(omp_get_max_threads());

    // Прогрев: один раз считаем среднее, чтобы уменьшить влияние "холодного" кеша
    volatile double warm = averageSequential(a4, n4);

    // Время начала последовательного вычисления
    long long t5 = usNow();

    // Среднее последовательно
    double avgSeq = averageSequential(a4, n4);

    // Время конца последовательного вычисления
    long long t6 = usNow();

    // Время начала OpenMP вычисления
    long long t7 = usNow();

    // Среднее параллельно (OpenMP reduction)
    double avgPar = averageParallelOMP(a4, n4);

    // Время конца OpenMP вычисления
    long long t8 = usNow();

    // Выводим средние значения
    cout << "Среднее (послед.) = " << avgSeq << "\n";
    cout << "Среднее (OpenMP)  = " << avgPar << "\n";

    // Выводим времена выполнения в микросекундах
    cout << "Время (послед.)   = " << (t6 - t5) << " us\n";
    cout << "Время (OpenMP)    = " << (t8 - t7) << " us\n";

    // Ускорение (если время OpenMP не 0)
    if ((t8 - t7) > 0) {
        cout << "Ускорение ~ " << (double)(t6 - t5) / (double)(t8 - t7) << "x\n\n";
    }

    // Освобождаем память массива задания 4
    delete[] a4;

    return 0;
}
