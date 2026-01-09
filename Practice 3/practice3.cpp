%%writefile gpu_sorts.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <climits>

using namespace std;

// Проверка ошибок CUDA
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "ОШИБКА CUDA: " << msg << " -> "
             << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

// Проверка, отсортирован ли массив
bool isSorted(const vector<int>& a) {
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i - 1] > a[i]) return false;
    }
    return true;
}

// Генерация случайного массива
vector<int> generateRandom(size_t n) {
    vector<int> a(n);
    mt19937 gen(42);
    uniform_int_distribution<int> dist(0, 1'000'000);

    for (size_t i = 0; i < n; i++) {
        a[i] = dist(gen);
    }
    return a;
}


// ---------- MergeSort (CPU) ----------
void mergeCPU(vector<int>& a, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = a[l + i];
    for (int j = 0; j < n2; j++) R[j] = a[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) a[k++] = L[i++];
        else a[k++] = R[j++];
    }

    while (i < n1) a[k++] = L[i++];
    while (j < n2) a[k++] = R[j++];
}

void mergeSortCPURec(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int m = l + (r - l) / 2;
    mergeSortCPURec(a, l, m);
    mergeSortCPURec(a, m + 1, r);
    mergeCPU(a, l, m, r);
}

void mergeSortCPU(vector<int>& a) {
    if (!a.empty())
        mergeSortCPURec(a, 0, (int)a.size() - 1);
}

// ---------- QuickSort (CPU) ----------
int partitionCPU(vector<int>& a, int low, int high) {
    int pivot = a[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (a[j] <= pivot) {
            i++;
            swap(a[i], a[j]);
        }
    }
    swap(a[i + 1], a[high]);
    return i + 1;
}

void quickSortCPURec(vector<int>& a, int low, int high) {
    if (low >= high) return;
    int p = partitionCPU(a, low, high);
    quickSortCPURec(a, low, p - 1);
    quickSortCPURec(a, p + 1, high);
}

void quickSortCPU(vector<int>& a) {
    if (!a.empty())
        quickSortCPURec(a, 0, (int)a.size() - 1);
}

// ---------- HeapSort (CPU) ----------
void siftDownCPU(vector<int>& a, int n, int i) {
    while (true) {
        int largest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;

        if (l < n && a[l] > a[largest]) largest = l;
        if (r < n && a[r] > a[largest]) largest = r;

        if (largest == i) break;
        swap(a[i], a[largest]);
        i = largest;
    }
}

void heapSortCPU(vector<int>& a) {
    int n = a.size();

    for (int i = n / 2 - 1; i >= 0; i--)
        siftDownCPU(a, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(a[0], a[i]);
        siftDownCPU(a, i, 0);
    }
}

//CUDA

// Cлияние двух отсортированных сегментов
__global__ void mergeKernel(const int* in, int* out, int n, int seg) {
    int pairId = blockIdx.x;
    int start = pairId * 2 * seg;
    int mid = min(start + seg, n);
    int end = min(start + 2 * seg, n);

    int i = start, j = mid, k = start;

    while (k < end) {
        int left = (i < mid) ? in[i] : INT_MAX;
        int right = (j < end) ? in[j] : INT_MAX;

        if (left <= right) {
            out[k++] = left;
            i++;
        } else {
            out[k++] = right;
            j++;
        }
    }
}

// ---------- Bitonic Sort для чанка (MergeSort GPU) ----------
template<int CHUNK>
__global__ void bitonicChunkSort(int* data, int n) {
    __shared__ int s[CHUNK];

    int base = blockIdx.x * CHUNK;
    int tid = threadIdx.x;
    int idx = base + tid;

    if (tid < CHUNK) {
        if (idx < n) s[tid] = data[idx];
        else s[tid] = INT_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                bool asc = ((tid & k) == 0);
                if ((asc && s[tid] > s[ixj]) ||
                    (!asc && s[tid] < s[ixj])) {
                    int tmp = s[tid];
                    s[tid] = s[ixj];
                    s[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if (idx < n)
        data[idx] = s[tid];
}

// GPU MergeSort (чанки + поэтапное слияние)
void mergeSortGPU(int* d, int n) {
    const int CHUNK = 1024;
    int blocks = (n + CHUNK - 1) / CHUNK;

    bitonicChunkSort<CHUNK><<<blocks, CHUNK>>>(d, n);
    checkCuda(cudaDeviceSynchronize(), "bitonic sort");

    int* tmp;
    checkCuda(cudaMalloc(&tmp, n * sizeof(int)), "malloc tmp");

    int* in = d;
    int* out = tmp;

    for (int seg = CHUNK; seg < n; seg *= 2) {
        int pairs = (n + 2 * seg - 1) / (2 * seg);
        mergeKernel<<<pairs, 1>>>(in, out, n, seg);
        checkCuda(cudaDeviceSynchronize(), "merge pass");
        swap(in, out);
    }

    if (in != d)
        cudaMemcpy(d, in, n * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(tmp);
}

// Измерение времени
 
double measureCPU(void (*sortFunc)(vector<int>&), vector<int> data) {
    auto start = chrono::high_resolution_clock::now();
    sortFunc(data);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

double measureGPU(void (*gpuFunc)(int*, int), vector<int>& data) {
    int n = data.size();
    int* d;
    cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, data.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuFunc(d, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    vector<size_t> sizes = {10000, 100000, 1000000};

    cout << "Сравнение производительности CPU и GPU сортировок\n\n";

    for (size_t n : sizes) {
        cout << "Размер массива: " << n << endl;

        vector<int> data = generateRandom(n);

        cout << "CPU MergeSort: " << measureCPU(mergeSortCPU, data) << " мс\n";
        cout << "CPU QuickSort: " << measureCPU(quickSortCPU, data) << " мс\n";
        cout << "CPU HeapSort : " << measureCPU(heapSortCPU,  data) << " мс\n";

        cout << "GPU MergeSort (CUDA): " << measureGPU(mergeSortGPU, data) << " мс\n";
        cout << endl;
    }

    cout << "Завершено." << endl;
    return 0;
}
