//用线性回归去拟合我的灰度化参数 有种脱裤子放屁的感觉

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./include/stb_image.h"
#include "./include/stb_image_write.h"

//0-不使用GPU加速 1-使用简单CUDA核函数加速 2-使用cuBLAS加速
#define GPU_BOOST_TYPE 0

//用二次回归，看看过拟合的情况
#define SECOND_ORDER 0

typedef struct {
	int m;
	int n;
	float* data;
} Mat;

static Mat* buildMat(int m, int n) {
	Mat* mat = (Mat*)malloc(sizeof(Mat));
	mat->m = m;
	mat->n = n;
	mat->data = (float*)malloc(m * n * sizeof(float));
	return mat;
}

static Mat* matMul(Mat* A, Mat* B) {
	if (A->n != B->m) {
		return NULL;
	}
	Mat* C = buildMat(A->m, B->n);
	for (int i = 0; i < A->m; i++) {
		for (int j = 0; j < B->n; j++) {
			float sum = 0.0f;
			for (int k = 0; k < A->n; k++) {
				sum += A->data[i * A->n + k] * B->data[k * B->n + j];
			}
			C->data[i * C->n + j] = sum;
		}
	}
	return C;
}

static Mat* matTranspose(Mat* mat) {
	Mat* trans = buildMat(mat->n, mat->m);
	for (int i = 0; i < mat->m; i++) {
		for (int j = 0; j < mat->n; j++) {
			trans->data[j * trans->m + i] = mat->data[i * mat->n + j];
		}
	}
	return trans;
}

static Mat* matInvert(Mat* mat) {
	if (mat->m != mat->n) {
		return NULL;
	}

	int n = mat->m;
	Mat* inv = buildMat(n, n);
	float* aug = (float*)malloc(n * 2 * n * sizeof(float));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			aug[i * 2 * n + j] = mat->data[i * n + j];
			aug[i * 2 * n + j + n] = (i == j) ? 1.0f : 0.0f;
		}
	}
	for (int i = 0; i < n; i++) {
		int maxRow = i;
		float maxVal = fabsf(aug[i * 2 * n + i]);
		for (int k = i + 1; k < n; k++) {
			if (fabsf(aug[k * 2 * n + i]) > maxVal) {
				maxVal = fabsf(aug[k * 2 * n + i]);
				maxRow = k;
			}
		}
		if (maxRow != i) {
			for (int j = 0; j < 2 * n; j++) {
				float temp = aug[i * 2 * n + j];
				aug[i * 2 * n + j] = aug[maxRow * 2 * n + j];
				aug[maxRow * 2 * n + j] = temp;
			}
		}
		if (fabsf(aug[i * 2 * n + i]) < 1e-8f) {
			free(aug);
			free(inv->data);
			free(inv);
			return NULL;
		}
		float pivot = aug[i * 2 * n + i];
		for (int j = 0; j < 2 * n; j++) {
			aug[i * 2 * n + j] /= pivot;
		}
		for (int k = 0; k < n; k++) {
			if (k != i) {
				float factor = aug[k * 2 * n + i];
				for (int j = 0; j < 2 * n; j++) {
					aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
				}
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			inv->data[i * n + j] = aug[i * 2 * n + j + n];
		}
	}

	free(aug);
	return inv;
}


static inline const char* fileTrainX(int x) {
	static char buffer[256];
	sprintf(buffer, "E:\\workbench\\cuda\\LearnCuda\\regression\\data\\train140000\\%d.jpeg", 140000 + x);
	return buffer;
}

static inline const char* fileTestX(int x) {
	static char buffer[256];
	sprintf(buffer, "E:\\workbench\\cuda\\LearnCuda\\regression\\data\\test_%d.jpg", x);
	return buffer;
}

static inline const char* fileResultX(int x) {
	static char buffer[256];
	sprintf(buffer, "E:\\workbench\\cuda\\LearnCuda\\regression\\data\\result_%d.png", x);
	return buffer;
}


static Mat* matX;
static Mat* matY;
static const int TRAIN_CAP = 2;
#if SECOND_ORDER
static const int featureDim = 6;
#else
static const int featureDim = 3;
#endif

static void genFeatureMat() {
	unsigned char* sampleX[TRAIN_CAP];
	int size[TRAIN_CAP];
	int N = 0;
	for (int x = 0; x < TRAIN_CAP; x++) {
		int width, height, channel;
		sampleX[x] = stbi_load(fileTrainX(x), &width, &height, &channel, 3);
		if (sampleX[x] == NULL) {
			printf("sample %d is null\n", x);
		}
		N += size[x] = width * height;
	}
	matX = buildMat(N, featureDim);
	matY = buildMat(N, 1);
	int index = 0;
	float r, g, b;
	for (int x = 0; x < TRAIN_CAP; x++) {
		for (int i = 0; i < size[x]; i++) {
			//printf("%d\n", i);
			matX->data[index * featureDim + 0] = r = (float)sampleX[x][i * 3 + 0];
			matX->data[index * featureDim + 1] = g = (float)sampleX[x][i * 3 + 1];
			matX->data[index * featureDim + 2] = b = (float)sampleX[x][i * 3 + 2];
			if (featureDim==6) {
				matX->data[index * featureDim + 3] = r * r;
				matX->data[index * featureDim + 4] = g * g;
				matX->data[index * featureDim + 5] = b * b;
			}
			//加权平均值法灰度化
			matY->data[index * 1 + 0] = 0.299f * r + 0.587f * g + 0.114f * b;
			index++;
		}
		stbi_image_free((void*)sampleX[x]);
	}
}

static Mat* w_linear = buildMat(featureDim, 1);
#if GPU_BOOST_TYPE==1

//主机端矩阵
typedef struct {
	int m;
	int n;
	float* data;
	void* devPtr;
}MatHost;
static MatHost* uploadMat(Mat* mat) {
	MatHost* matH = (MatHost*)malloc(sizeof(MatHost));
	matH->m = mat->m;
	matH->n = mat->n;
	matH->data = mat->data;
	int size_in_bytes = mat->m * mat->n * sizeof(float);
	cudaMalloc(&matH->devPtr, size_in_bytes);
	cudaMemcpy(matH->devPtr, mat->data, size_in_bytes, cudaMemcpyHostToDevice);
	return matH;
}

static void downloadMat(MatHost* matH, Mat* mat) {
	int size_in_bytes = matH->m * matH->n * sizeof(float);
	cudaMemcpy(mat->data, matH->devPtr, size_in_bytes, cudaMemcpyDeviceToHost);
}

__global__ void matTransposeKernel(float* source, float* result, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < m && col < n) {
		result[col * m + row] = source[row * n + col];
	}
}

__global__ void matMulKernel(float* left, float* right, float* result, int m1, int n1, int m2, int n2) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int m = m1;
	int n = n2;
	int k = n1;
	if (row < m && col < n) {
		float sum = 0;
		for (int i = 0; i < k; i++) {
			sum += left[row * n1 + i] * right[i * n2 + col];
		}
		result[row * n + col] = sum;
	}
}

static MatHost* matTransposeCUDA(MatHost* sourceHost) {
	MatHost* resultHost = uploadMat(buildMat(sourceHost->n, sourceHost->m));
	dim3 grid((sourceHost->n + 15) / 16, (sourceHost->m + 15) / 16);
	dim3 block(16, 16);
	matTransposeKernel << <grid, block >> > ((float*)sourceHost->devPtr, (float*)resultHost->devPtr, sourceHost->m, sourceHost->n);
	return resultHost;
}

static MatHost* matMulCUDA(MatHost* leftHost, MatHost* rightHost) {
	if (leftHost->n != rightHost->m) {
		return NULL;
	}
	MatHost* resultHost = uploadMat(buildMat(leftHost->m, rightHost->n));
	int m = leftHost->m;
	int n = rightHost->n;
	dim3 grid((n + 15) / 16, (m + 15) / 16);
	dim3 block(16, 16);
	//显然分治求和是更好的选择，但是时间不太够，先验证正确性。
	matMulKernel << <grid, block >> > ((float*)leftHost->devPtr, (float*)rightHost->devPtr, (float*)resultHost->devPtr,
		leftHost->m, leftHost->n, rightHost->m, rightHost->n);
	return resultHost;
}



static void trainLinearRegression() {
	cudaSetDevice(0);
	Mat* A = buildMat(1, 3);
	Mat* B = buildMat(3, 1);
	A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
	B->data[0] = 4.0f; B->data[1] = 5.0f; B->data[2] = 6.0f;
	MatHost* testA = matTransposeCUDA(uploadMat(A));
	MatHost* testB = matTransposeCUDA(uploadMat(B));
	MatHost* testC = matMulCUDA(testA, testB);
	Mat* C = buildMat(testC->m, testC->n);
	downloadMat(testC, C);
	printf("testC:\n");
	for (int x = 0; x < C->n; x++) {
		for (int y = 0; y < C->m; y++) {
			printf("%f ", C->data[y * C->n + x]);
		}
		printf("\n");
	}

	MatHost* devMatX = uploadMat(matX);
	MatHost* devMatY = uploadMat(matY);
	MatHost* X_T = matTransposeCUDA(devMatX);
	Mat* reader = buildMat(X_T->m, X_T->n);
	downloadMat(X_T, reader);
	printf("reader:\n");
	for (int y = 0; y < reader->m && y < 100; y++) {
		for (int x = 0; x < reader->n && x < 100; x++) {
			printf("%f ", reader->data[y * reader->n + x]);
		}
		printf("\n");
	}
	MatHost* X_T_X = matMulCUDA(X_T, devMatX);
	MatHost* X_T_Y = matMulCUDA(X_T, devMatY);
	Mat* temp = buildMat(X_T_X->m, X_T_X->n);
	downloadMat(X_T_X, temp);
	printf("temp:%d\n", temp == NULL);
	for (int x = 0; x < temp->n; x++) {
		for (int y = 0; y < temp->m; y++) {
			printf("%f ", temp->data[y * temp->n + x]);
		}
		printf("\n");
	}
	Mat* temp_inv = matInvert(temp);
	printf("temp_inv:%d\n", temp_inv == NULL);
	MatHost* X_T_X_inv = uploadMat(temp_inv);
	MatHost* temp1 = matMulCUDA(X_T_X_inv, X_T_Y);
	downloadMat(temp1, w_linear);
	cudaDeviceReset();
	printf("w:[%f,%f,%f]\n", w_linear->data[0], w_linear->data[1], w_linear->data[2]);
}
#elif GPU_BOOST_TYPE==2

#else
static void trainLinearRegression() {
	Mat* X_T = matTranspose(matX);
	Mat* X_T_X = matMul(X_T, matX);
	for (int x = 0; x < X_T_X->n; x++) {
		for (int y = 0; y < X_T_X->m; y++) {
			printf("%f ", X_T_X->data[y * X_T_X->n + x]);
		}
		printf("\n");
	}
	Mat* X_T_X_inv = matInvert(X_T_X);
	if (X_T_X_inv == NULL) {
		printf("Matrix inversion failed!\n");
		return;
	}
	Mat* X_T_Y = matMul(X_T, matY);
	w_linear = matMul(X_T_X_inv, X_T_Y);
	printf("w:[%f,%f,%f]\n", w_linear->data[0], w_linear->data[1], w_linear->data[2]);
}
#endif


static const int TEST_CAP = 1;
static void testLinearRegression() {
	unsigned char* data;
	int width, height, channel;
	unsigned char r, g, b;
	float gray;
	for (int x = 0; x < TEST_CAP; x++) {
		data = stbi_load(fileTestX(x), &width, &height, &channel, 3);
		for (int i = 0; i < width * height; i++) {
			r = data[i * 3 + 0];
			g = data[i * 3 + 1];
			b = data[i * 3 + 2];
			//用模型预测灰度值
			gray = w_linear->data[0] * r + w_linear->data[1] * g + w_linear->data[2] * b;
			if (featureDim==6) {
				gray += w_linear->data[3] * r * r + w_linear->data[4] * g * g + w_linear->data[5] * b * b;
			}
			data[i * 3 + 0] = (unsigned char)(gray);
			data[i * 3 + 1] = (unsigned char)(gray);
			data[i * 3 + 2] = (unsigned char)(gray);
		}
		const int stride_in_bytes = width * 3;
		stbi_write_png(fileResultX(x), width, height, 3, (void*)data, stride_in_bytes);
		stbi_image_free(data);
	}
}

int main()
{
	genFeatureMat();
	trainLinearRegression();
	testLinearRegression();
	return 0;
}