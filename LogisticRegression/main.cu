//用逻辑回归解决三分类任务 鸢尾花和红酒数据集

#include "cublas_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#define DATASET 0
#if DATASET == 0
static const char* DATASET_PATH = "E:/study/logistic regression/iris_dataset.txt";
static const int FEATURE_COUNT = 1 + 4;
static const int SAMPLE_COUNT = 150;
static const int LABEL_COUNT = 1;
static const int TRAIN_CAP = 100;
static const double LEARNING_RATE = 1;
static const int ITERATION_TIMES = 2000;
#elif DATASET == 1
static const char* DATASET_PATH = "E:/study/logistic regression/wine_dataset.txt";
static const int FEATURE_COUNT = 1 + 13;
static const int SAMPLE_COUNT = 178;
static const int LABEL_COUNT = 1;
static const int TRAIN_CAP = 120;
static const double LEARNING_RATE = 0.00001;
static const int ITERATION_TIMES = 500000;
#endif
// 包含训练记录器（需要在 FEATURE_COUNT 定义之后）
#include "training_logger.h"

typedef struct
{
    int m;
    int n;
    // device pointer
    double* data;
} Mat;

static void sizeMat(Mat* mat, int m, int n)
{
    mat->m = m;
    mat->n = n;
    cudaMalloc((void**)&(mat->data), sizeof(double) * m * n);
}

static void freeMat(Mat* mat)
{
    cudaFree(mat->data);
    mat->m = 0;
    mat->n = 0;
    mat->data = NULL;
}

typedef struct
{
    // Row-major 一行是一个样本 每列是一个特征 其中第0列恒为1 用于bias
    // 由于cuBLAS是Column-major 所以会隐式带一个转置成一列是一个样本
    // 这与后续计算是契合的
    Mat features;
    // Row-major 一行是一个样本 由于只有一列 无论在Host还是cuBLAS都可以视为列向量
    Mat labels;
} DataSet;

typedef struct
{
    double parameters[FEATURE_COUNT] = {0};
} Classifier;

typedef struct
{
    Classifier classifier[3];
} IRISClassifier;

static DataSet trainSet;
static DataSet testSet;
static IRISClassifier irisClassifier;

cublasHandle_t cublasH = NULL;

static void readData(const char* filePath, int sampleCount, int featureCount, int labelCount, int trainCap)
{
    FILE* file = fopen(filePath, "r");
    char* header = (char*)malloc(sizeof(char) * 256);
    fgets(header, 256, file);
    sizeMat(&trainSet.features, trainCap, featureCount);
    sizeMat(&testSet.features, sampleCount - trainCap, featureCount);
    sizeMat(&trainSet.labels, trainCap, labelCount);
    sizeMat(&testSet.labels, sampleCount - trainCap, labelCount);
    double* hostFeatureBuffer = (double*)malloc(sizeof(double) * sampleCount * featureCount);
    double* hostLabelBuffer = (double*)malloc(sizeof(double) * sampleCount);
    int indexFeature = 0;
    int indexLabel = 0;
    for (int n = 0; n < sampleCount; n++)
    {
        for (int i = 0; i < featureCount; i++)
        {
            fscanf(file, "%lf", hostFeatureBuffer + indexFeature);
            indexFeature++;
        }
        for (int i = 0; i < labelCount; i++)
        {
            fscanf(file, "%lf", hostLabelBuffer + indexLabel);
            indexLabel++;
        }
    }
    cudaMemcpy(trainSet.features.data, hostFeatureBuffer, trainCap * featureCount * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(testSet.features.data, hostFeatureBuffer + trainCap * featureCount,
               (sampleCount - trainCap) * featureCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(trainSet.labels.data, hostLabelBuffer, trainCap * labelCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(testSet.labels.data, hostLabelBuffer + trainCap * labelCount,
               (sampleCount - trainCap) * labelCount * sizeof(double), cudaMemcpyHostToDevice);
    free(header);
    free(hostFeatureBuffer);
    free(hostLabelBuffer);
    // see freeMat
}

/*
因为样本量就一百多个 只用threadIdx.x来索引就行了
*/
__global__ static void genVectorLabelsKernel(double* matLabels, double* dest, double labelIndex)
{
    int i = threadIdx.x;
    dest[i] = fabs(matLabels[i] - labelIndex) < 0.01;
}

__global__ static void sigmoidKernelAndSubY(double* matX, double* vecY, double* parameters, double* dest)
{
    int i = threadIdx.x;
    double z = 0;
    for (int j = 0; j < FEATURE_COUNT; j++)
    {
        z += matX[i * FEATURE_COUNT + j] * parameters[j];
    }
    dest[i] = 1.0 / (1.0 + exp(-z)) - vecY[i];
}

static void genVectorLabels(double* matLabels, double* dest, int size, double labelIndex)
{
    genVectorLabelsKernel<<<1, size>>>(matLabels, dest, labelIndex);
}

static void gradientAndStepAdd(double* matX, double* vecP, double* vecY, double* parameters, double* result, int size)
{
    double* vecPsubY = vecP;
    sigmoidKernelAndSubY<<<1, size>>>(matX, vecY, parameters, vecPsubY);
    // 取平均值
    double alpha = 1.0 / size;
    const double beta = 0;
    const int incx = 1;
    const int incy = 1;
    const int lda = FEATURE_COUNT;
    CUBLAS_CHECK(
        cublasDgemv(cublasH, CUBLAS_OP_N, FEATURE_COUNT, size, &alpha, matX, lda, vecPsubY, incx, &beta, result, incy));
    const int n = FEATURE_COUNT;
    const double learningRate = -LEARNING_RATE; // 梯度下降需要减去梯度，所以学习率取负
    /*
    parameters=parameters-LEARNING_RATE*result
    梯度通常不会归一化
    */
    cublasDaxpy(cublasH, FEATURE_COUNT, &learningRate, result, 1, parameters, 1);
}


/*
labelIndex取{0,1,2}作为预测类型，例如取0，表示正类是0，负类是1和2
*/
static void trainSingleClassifier(Classifier* classifier, int labelIndex)
{
    // 考虑到公式 梯度J(B)=X(P-Y)/N
    // 需要有个地方存储每次计算得到的P，带遮罩的Y和结果梯度向量
    double* parameters;
    double* stepResult;
    double* vecP;
    double* vecY;
    cudaMalloc((void**)&(parameters), sizeof(double) * FEATURE_COUNT);
    cudaMalloc((void**)&(stepResult), sizeof(double) * FEATURE_COUNT);
    cudaMalloc((void**)&(vecP), sizeof(double) * TRAIN_CAP);
    cudaMalloc((void**)&(vecY), sizeof(double) * TRAIN_CAP);
    cudaMemcpy(parameters, (void*)classifier->parameters, sizeof(double) * FEATURE_COUNT, cudaMemcpyHostToDevice);
    genVectorLabels(trainSet.labels.data, vecY, TRAIN_CAP, (double)labelIndex);
    static const int iterationTimes = ITERATION_TIMES;
    
    // 初始化记录器（可选功能，不影响核心逻辑）
    TrainingLogger* logger = createLogger(labelIndex, trainSet.features.data, vecY, TRAIN_CAP, iterationTimes);
    logTrainingStep(logger, parameters, NULL);  // 记录初始状态
    
    for (int i = 0; i < iterationTimes; i++)
    {
        gradientAndStepAdd(trainSet.features.data, vecP, vecY, parameters, stepResult, TRAIN_CAP);
        logTrainingStep(logger, parameters, stepResult);  // 记录训练过程
    }
    
    destroyLogger(logger);
    
    cudaMemcpy(classifier->parameters, parameters, sizeof(double) * FEATURE_COUNT, cudaMemcpyDeviceToHost);
    cudaFree(parameters);
    cudaFree(stepResult);
    cudaFree(vecP);
    cudaFree(vecY);
}

static void trainModel()
{
    trainSingleClassifier(irisClassifier.classifier + 0, 0);
    trainSingleClassifier(irisClassifier.classifier + 1, 1);
    trainSingleClassifier(irisClassifier.classifier + 2, 2);
    for (int i = 0; i < 3; i++)
    {
        printf("Classifier %d : [%lf,%lf,%lf,%lf,%lf]\n", i, irisClassifier.classifier[i].parameters[0],
               irisClassifier.classifier[i].parameters[1], irisClassifier.classifier[i].parameters[2],
               irisClassifier.classifier[i].parameters[3], irisClassifier.classifier[i].parameters[4]);
    }
}

static inline int de(double a, double b)
{
    register double diff = a - b;
    return (diff < 0 ? -diff : diff) < 0.01;
}

static void testModel()
{
    int testCap = testSet.features.m;
    int correctCount = 0;
    double* parameters = NULL;
    double* vecZ[3] = {NULL};
    cudaMalloc((void**)&parameters, sizeof(double) * FEATURE_COUNT);
    cudaMalloc((void**)&vecZ[0], sizeof(double) * testCap);
    cudaMalloc((void**)&vecZ[1], sizeof(double) * testCap);
    cudaMalloc((void**)&vecZ[2], sizeof(double) * testCap);
    double alpha = 1.0;
    const double beta = 0;
    const int incx = 1;
    const int incy = 1;
    const int lda = FEATURE_COUNT;
    for (int i = 0; i < 3; i++)
    {
        cudaMemcpy(parameters, (void*)irisClassifier.classifier[i].parameters, sizeof(double) * FEATURE_COUNT,
                   cudaMemcpyHostToDevice);
        CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_T, FEATURE_COUNT, testCap, &alpha, testSet.features.data,
                                 FEATURE_COUNT, parameters, incx, &beta, vecZ[i], incy));
    }
    double* hostVecZ[3];
    double* trueLabels = (double*)malloc(sizeof(double) * testCap);
    hostVecZ[0] = (double*)malloc(sizeof(double) * testCap);
    hostVecZ[1] = (double*)malloc(sizeof(double) * testCap);
    hostVecZ[2] = (double*)malloc(sizeof(double) * testCap);
    cudaMemcpy(trueLabels, testSet.labels.data, sizeof(double) * testCap, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostVecZ[0], vecZ[0], sizeof(double) * testCap, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostVecZ[1], vecZ[1], sizeof(double) * testCap, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostVecZ[2], vecZ[2], sizeof(double) * testCap, cudaMemcpyDeviceToHost);

    for (int i = 0; i < testCap; i++)
    {
        double label = 0;
        double confidence = hostVecZ[0][i];
        if (hostVecZ[1][i] > confidence)
        {
            label = 1;
            confidence = hostVecZ[1][i];
        }
        if (hostVecZ[2][i] > confidence)
        {
            label = 2;
            confidence = hostVecZ[2][i];
        }
        printf("prediction:%.1lf truth:%.1lf possibility:%lf\n", label, trueLabels[i],1.0/(1.0+exp(-confidence)));
        if (de(label, trueLabels[i]))
        {
            correctCount++;
        }
    }

    printf("Test Accuracy: %.3lf%%\n", (double)correctCount / testCap * 100.0);

    cudaFree(parameters);
    cudaFree(vecZ[0]);
    cudaFree(vecZ[1]);
    cudaFree(vecZ[2]);
    free(trueLabels);
    free(hostVecZ[0]);
    free(hostVecZ[1]);
    free(hostVecZ[2]);
}

int main()
{
    CUBLAS_CHECK(cublasCreate(&cublasH));
    readData(DATASET_PATH, SAMPLE_COUNT, FEATURE_COUNT, LABEL_COUNT, TRAIN_CAP);
    trainModel();
    testModel();
    freeMat(&trainSet.features);
    freeMat(&trainSet.labels);
    freeMat(&testSet.features);
    freeMat(&testSet.labels);
    CUBLAS_CHECK(cublasDestroy(cublasH));
    cudaDeviceReset();
    return 0;
}
