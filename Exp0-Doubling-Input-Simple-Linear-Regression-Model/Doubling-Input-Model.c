/**
This example provides a basic understanding of linear models, represented by the equation y = mx + c or y = wx + b. In this context, we are training a model to learn the optimal value of the weight (w), which represents the slope of the line, and the bias (b), which represents 
the y-intercept.

In this example, we use imaginary sample data where the independent variable (x) is the sopme number and the dependent variable (y) is the twice of input. Our goal is to train the model using this data to learn the relationship between 
input and output.

Once the model is trained, we test it by providing random numbers as  an input, and the model predicts the outrput. We evaluate the model using a single metric, the loss (cost) function, which serves as our objective function to quantify the prediction error 
during the training process.
**/

#include <stdio.h>
#include <stdlib.h>


//
// Macros
//
#define SUCCESS             1
#define FAILURE             0

//
// Dataset for Training - 60%
//
static double DatapointTrain[][2] = {
    {1.234,     2.468},
    {3,         6},
    {52.8965,   105.7},
    {76.098,    152.196},
    {99,        198},
    
    {1.0641,    2.1282},
    {0.12316,   0.24},
    {23.67340,  47.347},
    {26,        52},
    {45.0876,   90.1752},
    {0,         0}
};

//
// Dataset for Validation - 20%
//
static double DatapointValidation[][2] = {
    {20,                    40.0},
    {0.5678,                1.1356},
    {6.8999999999999995,    13.80},
    {10.4,                  20.8},
    {9.1,                   18.2},
    {4.6,                   9.2},
    {3.3000000000000003,    6.6}
};

//
// Dataset for Testing - 20%
//
static double DatapointTest[][2] = {
  //  x          y
    {11.5,      0.0},
    {45.23,     0.0},
    {90.123,    0.0},
    {0.555,     0.0},
    {100.111,   0.0},
    {0,         0.0}
};

static double weight = 0;
static double bias = 0;


void SplitDataset (void) {
    //
    // Harcoded for current experiment
    //
    printf ("Dataset split Hardcoded.\n");
}


/**
 *  Responsible for predicting (y = wx+b)
**/
static double* PredictOutput (double inputs[][2], int size, double weight, double bias) {

    double* y_predicted = (double*)malloc(size * sizeof(double));
    if (y_predicted == NULL) {
        printf ("Memory allocation failed\n");
        exit (EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        y_predicted[i] = inputs[i][0] * weight + bias; // y = wx+b
    }

    return y_predicted;
}

static double CostFunction (double inputs[][2], int size, double weight, double bias) {
    double loss_value = 0;
    double sum_loss = 0;
    double *y_predicted = PredictOutput (inputs, size, weight, bias);

    for (int i = 0; i < size; i++) {
        loss_value = (inputs[i][1] - y_predicted[i]) * (inputs[i][1] - y_predicted[i]);
        sum_loss += loss_value;
    }

    free (y_predicted);
    return (sum_loss / size);
}

/**
 * Calculates gradient for weight
**/
static double WeightGradient (double inputs[][2], int size) {
    double grad = 0;
    double *y_predicted = PredictOutput (inputs, size, weight, bias);

    for (int i = 0; i < size; i++) {
        grad += (y_predicted[i] - inputs[i][1]) * inputs[i][0];
    }

    free(y_predicted);
    return (2 * grad) / size;
}

/**
 * Calculates gradient for bias
**/
static double BiasGradient (double inputs[][2], int size) {
    double grad = 0;
    double *y_predicted = PredictOutput (inputs, size, weight, bias);

    for (int i = 0; i < size; i++) {
        grad += (y_predicted[i] - inputs[i][1]);
    }

    free (y_predicted);
    return (2 * grad) / size;
}

/**
 * Linear regression-Gradiant decent algorithm done on Training data
 */
void TrainOnTrainingData(void) {

    int     Epoch        = 10000;
    double  LearningRate = 0.0001;  
    int     size         = sizeof (DatapointTrain) / (2 * sizeof(DatapointTrain[0][0]));
    double  Loss         = 0;
    double  grad_w       = 0;
    double  grad_b       = 0;

    for (int i = 1; i <= Epoch; i++) {
        Loss   = CostFunction (DatapointTrain, size, weight, bias);
        grad_w = WeightGradient (DatapointTrain, size);
        grad_b = BiasGradient (DatapointTrain, size);

        weight = weight - LearningRate * grad_w;
        bias   = bias - LearningRate * grad_b;

        printf ("Epoch %d ---- Loss: %lf \n", i, Loss);
        printf ("Weight: %lf, Bias: %lf, Grad_W: %lf, Grad_B: %lf\n", weight, bias, grad_w, grad_b); 
    }
    printf ("\n");
    printf ("Model Loss: %lf \n", Loss);
    printf ("Optimum Weight: %lf \n", weight);
    printf ("Optimum Bias: %lf \n", bias);
    printf ("===========================");
    printf ("\n\n\n\n");
}

/**
 * Testing Data: It is only run after the final model has been fully trained and tuned. It provides the ultimate measure of the model's performance.
**/

void ValidateTrainedModel (void) {

    int     size         = sizeof(DatapointValidation) / (2 * sizeof(DatapointValidation[0][0]));
    double  Loss         = 0;
    double* predictions  = PredictOutput(DatapointValidation, size, weight, bias);

    Loss = CostFunction (DatapointTrain, size, weight, bias);

    printf ("Prediction for Validation set: \n\n");
    printf ("Weight: %lf \n", weight);
    printf ("Bias: %lf \n", bias);
    printf ("Prediction Model Loss: %lf \n", Loss);

    for (int i = 0; i < size; i++) {
        printf ("Input %lf  : Output %lf\n", DatapointValidation[i][0], predictions[i]);
    }

    free (predictions);

    printf ("===========================");
    printf ("\n\n\n\n");
}

/**
 * Testing Data: It is only run after the final model has been fully trained and tuned. It provides the ultimate measure of the model's performance.
**/

void TestModel (void) {

    int     size         = sizeof (DatapointTest) / (2 * sizeof(DatapointTest[0][0]));
    double* predictions  = PredictOutput (DatapointTest, size, weight, bias);

    printf ("Prediction for Test\n\n");

    for (int i = 0; i < size; i++) {
        printf ("Input %lf  : Output %lf\n", DatapointTest[i][0], predictions[i]);
    }

    free (predictions);

    printf ("===========================");
    printf ("\n\n\n\n");
}

int main () {

    // Step 1:
    SplitDataset ();

    /** Step 2
     * Purpose: Used to train the model.
     * 
     * Role:
     *      The model learns patterns, relationships, and parameters (e.g., weights in linear regression or layers in deep learning) from this data.
     *      The goal is to minimize the error (loss function) on this data by adjusting model parameters during training.
     * 
    **/
    TrainOnTrainingData ();

    /** Step 3
     * Purpose: Used to tune the model and prevent overfitting.
     * 
     * Role:
     *      During training, the model's performance is periodically evaluated on the validation data to monitor progress and guide hyperparameter tuning (e.g., learning rate, number of layers, regularization strength).
     * It helps in:
     *      Detecting overfitting: If the model performs well on the training data but poorly on the validation data, it’s overfitting.
     * Interaction: 
     *      The validation data is not used to adjust the model's parameters directly, but it guides hyperparameter choices and early stopping.
     * 
    **/
    ValidateTrainedModel ();

    if(1) {
        /** Step 4
         * Purpose: Used to evaluate the final model's performance.
         * Role:
         *      After the model has been trained and tuned (using training and validation data), it is tested on the testing data.
         *      This step provides an unbiased estimate of the model's generalization ability (how well it performs on unseen data).
        **/
        TestModel ();
    }

    return SUCCESS;
}