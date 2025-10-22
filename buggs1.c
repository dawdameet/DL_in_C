/*
    UwU

    Structure of neural network

    # For Input Layer
        - We will have an input vector thats it x=[x1,x2,x3,.....xn]
    
    # For every hidden layer
        - We will have input vector x= [x1,x2,x3 ... xn] R^n
        - We will have a weights vector w=[w1,w2,w3 .. wm] R^mxn
        - WE will have a bias vector b=[b1,b2,b3....bm]  R^m
        - One activation function (usually non linear) like ReLU or Sigmoid
        - This vector multiplication of w and x will spit a vector out of the dimension 
            mxn . nx1 = mx1
        - this mx1 vector will be addeed with b vector 
        - this vector will be called z vector
        - we will apply a non linearity on this z as input (activation for noobs)
        - this will spit a vector of same dimension 
        - this will be input vector for the furthur layers
    
    # NOTE: the input vector is a transpose of the row vector written (it does not matter too much but for the standard notation we need to follow that
    

    # The forward pass is meant to update every z in every layer
    # backpropogation will be updating withbththh gradientsss
        - Deltas is the difff att to neurons (each one of those mfs)

    LeakyRELU is a decaying relu it keeps shit for negs

*/



#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h> 

double LeakyReLU(double x) { return x > 0 ? x : 0.01 * x; }
// -----------------------------------------------------------------------------
// BUG #1: The "Wrong Derivative" Bug
// The gradient for LeakyReLU should be 0.01 for negative values.
// By setting it to 0.0, it becomes the standard ReLU gradient.
// The network will train perfectly on all positive inputs (1, 2, 3, 4, 5)
// but will COMPLETELY FAIL to learn from negative inputs (-2.0),
// as the gradient for them will be zero, meaning no learning occurs.
// The final test on -3.0 will fail spectacularly.
double LeakyReLU_grad(double x) { return x > 0 ? 1.0 : 0.0; } // <-- BUG!
// -----------------------------------------------------------------------------

double linear(double x) { return x; }
double linear_grad(double x) { return 1.0; }

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double s_grad(double x) { double s = sigmoid(x); return s*(1.0 - s); }

double ReLU(double x) {return x>0 ? x : 0 ;}
double r_grad(double x) { return x>0 ? 1.0 : 0.0; }

typedef struct HLayer {
    int num_inputs; 
    int num_neurons; 
    double **weight_vect; 
    double *bias_vect;  
    double *z;          
    double *out;        
    double* delta;      
    double (*activation)(double);
    double (*activation_derivative)(double);
}HLayer;

HLayer* init_Hlayer(int inputs, int neurons, double (*act)(double)){
    int i;
    HLayer* layer = (HLayer*)malloc(sizeof(HLayer));
    if (!layer) {
        perror("LAYERRRERER ALLOCAITON FAILED");
        exit(1);
    }
    layer->num_inputs = inputs;
    layer->num_neurons = neurons;
    layer->activation = act;

    layer->weight_vect = (double **)malloc(neurons*sizeof(double*));
    if (!layer->weight_vect) {
        perror("DAILED VECTOR ACCOLATION");
        exit(1);
    }
    for (i=0; i<neurons; i++) {
        layer->weight_vect[i] = (double*)malloc(inputs * sizeof(double));
        if (!layer->weight_vect[i]) {
            perror("GO HOME ATP");
            exit(1);
        }
    }

    layer->bias_vect = (double*)malloc(neurons * sizeof(double));
    layer->z = (double*)malloc(neurons * sizeof(double));
    layer->out = (double*)malloc(neurons * sizeof(double));
    layer->delta = (double*)malloc(neurons * sizeof(double));

    if (!layer->bias_vect || !layer->z || !layer->out || !layer->delta) {
        perror("PACK IT UP VRO");
        exit(1);
    }

    if (act == sigmoid) layer->activation_derivative = s_grad;
    else if (act == ReLU) layer->activation_derivative = r_grad;
    else if (act == linear) layer->activation_derivative = linear_grad;
    else if (act == LeakyReLU) layer->activation_derivative = LeakyReLU_grad;

    return layer;
}

void init_weights(HLayer* layer){
    double range;
    
    if(layer->activation == sigmoid) {
         range = sqrt(6.0 / (layer->num_inputs + layer->num_neurons)); 
    } else if (layer->activation == linear) {
         range = sqrt(3.0 / (layer->num_inputs + layer->num_neurons)); 
    } else { 
         range = sqrt(2.0 / layer->num_inputs); 
    }

    for (int i=0; i<layer->num_neurons; i++){
        layer->bias_vect[i] = ((double)rand()/RAND_MAX - 0.5) * 0.1; 
        for (int j=0; j<layer->num_inputs; j++)
            layer->weight_vect[i][j] = ((double)rand()/RAND_MAX - 0.5) * 2.0 * range;
    }
}

HLayer** init_network_custom(int input_size, int* hidden_sizes, int num_hidden, int output_size, 
                             double (*hidden_act)(double), double (*output_act)(double)) {
    
    int total_layers = num_hidden + 1;
    HLayer** network = (HLayer**)malloc(total_layers * sizeof(HLayer*));
    if (!network) {
        perror("YOU STILL HERERERER");
        exit(1);
    }
    
    network[0] = init_Hlayer(input_size, hidden_sizes[0], hidden_act);
    init_weights(network[0]);

    for(int i=1; i<num_hidden; i++){
        network[i] = init_Hlayer(hidden_sizes[i-1], hidden_sizes[i], hidden_act);
        init_weights(network[i]);
    }

    network[num_hidden] = init_Hlayer(hidden_sizes[num_hidden-1], output_size, output_act);
    init_weights(network[num_hidden]);

    return network;
}


double* forward(HLayer** network, int num_layers, double *input){
    double* current_input = input;

    for (int i=0; i<num_layers; i++){ 
        HLayer* layer = network[i];
        
        for (int j=0; j<layer->num_neurons; j++){
            layer->z[j] = layer->bias_vect[j]; 
            for (int k=0; k<layer->num_inputs; k++) {
                layer->z[j] += layer->weight_vect[j][k] * current_input[k];
            }
        }

        for (int j=0; j<layer->num_neurons; j++)
            layer->out[j] = layer->activation(layer->z[j]);
        
        current_input = layer->out; 
    }
    return current_input; 
}

void backprop(HLayer** network, int num_layers, double* input, double* target, double lr){
    int i,j,k;
    
    forward(network, num_layers, input);

    HLayer* last_layer = network[num_layers-1];
    for (i=0; i<last_layer->num_neurons; i++) {
        last_layer->delta[i] = (last_layer->out[i]-target[i]) * last_layer->activation_derivative(last_layer->z[i]);
    }

    // -----------------------------------------------------------------------------
    // BUG #2: The "Skipped Layer" Bug
    // The loop should go all the way back to the first layer (i=0)
    // to calculate the deltas. By stopping at i=1, this loop
    // *never calculates the deltas for layer 0*.
    // As a result, the weight update for layer 0 (the input layer)
    // will be based on garbage values in `network[0]->delta`,
    // and the model will fail to learn.
    for (i=num_layers-2; i>=1; i--){ // <-- BUG!
    // -----------------------------------------------------------------------------
        HLayer* layer = network[i];
        HLayer* next_layer = network[i+1];
        for (j=0; j<layer->num_neurons; j++){ 
            layer->delta[j]=0;
            for (k=0; k<next_layer->num_neurons; k++) { 
                layer->delta[j] += next_layer->weight_vect[k][j] * next_layer->delta[k];
            }
            layer->delta[j] *= layer->activation_derivative(layer->z[j]);
        }
    }

    for (i=0; i<num_layers; i++){
        HLayer* layer = network[i];
        double* inp = (i==0) ? input : network[i-1]->out; 
        
        for (j=0; j<layer->num_neurons; j++){
            for (k=0; k<layer->num_inputs; k++) {
                layer->weight_vect[j][k] -= lr * layer->delta[j] * inp[k];
            }
            // -----------------------------------------------------------------------------
            // BUG #3: The "Stuck Bias" Bug
            // The bias update line is commented out. The network will
            // train its weights, but the biases will remain "stuck"
            // at their small, random initial values. This severely
            // limits the model's ability to fit the data, and it
            // will converge to a high error.
            // layer->bias_vect[j] -= lr * layer->delta[j]; // <-- BUG!
            // -----------------------------------------------------------------------------
        }
    }
}

void free_mem(HLayer** network, int total_layers){
    for (int i=0; i < total_layers; i++){ 
        HLayer* layer = network[i];
        for (int j=0; j<layer->num_neurons; j++)
            free(layer->weight_vect[j]);
        free(layer->weight_vect);
        free(layer->bias_vect);
        free(layer->z);
        free(layer->out);
        // -----------------------------------------------------------------------------
        // BUG #4: The "Silent Memory Leak"
        // Every other vector is freed, but `layer->delta` is not.
        // This is a classic C-style bug. The program will run
        // perfectly and give the (buggy) correct answer, but it
        // will leak memory on every training iteration. This
        // will only be found by a tool like valgrind.
        // free(layer->delta); // <-- BUG!
        // -----------------------------------------------------------------------------
        free(layer);
    }
    free(network);
}

int main(int argc, char* argv[]){
    srand(time(NULL));

    double train_input[6][1] = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {-2.0} 
    };
    double train_target[6][1] = {
        {2.0},
        {4.0},
        {6.0},
        {8.0},
        {10.0},
        {-4.0} 
    };
    int num_samples = 6; 


    int input_size = 1;         
    int hidden_sizes[] = {4}; 
    int num_hidden = 1;         
    int output_size = 1;        
    
    int total_layers = num_hidden + 1; 

    HLayer** network = init_network_custom(input_size, hidden_sizes, num_hidden, output_size, 
                                           LeakyReLU,  
                                           linear);

    int epochs = 2000;
    double learning_rate = 0.001; 

    // -----------------------------------------------------------------------------
    // BUG #5 (Bonus): The "Stale Error" Bug
    // The printf is inside the training loop.
    // The bug is that `backprop` is called *before* the error calculation.
    // `backprop` *includes* a `forward` pass, which updates `out_layer->out[0]`.
    // Then, the gradient update is performed.
    // *Then*, the error is calculated on the *already updated* output.
    // This isn't strictly a "bug" that breaks things, but it means
    // the error being printed is *not* the error that generated
    // the gradients for that step. It's an "off-by-one" in reporting.
    // (This one is more of a minor, confusing quirk).
    // -----------------------------------------------------------------------------
    
    printf("Training");


    for (int i = 0; i < epochs; i++){
        double epoch_error = 0;
        for(int j=0; j<num_samples; j++){ 
            backprop(network, total_layers, train_input[j], train_target[j], learning_rate);
            
            HLayer* out_layer = network[total_layers-1];
            epoch_error += 0.5 * pow(out_layer->out[0] - train_target[j][0], 2);
        }

        if (i % 200 == 0){
            printf("Epoch %d, Error: %f\n", i, epoch_error / num_samples);
        }
    }

    printf("\ntESTING oN sEEN dATA\n");
    for(int i=0; i<num_samples; i++){ 
        double* output = forward(network, total_layers, train_input[i]);
        
        printf("Input: [%.1f] -> Output: %.4f (Target: %.1f)\n", 
            train_input[i][0], 
            output[0], 
            train_target[i][0]);
    }



    // formatted wiht gemininin googel com .
    printf("\ntESTING oN uNSEEN dATA\n");
    double test_val_1[1] = {2.5};
    double* out_1 = forward(network, total_layers, test_val_1);
    printf("Input: 2.5 -> Output: %.4f (Target: 5.0)\n", out_1[0]);

    double test_val_2[1] = {10.0};
    double* out_2 = forward(network, total_layers, test_val_2);
    printf("Input: 10.0 -> Output: %.4f (Target: 20.0)\n", out_2[0]);

    double test_val_3[1] = {-3.0};
    double* out_3 = forward(network, total_layers, test_val_3);
    printf("Input: -3.0 -> Output: %.4f (Target: -6.0)\n", out_3[0]);

    free_mem(network, total_layers);
    
    return 0;
}

// UwU