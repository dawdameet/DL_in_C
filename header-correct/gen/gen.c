/*
    UwU

    Structure of neural network
    (Your comments retained)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // <-- ADDED THIS

// --- Activation Functions ---
double LeakyReLU(double x) { return x > 0 ? x : 0.01 * x; }
double LeakyReLU_grad(double x) { return x > 0 ? 1.0 : 0.01; }

double linear(double x) { return x; }
double linear_grad(double x) { return 1.0; }

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double s_grad(double x) { double s = sigmoid(x); return s*(1.0 - s); }

double ReLU(double x) {return x>0 ? x : 0 ;}
double r_grad(double x) { return x>0 ? 1.0 : 0.0; }

// --- Structs ---
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

// --- Helper Functions (MODIFIED) ---
double (*get_activation(const char* s))(double) {
    if (strcmp(s, "sigmoid") == 0) return sigmoid;
    if (strcmp(s, "relu") == 0) return ReLU;
    if (strcmp(s, "leaky_relu") == 0) return LeakyReLU;
    if (strcmp(s, "linear") == 0) return linear;
    return linear; // Default to linear
}

void assign_derivative(HLayer* layer) {
    if (layer->activation == sigmoid) layer->activation_derivative = s_grad;
    else if (layer->activation == ReLU) layer->activation_derivative = r_grad;
    else if (layer->activation == linear) layer->activation_derivative = linear_grad;
    else if (layer->activation == LeakyReLU) layer->activation_derivative = LeakyReLU_grad;
    else layer->activation_derivative = linear_grad; // Default
}

// --- Core NN Logic (MODIFIED) ---
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

    assign_derivative(layer); // <-- MODIFIED this part

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

// MODIFIED to accept strings for activation
HLayer** init_network_custom(int input_size, int* hidden_sizes, int num_hidden, int output_size, 
                             const char* hidden_act_str, const char* output_act_str) {
    
    double (*hidden_act)(double) = get_activation(hidden_act_str);
    double (*output_act)(double) = get_activation(output_act_str);

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

    for (i=num_layers-2; i>=0; i--){ 
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
            layer->bias_vect[j] -= lr * layer->delta[j];
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
        free(layer->delta);
        free(layer);
    }
    free(network);
}

// --- PYTHON API WRAPPERS ---
// These are the only functions Python will see

void* create_network(int input_size, int* hidden_sizes, int num_hidden, int output_size, 
                     const char* hidden_act, const char* output_act) {
    // Seed random number generator when network is created
    srand(time(NULL)); 
    return (void*)init_network_custom(input_size, hidden_sizes, num_hidden, output_size, hidden_act, output_act);
}

void train_step(void* network_handle, int num_layers, double* input, double* target, double lr) {
    HLayer** network = (HLayer**)network_handle;
    backprop(network, num_layers, input, target, lr);
}

void predict(void* network_handle, int num_layers, double* input, double* output_buffer) {
    HLayer** network = (HLayer**)network_handle;
    double* result = forward(network, num_layers, input);
    
    // Get output size from the last layer
    int output_size = network[num_layers - 1]->num_neurons;
    
    // Copy result to the Python-managed buffer
    for (int i = 0; i < output_size; i++) {
        output_buffer[i] = result[i];
    }
}

void free_network(void* network_handle, int num_layers) {
    HLayer** network = (HLayer**)network_handle;
    free_mem(network, num_layers);
}

// --- NO MAIN FUNCTION ---