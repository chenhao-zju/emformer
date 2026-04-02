#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void triple_conv_forward_kernel(
    const scalar_t* input,
    const scalar_t* weight1, const scalar_t* weight3, const scalar_t* weight5,
    scalar_t* output,
    const int batch_size, const int in_channels, const int out_channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_size, const int padding) {
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    const int input_size_per_channel = input_height * input_width;
    const int output_size_per_channel = output_height * output_width;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
         index < output_size; 
         index += blockDim.x * gridDim.x) {
        
        const int n = index / (out_channels * output_size_per_channel);
        const int c_out = (index / output_size_per_channel) % out_channels;
        const int h_out = (index / output_width) % output_height;
        const int w_out = index % output_width;
        
        scalar_t result = 0;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            const int input_start_idx = (n * in_channels + c_in) * input_size_per_channel;
            
            for (int kh = -padding; kh <= padding; kh++) {
                for (int kw = -padding; kw <= padding; kw++) {
                    const int h_in = h_out + kh;
                    const int w_in = w_out + kw;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        const int input_idx = input_start_idx + h_in * input_width + w_in;
                        
                        scalar_t combined_weight = 0;
                        
                        if (kh == 0 && kw == 0) {
                            const int weight1_idx = c_out * in_channels + c_in;
                            combined_weight += weight1[weight1_idx];
                        }
                        
                        if (abs(kh) <= 1 && abs(kw) <= 1) {
                            const int weight3_idx = (c_out * in_channels + c_in) * 9 + (kh + 1) * 3 + (kw + 1);
                            combined_weight += weight3[weight3_idx];
                        }
                        
                        if (abs(kh) <= 2 && abs(kw) <= 2) {
                            const int weight5_idx = (c_out * in_channels + c_in) * 25 + (kh + 2) * 5 + (kw + 2);
                            combined_weight += weight5[weight5_idx];
                        }
                        
                        result += input[input_idx] * combined_weight;
                    }
                }
            }
        }
        
        output[index] = result;
    }
}

template <typename scalar_t>
__global__ void triple_conv_weight_grad_kernel(
    const scalar_t* grad_output,
    const scalar_t* input,
    scalar_t* grad_weight1, scalar_t* grad_weight3, scalar_t* grad_weight5,
    const int batch_size, const int in_channels, const int out_channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_size, const int padding) {
    
    const int output_size = batch_size * out_channels * output_height * output_width;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
         index < output_size; 
         index += blockDim.x * gridDim.x) {
        
        const int n = index / (out_channels * output_height * output_width);
        const int c_out = (index / (output_height * output_width)) % out_channels;
        const int h_out = (index / output_width) % output_height;
        const int w_out = index % output_width;
        
        const scalar_t grad = grad_output[index];
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            const int input_start_idx = (n * in_channels + c_in) * input_height * input_width;
            
            for (int kh = -padding; kh <= padding; kh++) {
                for (int kw = -padding; kw <= padding; kw++) {
                    const int h_in = h_out + kh;
                    const int w_in = w_out + kw;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        const int input_idx = input_start_idx + h_in * input_width + w_in;
                        const scalar_t input_val = input[input_idx];
                        
                        // const scalar_t grad_divided = grad * input_val / 3.0;
                        const scalar_t grad_divided = grad * input_val;
                        
                        if (kh == 0 && kw == 0) {
                            const int grad_weight1_idx = c_out * in_channels + c_in;
                            atomicAdd(&grad_weight1[grad_weight1_idx], grad_divided);
                        }
                        
                        if (abs(kh) <= 1 && abs(kw) <= 1) {
                            const int grad_weight3_idx = (c_out * in_channels + c_in) * 9 + (kh + 1) * 3 + (kw + 1);
                            atomicAdd(&grad_weight3[grad_weight3_idx], grad_divided);
                        }
                        
                        if (abs(kh) <= 2 && abs(kw) <= 2) {
                            const int grad_weight5_idx = (c_out * in_channels + c_in) * 25 + (kh + 2) * 5 + (kw + 2);
                            atomicAdd(&grad_weight5[grad_weight5_idx], grad_divided);
                        }
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void triple_conv_input_grad_kernel(
    const scalar_t* grad_output,
    const scalar_t* weight1, const scalar_t* weight3, const scalar_t* weight5,
    scalar_t* grad_input,
    const int batch_size, const int in_channels, const int out_channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_size, const int padding) {
    
    const int input_size = batch_size * in_channels * input_height * input_width;
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; 
         index < input_size; 
         index += blockDim.x * gridDim.x) {
        
        const int n = index / (in_channels * input_height * input_width);
        const int c_in = (index / (input_height * input_width)) % in_channels;
        const int h_in = (index / input_width) % input_height;
        const int w_in = index % input_width;
        
        scalar_t grad = 0;
        
        for (int c_out = 0; c_out < out_channels; c_out++) {
            for (int kh = -padding; kh <= padding; kh++) {
                for (int kw = -padding; kw <= padding; kw++) {
                    const int h_out = h_in - kh;
                    const int w_out = w_in - kw;
                    
                    if (h_out >= 0 && h_out < output_height && w_out >= 0 && w_out < output_width) {
                        const int output_idx = (n * out_channels + c_out) * output_height * output_width + 
                                             h_out * output_width + w_out;
                        const scalar_t grad_out_val = grad_output[output_idx];
                        
                        scalar_t combined_weight = 0;
                        
                        if (kh == 0 && kw == 0) {
                            const int weight1_idx = c_out * in_channels + c_in;
                            combined_weight += weight1[weight1_idx];
                        }
                        
                        if (abs(kh) <= 1 && abs(kw) <= 1) {
                            const int weight3_idx = (c_out * in_channels + c_in) * 9 + (kh + 1) * 3 + (kw + 1);
                            combined_weight += weight3[weight3_idx];
                        }
                        
                        if (abs(kh) <= 2 && abs(kw) <= 2) {
                            const int weight5_idx = (c_out * in_channels + c_in) * 25 + (kh + 2) * 5 + (kw + 2);
                            combined_weight += weight5[weight5_idx];
                        }
                        
                        grad += grad_out_val * combined_weight;
                    }
                }
            }
        }
        
        grad_input[index] = grad;
    }
}

torch::Tensor triple_conv_forward(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor weight3, 
    torch::Tensor weight5) {
    
    AT_ASSERTM(input.dim() == 4, "Input must be 4D tensor");
    AT_ASSERTM(weight1.dim() == 2, "weight1 must be 2D tensor");
    AT_ASSERTM(weight3.dim() == 2, "weight3 must be 2D tensor");  
    AT_ASSERTM(weight5.dim() == 2, "weight5 must be 2D tensor");
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight1.size(0);
    
    AT_ASSERTM(weight1.size(1) == in_channels, "weight1 input channels mismatch");
    AT_ASSERTM(weight3.size(0) == out_channels && weight3.size(1) == in_channels * 9, "weight3 dimension error");
    AT_ASSERTM(weight5.size(0) == out_channels && weight5.size(1) == in_channels * 25, "weight5 dimension error");
    
    const int kernel_size = 5;
    const int padding = 2;
    const int output_height = input_height;
    const int output_width = input_width;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * output_height * output_width + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "triple_conv_forward", ([&] {
        triple_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight1.data_ptr<scalar_t>(),
            weight3.data_ptr<scalar_t>(),
            weight5.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, padding);
    }));
    
    cudaDeviceSynchronize();
    
    return output;
}

std::vector<torch::Tensor> triple_conv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor weight3,
    torch::Tensor weight5) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight1.size(0);
    
    const int kernel_size = 5;
    const int padding = 2;
    const int output_height = grad_output.size(2);
    const int output_width = grad_output.size(3);
    
    auto grad_input = torch::zeros_like(input);
    auto grad_weight1 = torch::zeros_like(weight1);
    auto grad_weight3 = torch::zeros_like(weight3);
    auto grad_weight5 = torch::zeros_like(weight5);
    
    const int threads1 = 256;
    const int blocks1 = (batch_size * out_channels * output_height * output_width + threads1 - 1) / threads1;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "triple_conv_backward_weights", ([&] {
        triple_conv_weight_grad_kernel<scalar_t><<<blocks1, threads1>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grad_weight1.data_ptr<scalar_t>(),
            grad_weight3.data_ptr<scalar_t>(),
            grad_weight5.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, padding);
    }));
    
    const int threads2 = 256;
    const int blocks2 = (batch_size * in_channels * input_height * input_width + threads2 - 1) / threads2;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "triple_conv_backward_input", ([&] {
        triple_conv_input_grad_kernel<scalar_t><<<blocks2, threads2>>>(
            grad_output.data_ptr<scalar_t>(),
            weight1.data_ptr<scalar_t>(),
            weight3.data_ptr<scalar_t>(),
            weight5.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, padding);
    }));
    
    cudaDeviceSynchronize();
    
    return {grad_input, grad_weight1, grad_weight3, grad_weight5};
}