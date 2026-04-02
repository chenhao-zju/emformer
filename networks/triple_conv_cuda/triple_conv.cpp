#include <torch/extension.h>
#include <vector>

torch::Tensor triple_conv_forward(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor weight3,
    torch::Tensor weight5);

std::vector<torch::Tensor> triple_conv_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor weight3,
    torch::Tensor weight5);

torch::Tensor triple_conv(
    torch::Tensor input,
    torch::Tensor weight1,
    torch::Tensor weight3,
    torch::Tensor weight5) {
    return triple_conv_forward(input, weight1, weight3, weight5);
}

std::vector<torch::Tensor> create_triple_conv_weights(int64_t in_channels, int64_t out_channels) {
    auto weight1 = torch::randn({out_channels, in_channels});
    auto weight3 = torch::randn({out_channels, in_channels * 9});
    auto weight5 = torch::randn({out_channels, in_channels * 25});
    
    weight1 = torch::nn::init::xavier_uniform_(weight1);
    weight3 = torch::nn::init::xavier_uniform_(weight3);
    weight5 = torch::nn::init::xavier_uniform_(weight5);
    
    return {weight1, weight3, weight5};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triple_conv", &triple_conv, "Triple convolution operator");
    m.def("create_weights", &create_triple_conv_weights, "Create weights for triple convolution");
    m.def("forward", &triple_conv_forward, "Triple Conv forward");
    m.def("backward", &triple_conv_backward, "Triple Conv backward");
}