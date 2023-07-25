#include "model.hpp"

QNT_5_5::QNT_5_5()
{
  conv = register_module("conv", torch::nn::Conv2d(6, 32, 3));
  fc1 = register_module("fc1", torch::nn::Linear(288, 396));
  fc2 = register_module("fc2", torch::nn::Linear(396, 496));
  fc3 = register_module("fc3", torch::nn::Linear(496, 6));
}

torch::Tensor QNT_5_5::forward(torch::Tensor x)
{
  x = torch::relu(conv->forward(x));
  x = torch::relu(fc1->forward(x));
  x = torch::relu(fc2->forward(x));
  x = torch::softmax(fc3->forward(x), 1);

  return x;
}