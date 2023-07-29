#ifndef PDB_ANN_MODEL
#define PDB_ANN_MODEL

#include <torch/torch.h>
#include <iostream>

class QNT_5_5 : torch::nn::Module
{
private:
  std::shared_ptr<torch::nn::Conv2dImpl> conv;
  std::shared_ptr<torch::nn::LinearImpl> fc1;
  std::shared_ptr<torch::nn::LinearImpl> fc2;
  std::shared_ptr<torch::nn::LinearImpl> fc3;

public:
  QNT_5_5();
  torch::Tensor forward(torch::Tensor x);
  int getHeuristic(torch::Tensor v, float q);
};

#endif