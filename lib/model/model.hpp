#ifndef PDB_ANN_MODEL
#define PDB_ANN_MODEL

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>
#include <tuple>
#include <string>

#include "dataset/dataset.hpp"

using json = nlohmann::json;

class QNT : public torch::nn::Module
{
private:
  std::shared_ptr<torch::nn::Conv2dImpl> conv;
  std::shared_ptr<torch::nn::LinearImpl> fc1;
  std::shared_ptr<torch::nn::LinearImpl> fc2;
  std::shared_ptr<torch::nn::LinearImpl> fc3;

public:
  QNT();
  torch::Tensor forward(torch::Tensor x);
  int getHeuristic(torch::Tensor v, float q);
  void train(json params);
  void saveQNT();
  static std::shared_ptr<QNT> loadQNT();
};

#endif