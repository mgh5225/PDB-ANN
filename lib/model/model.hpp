#ifndef PDB_ANN_MODEL
#define PDB_ANN_MODEL

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>
#include <tuple>
#include <string>
#include <iomanip>

#include "dataset/dataset.hpp"

using json = nlohmann::json;

class QNT : public torch::nn::Module
{
private:
  std::shared_ptr<torch::nn::Conv2dImpl> conv;
  std::shared_ptr<torch::nn::LinearImpl> fc1;
  std::shared_ptr<torch::nn::LinearImpl> fc2;
  std::shared_ptr<torch::nn::LinearImpl> fc3;
  std::shared_ptr<torch::nn::LinearImpl> fc4;

public:
  QNT();
  torch::Tensor forward(torch::Tensor x);
  int getHeuristic(torch::Tensor v, float q);
  torch::Tensor getHeuristic(torch::Tensor v, torch::Tensor q);
  void train(json params);
  void saveQNT();
  static std::shared_ptr<QNT> loadQNT();
  double findQStar(json params);
  std::vector<std::tuple<std::string, int, int>> run(json params);
};

#endif