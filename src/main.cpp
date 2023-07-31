#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"
#include "dataset/dataset.hpp"
#include "model/model.hpp"

using json = nlohmann::json;

void createDataset();
void trainQNT();
std::shared_ptr<QNT> loadQNT();

int main()
{
  // createDataset();
  // trainQNT();
  auto qnt = loadQNT();
}

void createDataset()
{
  std::ifstream f("data/hyper_params.json");

  json params = json::parse(f);

  f.close();

  STPDataset::generateRandom(params);
}

void trainQNT()
{
  std::ifstream f("data/hyper_params.json");

  json params = json::parse(f);

  f.close();

  auto qnt = QNT();
  qnt.train(params);

  qnt.saveQNT();
}

std::shared_ptr<QNT> loadQNT()
{
  return QNT::loadQNT();
}
