#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <getopt.h>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"
#include "dataset/dataset.hpp"
#include "model/model.hpp"

using json = nlohmann::json;

void createPDBs();
void createDataset();
void trainQNT();
void run();
std::shared_ptr<QNT> loadQNT();

int main(int argc, char *argv[])
{
  option opts[] = {
      {"pdb", optional_argument, nullptr, 'p'},
      {"create", optional_argument, nullptr, 'c'},
      {"train", optional_argument, nullptr, 't'},
      {"run", optional_argument, nullptr, 'r'},
      {0},
  };

  while (1)
  {
    const int opt = getopt_long(argc, argv, "pctr::", opts, 0);

    if (opt == -1)
      break;

    switch (opt)
    {
    case 'p':
      createPDBs();
      break;
    case 'c':
      createDataset();
      break;

    case 't':
      trainQNT();
      break;

    case 'r':
      run();
      break;

    default:
      break;
    }
  }

  return 0;
}

void createPDBs()
{
  std::ifstream f("data/hyper_params.json");

  json params = json::parse(f);

  f.close();

  STPDataset::generatePDBs(params);
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

void run()
{
  auto qnt = loadQNT();
}
