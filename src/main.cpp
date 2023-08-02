#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <optional>
#include <fstream>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"
#include "dataset/dataset.hpp"
#include "model/model.hpp"

using json = nlohmann::json;

void createPDBs();
void createDataset();
void trainQNT();
void run(std::vector<int> state, std::vector<int> pattern, std::vector<int> dimension, std::optional<std::vector<double>> q);
std::shared_ptr<QNT> loadQNT();

int main(int argc, char *argv[])
{
  auto options = cxxopts::Options("main", "Pattern Database + ANN");

  auto glob_options = options.add_options();
  glob_options("h,help", "Print usage");
  glob_options("pdb", "Create PDBs");
  glob_options("create", "Create random database based on created PDBs");
  glob_options("t,train", "Train ANN");
  glob_options("r,run", "Run ANN");

  auto run_options = options.add_options("run");
  run_options("s,state", "State for ANN", cxxopts::value<std::vector<int>>());
  run_options("p,pattern", "Pattern for ANN", cxxopts::value<std::vector<int>>());
  run_options("d,dim", "Dimension for ANN", cxxopts::value<std::vector<int>>());
  run_options("q", "List of q for ANN", cxxopts::value<std::vector<double>>());

  auto result = options.parse(argc, argv);

  if (result.count("help"))
  {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  if (result.count("pdb"))
    createPDBs();

  if (result.count("create"))
    createDataset();

  if (result.count("train"))
    trainQNT();

  if (result.count("run"))
  {
    if (!result.count("state"))
    {
      std::cerr << "State is not given" << std::endl;
      exit(-1);
    }
    if (!result.count("pattern"))
    {
      std::cerr << "Pattern is not given" << std::endl;
      exit(-1);
    }

    if (!result.count("dim"))
    {
      std::cerr << "Dimension is not given" << std::endl;
      exit(-1);
    }

    std::vector<int> state = result["state"].as<std::vector<int>>();
    std::vector<int> pattern = result["pattern"].as<std::vector<int>>();
    std::vector<int> dimension = result["dim"].as<std::vector<int>>();

    std::optional<std::vector<double>> q = std::nullopt;

    if (result.count("q"))
    {
      q = result["q"].as<std::vector<double>>();
    }

    if (dimension.size() != 2)
    {
      std::cerr << "Dimension should be 2d" << std::endl;
      exit(-1);
    }

    run(state, pattern, dimension, q);
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

void run(std::vector<int> state, std::vector<int> pattern, std::vector<int> dimension, std::optional<std::vector<double>> q)
{
  torch::NoGradGuard no_grad;

  int width = dimension[0];
  int height = dimension[1];

  auto stp = STP(width, height);
  stp.initState(state);
  stp.toAbstract(pattern);

  torch::Tensor dual = stp.getFlattenState(pattern);

  auto options = torch::TensorOptions().dtype(torch::kFloat);

  torch::Tensor data = torch::zeros({1, (int)pattern.size(), height, width}, options);

  for (int i = 0; i < pattern.size(); i++)
  {
    int tile = dual[i].item<int>();
    int x_t = static_cast<int>(tile / width);
    int y_t = tile % width;

    data[0][i][x_t][y_t] = 1;
  }

  auto qnt = loadQNT();

  torch::Tensor classes = qnt->forward(data);

  std::cout << classes.data() << std::endl;

  if (q.has_value())
  {
    std::cout << "q"
              << "\t\t"
              << "h" << std::endl;
    for (auto &elem : q.value())
    {
      std::cout << elem << "\t\t" << qnt->getHeuristic(data, elem) << std::endl;
    }
  }
}
