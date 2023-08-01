#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <fstream>
#include <getopt.h>
#include <iomanip>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"
#include "dataset/dataset.hpp"
#include "model/model.hpp"

using json = nlohmann::json;

void createPDBs();
void createDataset();
void trainQNT();
void run(std::map<std::string, std::vector<std::string>> run_args);
std::shared_ptr<QNT> loadQNT();

int main(int argc, char *argv[])
{
  option opts[] = {
      {"pdb", optional_argument, nullptr, 'p'},
      {"create", optional_argument, nullptr, 'c'},
      {"train", optional_argument, nullptr, 't'},
      {"run", required_argument, nullptr, 'r'},
      {0},
  };

  while (1)
  {
    const int opt = getopt_long(argc, argv, "p::c::t::r:", opts, 0);

    if (opt == -1)
      break;

    std::map<std::string, std::vector<std::string>> run_args;
    std::string key = "";

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
      optind--;
      while (optind < argc && *argv[optind] != '-')
      {
        if (*argv[optind] > '9')
          key = argv[optind];
        else
          run_args[key].push_back(argv[optind]);
        optind++;
      }
      run(run_args);
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

void run(std::map<std::string, std::vector<std::string>> run_args)
{
  torch::NoGradGuard no_grad;

  auto perm = run_args.find("perm");
  auto pat = run_args.find("pat");
  auto dim = run_args.find("dim");
  auto q = run_args.find("q");

  if (perm == run_args.end())
  {
    std::cerr << "Permutation is not given" << std::endl;
    return;
  }

  if (pat == run_args.end())
  {
    std::cerr << "Pattern is not given" << std::endl;
    return;
  }

  if (dim == run_args.end())
  {
    std::cerr << "Dimension is not given" << std::endl;
    return;
  }

  if (dim->second.size() != 2)
  {
    std::cerr << "Dimension should be 2d" << std::endl;
    return;
  }

  auto permutation = std::vector<int>();
  auto pattern = std::vector<int>();

  for (auto &elem : perm->second)
  {
    permutation.push_back(std::stoi(elem));
  }

  for (auto &elem : pat->second)
  {
    pattern.push_back(std::stoi(elem));
  }

  int width = std::stoi(dim->second[0]);
  int height = std::stoi(dim->second[1]);

  auto stp = STP(width, height);
  stp.initState(permutation);
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

  if (q != run_args.end())
  {
    std::cout << "q"
              << "\t\t"
              << "h" << std::endl;
    for (auto &elem : q->second)
    {
      float d_q = std::stof(elem);
      std::cout << d_q << "\t\t" << qnt->getHeuristic(data, d_q) << std::endl;
    }
  }
}
