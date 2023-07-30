#ifndef PDB_ANN_PDB
#define PDB_ANN_PDB

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include <string>
#include <fstream>

#include "stp/stp.hpp"

using json = nlohmann::json;

class PDB
{
private:
  torch::Tensor _table;
  std::vector<bool> _states;
  std::vector<int> _pattern;
  STP _goal;

public:
  PDB();
  PDB(STP goalSTP, std::vector<int> pattern);
  int64_t size();
  torch::Tensor getTable();
  STP getSTP();
  void fill();
  static PDB load(std::string path);
  json toJSON();
  void save(std::string path);
};

typedef std::vector<PDB> PDBs;

#endif