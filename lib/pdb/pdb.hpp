#ifndef PDB_ANN_PDB
#define PDB_ANN_PDB

#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

#include "stp/stp.hpp"

class PDB
{
private:
  torch::Tensor _table;
  std::vector<bool> _states;
  std::vector<int> _pattern;
  STP _goal;

public:
  PDB(STP goalSTP, std::vector<int> pattern);
  int64_t size();
  torch::Tensor getTable();
  void fill();
};

#endif