#ifndef PDB_ANN_PDB
#define PDB_ANN_PDB

#include <torch/torch.h>

#include <iostream>
#include <vector>
#include <queue>

#include "stp/stp.hpp"

class PDB
{
private:
  torch::Tensor _table;
  std::vector<int> _pattern;
  STP _goal;

public:
  PDB(STP goalSTP, std::vector<int> pattern);
  int size();
  torch::Tensor getTable();
  void fill();
};

#endif