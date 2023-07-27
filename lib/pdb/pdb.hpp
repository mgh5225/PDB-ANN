#ifndef PDB_ANN_PDB
#define PDB_ANN_PDB

#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "stp/stp.hpp"

class PDB
{
private:
  torch::Tensor _table;
  torch::Tensor _pattern;

public:
  PDB(STP goalSTP, std::vector<int> pattern);
  int size();
  void fill();
};

#endif