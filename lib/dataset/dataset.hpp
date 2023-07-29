#ifndef PDB_ANN_DATASET
#define PDB_ANN_DATASET

#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"

class STPDataset : torch::data::Dataset<STPDataset>
{
private:
  PDBs _pdb_s;

public:
  STPDataset(PDBs pdb_s);
  STPDataset(STP goalSTP, std::vector<std::vector<int>> pattern_s);
  void generateRandom(int size);
};

#endif