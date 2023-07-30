#ifndef PDB_ANN_DATASET
#define PDB_ANN_DATASET

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"

class STPDataset : torch::data::Dataset<STPDataset>
{
private:
  PDBs _pdb_s;

public:
  STPDataset(std::string path);
  STPDataset(PDBs pdb_s);
  STPDataset(STP goalSTP, std::vector<std::vector<int>> pattern_s);
  static void generateRandom(std::string path);
  json toJSON();
  void save(std::string path);
  torch::data::Example<> get(size_t index) override;
};

#endif