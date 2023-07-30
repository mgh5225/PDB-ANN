#ifndef PDB_ANN_DATASET
#define PDB_ANN_DATASET

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <fstream>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"

class STPDataset : torch::data::Dataset<STPDataset>
{
private:
  PDBs _pdb_s;
  std::vector<json> _dataset;
  int64_t _size;
  int _permutation_size;
  int _h_max;
  std::tuple<int, int> _dimension;

public:
  explicit STPDataset(std::string path);
  static void generateRandom(std::string path);
  json toJSON();
  void save(std::string path);
  torch::data::Example<> get(size_t index) override;
  torch::optional<size_t> size() const override;
};

#endif