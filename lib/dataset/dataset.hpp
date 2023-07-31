#ifndef PDB_ANN_DATASET
#define PDB_ANN_DATASET

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <tuple>
#include <memory>
#include <fstream>

#include "stp/stp.hpp"
#include "pdb/pdb.hpp"

using json = nlohmann::json;

class STPDataset
{
private:
  PDBs _pdb_s;
  std::vector<json> _dataset;
  int64_t _size;
  int _h_max;
  std::tuple<int, int> _dimension;
  std::vector<int64_t> _train_indicies;
  std::vector<int64_t> _test_indicies;

public:
  STPDataset(std::string path, double random_split);
  static void generateRandom(json params);
  json toJSON();
  void save(std::string path);
  torch::data::Example<> get(size_t index);

  class STPSubset : public torch::data::Dataset<STPSubset>
  {
  private:
    std::shared_ptr<STPDataset> _dataset;
    std::shared_ptr<std::vector<int64_t>> _indicies;

  public:
    explicit STPSubset(std::shared_ptr<STPDataset> dataset, std::shared_ptr<std::vector<int64_t>> indicies);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
  };

  std::tuple<STPDataset::STPSubset, STPDataset::STPSubset> splitDataset();
};
#endif