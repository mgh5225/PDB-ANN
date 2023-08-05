#include "dataset.hpp"

STPDataset::STPDataset(std::string path, double random_split)
{
  std::ifstream f(path);

  json data = json::parse(f);

  f.close();

  _pdb_s = PDBs();

  _size = data["size"];
  _dataset = data["dataset"];
  _dimension = data["dimension"];
  _h_max = data["h_max"];

  std::string pdb_s_path = data["pdb_s"];

  std::ifstream f_pdb_s(pdb_s_path);
  std::vector<json> j_pdb_s = json::parse(f_pdb_s);
  f_pdb_s.close();

  for (auto &j_pdb : j_pdb_s)
  {
    _pdb_s.push_back(PDB::fromJSON(j_pdb));
  }

  std::vector<int64_t> indices(_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  if (random_split > 0 && random_split < 1)
  {
    int64_t split_point = static_cast<int64_t>(random_split * _size);

    _train_indicies = std::vector<int64_t>(indices.begin(), indices.begin() + split_point);
    _test_indicies = std::vector<int64_t>(indices.begin() + split_point, indices.end());
  }
  else
  {
    _train_indicies = indices;
    _test_indicies = std::vector<int64_t>(0, 0);
  }
}

void STPDataset::generatePDBs(json params)
{
  json settings = params["dataset"];

  std::tuple<int, int> dimension = settings["dimension"];
  std::vector<std::vector<int>> patterns = settings["patterns"];
  std::string pdb_s_path = settings["pdb_s"];

  auto pdb_s = PDBs();

  for (auto &pattern : patterns)
  {
    auto goalSTP = STP(dimension);
    goalSTP.initGoal();

    auto pdb = PDB(goalSTP, pattern);
    pdb_s.push_back(pdb);
  }

  auto j_pdb_s = std::vector<json>();

  for (auto &pdb : pdb_s)
  {
    pdb.fill();
    j_pdb_s.push_back(pdb.toJSON());
  }

  std::ofstream f_pdb_s(pdb_s_path);
  f_pdb_s << json(j_pdb_s);
  f_pdb_s.close();
}

void STPDataset::generateRandom(json params)
{
  json data;

  json settings = params["dataset"];

  std::tuple<int, int> dimension = settings["dimension"];
  std::string path = settings["path"];
  std::string pdb_s_path = settings["pdb_s"];
  int64_t size = settings["size"];

  int permutation_size = std::get<0>(dimension) * std::get<1>(dimension);
  auto pdb_s = PDBs();

  std::ifstream f_pdb_s(pdb_s_path);
  std::vector<json> j_pdb_s = json::parse(f_pdb_s);
  f_pdb_s.close();

  int h_max = 0;

  for (auto &j_pdb : j_pdb_s)
  {
    auto pdb = PDB::fromJSON(j_pdb);
    if (pdb.h_max() > h_max)
      h_max = pdb.h_max();
    pdb_s.push_back(pdb);
  }

  auto permutation = std::vector<int>(permutation_size);
  auto goal_permutation = std::vector<int>(permutation_size);

  std::iota(goal_permutation.begin(), goal_permutation.end(), 0);

  int64_t state_space = 1;

  for (int i = permutation_size; i > 1; i--)
  {
    state_space *= i;
  }

  auto dataset = std::vector<json>();

  std::srand(std::time(nullptr));
  int64_t i = std::rand() % state_space;

  for (int64_t j = 0; j < size; j++)
  {
    permutation = goal_permutation;

    int n = permutation_size;
    int64_t r = i;
    while (n > 0)
    {
      int k = r % n;
      int tmp = permutation[n - 1];
      permutation[n - 1] = permutation[k];
      permutation[k] = tmp;
      r = static_cast<int64_t>(r / n);
      n--;
    }

    auto heuristics = std::vector<json>();

    int sum_h = 0;
    int sum_md = 0;

    for (auto &pdb : pdb_s)
    {
      std::tuple<int, int> h = pdb.getHeuristic(permutation);
      int h_s = std::get<0>(h);
      int h_md = std::get<1>(h);

      heuristics.push_back({
          {"h", h_s},
          {"md", h_md},
          {"pattern", pdb.getPattern()},
      });

      sum_h += h_s;
      sum_md += h_md;
    }

    dataset.push_back({
        {"permutation", permutation},
        {"heuristics", heuristics},
        {"h", sum_h},
        {"md", sum_md},
    });

    i++;
    i %= state_space;
  }

  data["dataset"] = dataset;
  data["dimension"] = dimension;
  data["h_max"] = h_max;
  data["pdb_s"] = pdb_s_path;
  data["size"] = size;

  std::ofstream f_out(path);
  f_out << data;
  f_out.close();
}

json STPDataset::toJSON()
{
  json data;

  auto j_pdb_s = std::vector<json>();

  for (auto &pdb : _pdb_s)
  {
    j_pdb_s.push_back(pdb.toJSON());
  }

  data["pdb_s"] = j_pdb_s;
  data["size"] = _size;
  data["dataset"] = _dataset;
  data["dimension"] = _dimension;
  data["h_max"] = _h_max;

  return data;
}

void STPDataset::save(std::string path, std::string pdb_s_path)
{
  json data = toJSON();

  std::vector<json> j_pdb_s = data["pdb_s"];

  data["pdb_s"] = pdb_s_path;

  std::ofstream f(path);
  std::ofstream f_pdb_s(pdb_s_path);

  f << data;
  f_pdb_s << json(j_pdb_s);

  f.close();
  f_pdb_s.close();
}

torch::data::Example<> STPDataset::get(size_t index, int heuristic_idx)
{

  json row = _dataset[index];

  std::vector<int> f_state = row["permutation"];
  std::vector<json> heuristics = row["heuristics"];

  std::vector<int> pattern = heuristics[heuristic_idx]["pattern"];
  int h = heuristics[heuristic_idx]["h"];
  int h_max = _pdb_s[heuristic_idx].h_max();

  h /= 2;
  h_max /= 2;

  auto stp = STP(_dimension);
  stp.initState(f_state);
  stp.toAbstract(pattern);

  int width = std::get<0>(_dimension);
  int height = std::get<1>(_dimension);

  torch::Tensor dual = stp.getFlattenState(pattern);

  auto options = torch::TensorOptions().dtype(torch::kFloat);

  torch::Tensor data = torch::zeros({(int)pattern.size(), height, width}, options);

  for (int i = 0; i < pattern.size(); i++)
  {
    int tile = dual[i].item<int>();
    int x_t = static_cast<int>(tile / width);
    int y_t = tile % width;

    data[i][x_t][y_t] = 1;
  }

  torch::Tensor target = torch::zeros({h_max + 1}, options);
  target[h] = 1;

  return torch::data::Example<>({data, target});
}

std::tuple<STPDataset::STPSubset, STPDataset::STPSubset> STPDataset::splitDataset()
{
  auto dataset = std::make_shared<STPDataset>(*this);
  auto train_indicies = std::make_shared<std::vector<int64_t>>(_train_indicies);
  auto test_indicies = std::make_shared<std::vector<int64_t>>(_test_indicies);

  auto trainDataset = STPSubset(dataset, train_indicies);
  auto testDataset = STPSubset(dataset, test_indicies);

  return std::make_tuple(trainDataset, testDataset);
}

STPDataset::STPSubset::STPSubset(std::shared_ptr<STPDataset> dataset, std::shared_ptr<std::vector<int64_t>> indicies)
{
  _dataset = dataset;
  _indicies = indicies;
}

torch::data::Example<> STPDataset::STPSubset::get(size_t index)
{
  int64_t idx = _indicies->at(index);

  return _dataset->get(idx);
}

torch::optional<size_t> STPDataset::STPSubset::size() const
{
  return _indicies->size();
}