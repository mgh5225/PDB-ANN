#include "dataset.hpp"

STPDataset::STPDataset(std::string path)
{
  std::ifstream f(path);

  json data = json::parse(f);

  f.close();

  _pdb_s = PDBs();

  _size = data["size"];
  _dataset = data["dataset"];
  _permutation_size = data["permutation_size"];

  std::vector<json> j_pdb_s = data["pdb_s"];

  for (auto &j_pdb : j_pdb_s)
  {
    _pdb_s.push_back(PDB::fromJSON(j_pdb));
  }
}

void STPDataset::generateRandom(std::string path)
{
  std::ifstream f(path);

  json data = json::parse(f);

  f.close();

  auto pdb_s = PDBs();

  int64_t size = data["size"];
  int permutation_size = data["permutation_size"];
  std::vector<json> j_pdb_s = data["pdb_s"];

  for (auto &j_pdb : j_pdb_s)
  {
    std::vector<int> pattern = j_pdb["pattern"];
    std::tuple<int, int> goal_dimension = j_pdb["goal_dimension"];

    auto goalSTP = STP(goal_dimension);
    goalSTP.initGoal();

    auto pdb = PDB(goalSTP, pattern);
    pdb_s.push_back(pdb);
  }

  j_pdb_s.resize(0);

  for (auto &pdb : pdb_s)
  {
    pdb.fill();
    j_pdb_s.push_back(pdb.toJSON());
  }

  data["pdb_s"] = j_pdb_s;

  auto permutation = std::vector<int>(permutation_size);

  auto dataset = std::vector<json>();

  for (int64_t i = -1; i < size - 1; i++)
  {
    for (int j = 0; j < permutation_size; j++)
    {
      permutation[j] = j;
    }

    if (i >= 0)
    {
      int n = permutation_size;
      int64_t r = i;
      while (n > 0)
      {
        int k = r % n;
        int tmp = permutation[n - 1];
        permutation[n - 1] = permutation[k];
        permutation[k] = tmp;
        r = static_cast<int>(r / n);
        n--;
      }
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
  }

  data["dataset"] = dataset;

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
  data["permutation_size"] = _permutation_size;

  return data;
}

void STPDataset::save(std::string path)
{
  json data = toJSON();

  std::ofstream f(path);

  f << data;

  f.close();
}

torch::data::Example<> STPDataset::get(size_t index)
{
  return torch::data::Example<>();
}

torch::optional<size_t> STPDataset::size() const
{
  return _size;
}
