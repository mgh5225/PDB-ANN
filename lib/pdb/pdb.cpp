#include "pdb.hpp"

PDB::PDB()
{
  int64_t pdbSize = 0;

  auto options = torch::TensorOptions().dtype(torch::kInt);
  _table = torch::full({0}, -1, options);

  _goal = STP();
  _pattern = std::vector<int>();

  _states = std::vector<bool>(pdbSize, false);
}

PDB::PDB(STP goalSTP, std::vector<int> pattern)
{
  int64_t pdbSize = 1;
  int n_factor = goalSTP.size() - pattern.size();

  for (int i = goalSTP.size(); i > n_factor; i--)
  {
    pdbSize *= i;
  }

  auto options = torch::TensorOptions().dtype(torch::kInt);
  _table = torch::full({pdbSize}, -1, options);

  _goal = STP(goalSTP);
  _goal.toAbstract(pattern);
  _pattern = pattern;

  _states = std::vector<bool>(pdbSize * n_factor, false);
}

int64_t PDB::size()
{
  return _table.size(0);
}

torch::Tensor PDB::getTable()
{
  return _table;
}

STP PDB::getSTP()
{
  return _goal;
}

void PDB::fill()
{
  auto frontier = std::queue<std::tuple<STP, int>>();

  frontier.push({_goal, _goal.blank()});

  auto pattern_with_zero = std::vector<int>(_pattern);
  if (std::find(pattern_with_zero.begin(), pattern_with_zero.end(), 0) == pattern_with_zero.end())
    pattern_with_zero.push_back(0);

  _table[_goal.hashState(_pattern)] = 0;
  _states[_goal.hashState(pattern_with_zero, std::vector<int>({_goal.blank()}))] = true;

  while (!frontier.empty())
  {
    STP front = std::get<STP>(frontier.front());
    int tile = std::get<int>(frontier.front());
    frontier.pop();

    std::vector<std::tuple<STP, int>> successors = front.getSuccessors(tile);

    int h_f = _table[front.hashState(_pattern)].item<int>();

    for (auto &successor : successors)
    {
      STP n_stp = std::get<STP>(successor);
      int n_tile = std::get<int>(successor);

      int64_t idx_s_with_zero = n_stp.hashState(pattern_with_zero, std::vector<int>({n_tile}));

      if (_states[idx_s_with_zero])
        continue;

      int64_t idx_s = n_stp.hashState(_pattern);

      int h_md = front.getMDHeuristic() - n_stp.getMDHeuristic();

      int h_s = h_f + 1 + h_md;

      if (_table[idx_s].item<int>() == -1)
        _table[idx_s] = h_s;

      _states[idx_s_with_zero] = true;
      frontier.push({n_stp, n_tile});
    }
  }
}

PDB PDB::fromJSON(json data)
{
  int64_t pdbSize = data["pdbSize"];
  std::vector<int> pattern = data["pattern"];
  std::vector<int> table = data["table"];

  std::vector<int> goal = data["goal"];
  std::tuple<int, int> goal_dimension = data["goal_dimension"];

  auto goalSTP = STP(goal_dimension);
  goalSTP.initState(goal);

  auto pdb = PDB(goalSTP, pattern);

  auto options = torch::TensorOptions().dtype(torch::kInt);
  pdb._table = torch::from_blob(table.data(), {pdbSize}, options).clone();

  return pdb;
}

PDB PDB::load(std::string path)
{
  std::ifstream f(path);

  json data = json::parse(f);

  f.close();

  return fromJSON(data);
}

json PDB::toJSON()
{
  json data;

  torch::Tensor f_state = _goal.getFlattenState();

  data["pdbSize"] = size();
  data["pattern"] = _pattern;
  data["table"] = std::vector<int>(_table.data_ptr<int>(), _table.data_ptr<int>() + size());
  data["goal"] = std::vector<int>(f_state.data_ptr<int>(), f_state.data_ptr<int>() + _goal.size());
  data["goal_dimension"] = _goal.dimension();
  data["h_max"] = torch::max(_table).item<int>();

  return data;
}

void PDB::save(std::string path)
{
  json data = toJSON();

  std::ofstream f(path);

  f << data;

  f.close();
}

std::tuple<int, int> PDB::getHeuristic(std::vector<int> state)
{
  auto stp = STP(_goal.dimension());
  stp.initState(state);
  stp.toAbstract(_pattern);

  int h_md = stp.getMDHeuristic();

  return std::tuple<int, int>({_table[stp.hashState(_pattern)].item<int>(), h_md});
}

std::vector<int> PDB::getPattern()
{
  return _pattern;
}
