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
