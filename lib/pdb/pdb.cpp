#include "pdb.hpp"

PDB::PDB(STP goalSTP, std::vector<int> pattern)
{
  int pdbSize = 1;
  for (int i = goalSTP.size(); i > goalSTP.size() - pattern.size(); i--)
  {
    pdbSize *= i;
  }

  auto options = torch::TensorOptions().dtype(torch::kInt);
  _table = torch::full({pdbSize}, -1, options);

  _goal = STP(goalSTP);
  _goal.toAbstract(pattern);
  _pattern = pattern;
}

int PDB::size()
{
  return _table.size(0);
}

void PDB::fill()
{
  std::queue<STP> frontier = std::queue<STP>();

  frontier.push(_goal);

  _table[_goal.hashState(_pattern)] = 0;

  while (!frontier.empty())
  {
    STP front = frontier.front();
    frontier.pop();

    std::vector<std::tuple<STP, int>> successors = front.getSuccessors(front.blank());

    int h_f = _table[front.hashState(_pattern)].item<int>();

    for (auto &successor : successors)
    {
      STP stp = std::get<STP>(successor);
      int cost = std::get<int>(successor);

      int idx_s = stp.hashState(_pattern);
      int h_s = h_f + cost;

      if (_table[idx_s].item<int>() == -1)
      {
        _table[idx_s] = h_s;
        frontier.push(stp);
      }
    }
  }
}
