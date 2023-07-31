#include "stp.hpp"

STP::STP()
{
  _width = 0;
  _height = 0;

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
  _blank = 0;
}

STP::STP(int width, int height)
{
  _width = width;
  _height = height;

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
  _blank = 0;
}

STP::STP(std::tuple<int, int> dimension)
{
  _width = std::get<0>(dimension);
  _height = std::get<1>(dimension);

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
  _blank = 0;
}

STP::STP(const STP &_stp)
{
  _width = _stp._width;
  _height = _stp._height;

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
  _blank = 0;

  initState(_stp._state);
}

int STP::size()
{
  return _width * _height;
}

int STP::blank()
{
  return _blank;
}

int STP::getTile(int tile)
{
  int x_t = static_cast<int>(tile / _width);
  int y_t = tile % _width;

  return _state[x_t][y_t].item<int>();
}

std::tuple<int, int> STP::dimension()
{
  return std::make_tuple(_width, _height);
}

void STP::initGoal()
{
  int i = 0;
  int j = 0;
  for (int n = 0; n < size(); n++)
  {
    _state[i][j] = n;
    j++;
    j %= _width;
    if (j == 0)
      i++;
  }

  _blank = 0;
}

void STP::initState(std::vector<int> state)
{
  int i = 0;
  int j = 0;
  for (int n = 0; n < size(); n++)
  {
    _state[i][j] = state[n];
    j++;
    j %= _width;
    if (j == 0)
      i++;

    if (state[n] == 0)
      _blank = n;
  }
}

void STP::initState(torch::Tensor state)
{
  _state.copy_(state);
}

torch::Tensor STP::getState()
{
  return _state;
}

void STP::toAbstract(std::vector<int> pattern)
{
  auto state = std::vector<int>(size(), -1);
  torch::Tensor f_state = getFlattenState(true);

  for (int i = 0; i < pattern.size(); i++)
  {
    state[f_state[pattern[i]].item<int>()] = pattern[i];
  }

  initState(state);
}

torch::Tensor STP::getFlattenState(bool dual)
{
  torch::Tensor f_state = _state.flatten();

  if (dual)
  {
    auto options = torch::TensorOptions().dtype(torch::kInt);
    torch::Tensor pos = torch::full({size()}, -1, options);

    for (int i = 0; i < size(); i++)
    {
      if (f_state[i].item<int>() < 0)
        continue;

      pos[f_state[i]] = i;
    }

    return pos;
  }
  return f_state;
}

torch::Tensor STP::getFlattenState(std::vector<int> pattern)
{
  torch::Tensor f_state = getFlattenState(true);

  auto options = torch::TensorOptions().dtype(torch::kInt);
  torch::Tensor res = torch::zeros({(int)pattern.size()}, options);

  for (int i = 0; i < pattern.size(); i++)
  {
    res[i] = f_state[pattern[i]];
  }

  return res;
}

int64_t STP::hashState(std::optional<torch::Tensor> pi_optional)
{
  torch::Tensor pi = pi_optional.value_or(getFlattenState());

  int pi_size = pi.size(0);

  int64_t rank = 0;
  int k = size();

  for (int i = 0; i < pi_size; i++)
  {
    int number = pi[i].item<int>();
    int64_t factorial = number;
    for (int j = k - 1; j > size() - pi_size; j--)
    {
      factorial *= j;
    }
    rank += factorial;
    k--;

    for (int j = i + 1; j < pi_size; j++)
    {
      if (pi[j].item<int>() > number)
        pi[j] -= 1;
    }
  }

  return rank;
}

int64_t STP::hashState(std::vector<int> pattern, std::optional<std::vector<int>> pi_helper)
{
  torch::Tensor pi = getFlattenState(pattern);

  if (pi_helper.has_value())
  {
    std::vector<int> h_pi = pi_helper.value();

    int j = 0;
    for (int i = 0; i < h_pi.size(); i++)
    {
      for (; j < pi.size(0); j++)
      {
        if (pi[j].item<int>() == -1)
        {
          pi[j] = h_pi[i];
          j++;
          break;
        }
      }
    }
  }

  return hashState(pi);
}

bool STP::moveBlank(STPAction action)
{
  auto res = nextState(action, _blank);
  if (res.has_value())
  {
    initState(std::get<torch::Tensor>(res.value()));
    _blank = std::get<int>(res.value());
  }
  return false;
}

std::optional<std::tuple<torch::Tensor, int>> STP::nextState(STPAction action, int tile)
{
  int x_t = static_cast<int>(tile / _width);
  int y_t = tile % _width;
  int v_t = getTile(tile);

  torch::Tensor n_state = 1 * getState();

  switch (action)
  {
  case UP:
    if (x_t > 0)
    {
      int tmp = n_state[x_t - 1][y_t].item<int>();
      n_state[x_t - 1][y_t] = v_t;
      n_state[x_t][y_t] = tmp;
      tile = (x_t - 1) * _width + y_t;
    }
    else
      return std::nullopt;
    break;
  case RIGHT:
    if (y_t < _width - 1)
    {
      int tmp = n_state[x_t][y_t + 1].item<int>();
      n_state[x_t][y_t + 1] = v_t;
      n_state[x_t][y_t] = tmp;
      tile = (x_t)*_width + (y_t + 1);
    }
    else
      return std::nullopt;
    break;
  case DOWN:
    if (x_t < _height - 1)
    {
      int tmp = n_state[x_t + 1][y_t].item<int>();
      n_state[x_t + 1][y_t] = v_t;
      n_state[x_t][y_t] = tmp;
      tile = (x_t + 1) * _width + y_t;
    }
    else
      return std::nullopt;
    break;
  case LEFT:
    if (y_t > 0)
    {
      int tmp = n_state[x_t][y_t - 1].item<int>();
      n_state[x_t][y_t - 1] = v_t;
      n_state[x_t][y_t] = tmp;
      tile = (x_t)*_width + (y_t - 1);
    }
    else
      return std::nullopt;
    break;

  default:
    return std::nullopt;
  }

  return std::make_tuple(n_state, tile);
}

std::vector<STPAction> STP::getActions(int tile)
{
  auto actions = std::vector<STPAction>();

  int x_t = static_cast<int>(tile / _width);
  int y_t = tile % _width;

  if (x_t > 0)
    actions.push_back(STPAction::UP);
  if (y_t < _width - 1)
    actions.push_back(STPAction::RIGHT);
  if (x_t < _height - 1)
    actions.push_back(STPAction::DOWN);
  if (y_t > 0)
    actions.push_back(STPAction::LEFT);

  return actions;
}

std::vector<std::tuple<STP, int>> STP::getSuccessors(int tile)
{
  std::vector<STPAction> actions = getActions(tile);
  auto successors = std::vector<std::tuple<STP, int>>();

  for (auto &action : actions)
  {
    auto n_state = nextState(action, tile);
    if (n_state.has_value())
    {
      STP n_stp = STP(_width, _height);
      int n_tile = std::get<int>(n_state.value());
      n_stp.initState(std::get<torch::Tensor>(n_state.value()));
      successors.push_back({n_stp, n_tile});
    }
  }

  return successors;
}

int STP::getMDHeuristic(std::optional<torch::Tensor> state_optional)
{
  torch::Tensor state = state_optional.value_or(getState());

  int h = 0;

  for (int i = 0; i < _height; i++)
  {
    for (int j = 0; j < _width; j++)
    {
      int tile = state[i][j].item<int>();
      if (tile == -1)
        continue;

      int x_t = static_cast<int>(tile / _width);
      int y_t = tile % _width;

      h += std::abs(x_t - i) + std::abs(y_t - j);
    }
  }

  return h;
}