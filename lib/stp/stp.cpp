#include "stp.hpp"

STP::STP(unsigned int width, unsigned int height)
{
  _width = width;
  _height = height;

  auto options = torch::TensorOptions().dtype(torch::kInt);

  _state = torch::zeros({_height, _width}, options);
}

void STP::initGoal()
{
  int i = 0;
  int j = 0;
  for (int n = 0; n < _width * _height; n++)
  {
    _state[i][j] = n;
    j++;
    j %= _width;
    if (j == 0)
      i++;
  }
}

void STP::initState(int *state)
{
  int i = 0;
  int j = 0;
  for (int n = 0; n < _width * _height; n++)
  {
    _state[i][j] = state[n];
    j++;
    j %= _width;
    if (j == 0)
      i++;
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

torch::Tensor STP::getFlattenState()
{
  return _state.flatten();
}

int STP::hashState()
{
  torch::Tensor pi = getFlattenState();

  auto options = torch::TensorOptions().dtype(torch::kInt);
  torch::Tensor pi_1 = torch::zeros({_width * _height}, options);

  for (int i = 0; i < _width * _height; i++)
  {
    pi_1[pi[i]] = i;
  }

  int wh = (int)_width * _height;
  torch::Scalar n = torch::Scalar(wh);

  return rank(n, pi, pi_1);
}

int STP::rank(torch::Scalar n, torch::Tensor pi, torch::Tensor pi_1)
{
  if (n.equal(1))
    return 0;

  int s = pi[n.toInt() - 1].item<int>();

  int a = pi[n.toInt() - 1].item<int>();
  int b = pi[pi_1[n.toInt() - 1]].item<int>();

  pi[n.toInt() - 1] = b;
  pi[pi_1[n.toInt() - 1]] = a;

  a = pi_1[s].item<int>();
  b = pi_1[n.toInt() - 1].item<int>();

  pi_1[s] = b;
  pi_1[n.toInt() - 1] = a;

  return s + n.toInt() * rank(n.toInt() - 1, pi, pi_1);
}
