#include "stp.hpp"

STP::STP(unsigned int width, unsigned int height)
{
  _width = width;
  _height = height;

  _state = torch::zeros({1, _width * _height});
}

void STP::initGoal()
{
  for (int i = 0; i < _width * _height; i++)
  {
    _state[i] = i;
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

unsigned int STP::hashState()
{
  return 0;
}
