#ifndef PDB_ANN_PDB
#define PDB_ANN_PDB

#include <iostream>
#include <map>
#include <string>

#include "stp/stp.hpp"

class PDB
{
private:
  std::map<unsigned int, int> table;

public:
  PDB();
  void generatePDB(STP goalSTP, std::string pattern);
};

#endif