# PDB-ANN

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

## Overview

The PDB-ANN project aims to implement one of the methodologies mentioned in the research paper - [Optimal Search with Neural Networks: Challenges and Approaches](https://webdocs.cs.ualberta.ca/~nathanst/papers/li2022optimalnn.pdf).

## Requirements

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of [LibTorch](https://pytorch.org/cppdocs/installing.html). Detailed instructions related to the installation of LibTorch can be found on the linked page.

- This project extensively utilizes the [nlohmann/json](https://github.com/nlohmann/json) library to facilitate the conversion of models into JSON format. You can download the latest version of this library via [JSON for Modern C++](https://github.com/nlohmann/json/releases/latest).

- [cxxopts](https://github.com/jarro2783/cxxopts) is a lightweight C++ library used for parsing command-line arguments in this project. To get the latest version of cxxopts, visit the [releases page](https://github.com/jarro2783/cxxopts/releases/latest).

> **Note**: This project includes the `nlohmann/json` and `cxxopts` libraries, so there's **no need to download them manually**.

## Installation

To build PDB-ANN, follow these steps:

Linux:

```bash
git clone https://github.com/mgh5225/PDB-ANN.git
cd PDB-ANN
bash build.sh
```

> Please be aware that you'll need to update the absolute path to LibTorch in the `CMakeLists.txt` file.

## Usage

Once the build process is complete, you can initiate the program using the following commands:

```shell
Pattern Database + ANN
Usage:
  main [OPTION...]

  -h, --help    Print usage
      --pdb     Create PDBs
      --create  Create random database based on created PDBs
  -t, --train   Train ANN
  -r, --run     Run ANN

 run options:
  -s, --state arg    State for ANN
  -p, --pattern arg  Pattern for ANN
  -d, --dim arg      Dimension for ANN
  -q arg             List of q for ANN
```

## Example

the output of `./build/main -r --pattern 1,2,3,4 --state 8,7,3,0,4,5,1,6,2 --dim 3,3 -q 1e-2,1e-4,1` is as follows:

```shell
 5.5599e-12  5.7679e-04  9.9942e-01  4.1235e-09  8.6619e-09  2.9172e-10
[ CPUFloatType{1,6} ]
q       h
0.01    2
0.0001  1
1       2
```

## License

This project uses the following license: [MIT license](https://github.com/mgh5225/PDB-ANN/blob/main/LICENSE)
