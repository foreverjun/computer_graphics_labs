//
// Created by matvey on 10.11.2023.
//
#ifndef MAZE_H
#define MAZE_H
#include <vector>

#include "DSU.h"
using namespace std;


class maze {
    private:
    int size =0;
    DSU dsu;

    int get_index(int i, int j);

    // 0 - wall
    // 1 - free
    void initialize_dsu(vector<vector<bool>>& maze);
public: // constructor that takes a vector of vectors of bools as the maze representation
    explicit maze( vector<vector<bool>>& maze);
    // method that takes two pairs of ints as the coordinates of two cells and returns true if they are connected
    bool query(pair<int, int> cell1, pair<int, int> cell2);

    // other methods and fields

};



#endif //MAZE_H
