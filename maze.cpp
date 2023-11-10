//
// Created by matvey on 10.11.2023.
//

#include <vector>
#include "maze.h"
#include "DSU.h"

using namespace std;


int maze::get_index(int i, int j) {
        return i * size + j;
    }

    // 0 - wall
    // 1 - free
    void maze::initialize_dsu(vector<vector<bool>>& maze) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i + 1 < size && maze[i][j] && maze[i + 1][j]) {
                    dsu.union_sets(get_index(i, j), get_index(i + 1, j));
                }
                if (j + 1 < size && maze[i][j] && maze[i][j + 1]) {
                    dsu.union_sets(get_index(i, j), get_index(i, j + 1));
                }
            }
        }
    }


    maze::maze (vector<vector<bool>>& maze):size(maze.size()), dsu(DSU(size)){
        initialize_dsu(maze);
    }

    bool maze::query(pair<int, int> start, pair<int, int> finish) {
        return dsu.find(get_index(start.first, start.second)) == dsu.find(get_index(finish.first, finish.second));
    }

