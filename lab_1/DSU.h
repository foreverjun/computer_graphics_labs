//
// Created by matvey on 10.11.2023.
//

#ifndef DSU_H
#define DSU_H
#pragma once
#include <vector>
using namespace std;
class DSU {
private:
    vector<int> parent{};
    vector<int> rank{};
public:

    void union_sets(int k, int v);
    explicit DSU (int n);
    int find (int k);


};


#endif //DSU_H
