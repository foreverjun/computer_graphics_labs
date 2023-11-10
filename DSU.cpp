//
// Created by matvey on 10.11.2023.
//
#include "DSU.h"
#include <vector>
using namespace std;


DSU::DSU(int n) {
    parent.resize(n * n);
    rank.resize(n * n);
    for (int i = 0; i < n * n; i++) {
        parent[i] = i;
    }
}

int DSU::find(const int k) {
    if (k == parent[k]) {
        return k;
    }
    return find(parent[k]);
}

void DSU::union_sets(const int k, const int v) {
    const int r1 = find(k);
    const int r2 = find(v);
    if (rank[r1] < rank[r2]) {
        parent[r1] = r2;
    }
    else {
        parent[r2] = r1;
        if (rank[r1] == rank[r2]) {
            rank[r1]++;
        }
    }
}
