#include <iostream>
#include <vector>
#include "maze.h"
#include <cassert>
#include <time.h>

using namespace std;

void test1() {
    vector<vector<bool>> maze1 = {
        {true, true, true, true, true},
        {true, false, false, false, true},
        {true, true, true, true, true},
        {true, false, false, false, true},
        {true, true, true, true, true}
    };
    vector<vector<bool>> maze2 = {
        {true, true, true, true, true},
        {true, false, false, false, true},
        {true, true, true, false, true},
        {true, false, false, false, true},
        {true, true, true, true, true}
    };
    vector<vector<bool>> maze3 = {
        {true, true, false, true, true},
        {true, false, false, false, true},
        {true, true, true, false, true},
        {true, false, false, false, true},
        {true, true, true, false, true}
    };

    maze solve1 = maze(maze1);
    maze solve2 = maze(maze2);
    maze solve3 = maze(maze3);
    assert( solve1.query(pair<int,int> {0,0}, pair<int,int> {4,4}));
    assert( solve1.query(pair<int,int> {0,0}, pair<int,int> {3,3})== false);
    assert( solve2.query(pair<int,int> {0,0}, pair<int,int> {4,4}));
    assert( solve3.query(pair<int,int> {0,0}, pair<int,int> {4,4}) == false);
}


void test2(){
vector<vector<bool>> maze4 = {
  {true, true, true, true, true, true, true, true, true},
  {true, false, false, false, false, false, false, false, true},
  {true, true, true, false, true, true, true, false, true},
  {true, false, false, false, false, false, false, false, false},
  {true, false, true, true, true, true, true, false, true},
  {true, false, false, false, false, false, false, false, true},
  {true, true, true, true, true, true, true, false, true},
  {false, false, false, false, false, false, false, false, true},
  {true, true, true, true, true, true, true, true, true}
};
maze solve4 = maze(maze4);
assert( solve4.query(pair<int,int> {0,0}, pair<int,int> {8,8}) == false);
assert( solve4.query(pair<int,int> {0,0}, pair<int,int> {2,8}));
assert( solve4.query(pair<int,int> {8,0}, pair<int,int> {0,8}) == false);
}

maze* largest_maze(int n) {
    clock_t t0 = clock();
    vector out (n+1, vector(n+1,true));
    for (int i = 1;i<n;i+=4) {
        for (int j = 0;j<n-1;j++) {
            out[i][j] = false;
        }
    }
    for (int i = 3;i<n;i+=4) {
        for (int j = 1;j<n;j++) {
            out[i][j] = false;
        }
    }
    clock_t t1 = clock();
    clock_t dt = t1 - t0;
    double time_seconds = (double)dt / CLOCKS_PER_SEC;
    cout << "Время заполнения вектора: " << time_seconds << " секунд" << endl;
    t0 = clock();
    maze* point = new maze(out);
    t1 = clock();
    dt = t1 - t0;
    time_seconds = (double)dt / CLOCKS_PER_SEC;
    cout << "Время вычисления DSU: " << time_seconds << " секунд" << endl;
    return point;
}
bool generator()
{
    int g = std::rand();
    return (g % 2); // 1 is converted to true and 0 as false
}

maze* random_largest_maze(int n) {
    clock_t t0 = clock();
    vector out (n, vector(n,true));
    for (int i = 1;i<n;i++) {
        for (int j = 0;j<n;j++) {
            out[i][j] = generator();
        }
    }
    clock_t t1 = clock();
    clock_t dt = t1 - t0;
    double time_seconds = (double)dt / CLOCKS_PER_SEC;
    cout << "Время заполнения вектора: " << time_seconds << " секунд" << endl;
    t0 = clock();
    maze* point = new maze(out);
    t1 = clock();
    dt = t1 - t0;
    time_seconds = (double)dt / CLOCKS_PER_SEC;
    cout << "Время вычисления DSU: " << time_seconds << " секунд" << endl;
    return point;
}

void test3() {
    maze* m = largest_maze(40000);
    clock_t t0 = clock();
    assert(m->query(pair{0,0}, pair{10000,10000}));
    clock_t t1 = clock();
    clock_t dt = t1 - t0;
    double time_seconds = (double)dt / CLOCKS_PER_SEC;
    cout << "Время выполнение запроса: " << time_seconds << " секунд" << endl;
    delete m;

}

void test4() {
    maze* m = random_largest_maze(10001);
    clock_t t0 = clock();
    m->query(pair{0,0}, pair{10000,10000});
    clock_t t1 = clock();
    clock_t dt = t1 - t0;
    double time_seconds = (double)dt / CLOCKS_PER_SEC;
    cout << "Время выполнение запроса: " << time_seconds << " секунд" << endl;
    delete m;

}

int main() {
    test1();
    test2();
    test3();
    test4();
}

