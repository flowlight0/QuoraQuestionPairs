#ifndef ARTICULATION_POINT_DECOMPOSITION_H
#define ARTICULATION_POINT_DECOMPOSITION_H

#include <vector>
#include <stack>
#include <cstdlib>
#include <tuple>
#include <cassert>
#include <algorithm>
#include <iostream>
using namespace std;

namespace spanning_centrality {

  class ArticulationPointDecomposition {
    std::vector<std::vector<std::pair<int, int> > > G;
    std::vector<int> ord;
    std::vector<int> low;
    std::vector<int> par;
  
    void Dfs(int v, int p, int &times, std::stack<std::tuple<int, int, int> > &st) {
      ord[v] = low[v] = ++times;
      for (const auto &e : G[v]){
        int w = e.first;
        int i = e.second;
        if (ord[w] < ord[v] && w != p) {
          st.push(std::make_tuple(v, w, i));
        }

        if (ord[w] == -1){
          Dfs(w, v, times, st);
          low[v] = std::min(low[v], low[w]);

          if ((ord[v] == 1 && ord[w] != 2) || (ord[v] != 1 && low[w] >= ord[v])){
            articulation_points.push_back(v);
          }
      
          if (low[w] >= ord[v]){
            edge_groups.push_back(std::vector<int>());
            while (true){
              int a, b, j;
              std::tie(a, b, j) = st.top(); st.pop();
              edge_groups.back().push_back(j);
              if (v == a && w == b) break;
            }
          } 
        } else {
          low[v] = std::min(low[v], ord[w]);
        }
      }
    }
  
  public:
    std::vector<int> articulation_points;
    std::vector<std::vector<int> > edge_groups;
  
    ArticulationPointDecomposition(const std::vector<std::pair<int, int> > &es) {
      assert(es.size() < 1ull << 31);
      int V = 0;
      for (const auto &e : es) {
        assert(e.first != e.second);
        V = std::max(V, std::max(e.first, e.second) + 1);
      }
  
      ord = std::vector<int>(V, -1);
      low = std::vector<int>(V, -1);
      par = std::vector<int>(V, -1);
      G   = std::vector<std::vector<std::pair<int, int> > >(V);
      for (size_t i = 0; i < es.size(); i++){
        G[es[i].first].emplace_back(es[i].second, i);
        G[es[i].second].emplace_back(es[i].first, i);
      }

      for (int v = 0; v < V; v++) {
        if (ord[v] == -1){
          int times = 0;
          std::stack<std::tuple<int, int, int> > st;
          Dfs(v, -1, times, st);
        }
      }
      std::sort(articulation_points.begin(), articulation_points.end());
      articulation_points.erase(unique(articulation_points.begin(), articulation_points.end()), articulation_points.end());
    }
  };

}

#endif /* ARTICULATION_POINT_DECOMPOSITION_H */
