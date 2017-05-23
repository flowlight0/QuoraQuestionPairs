#ifndef BRIDGE_DECOMPOSITION_H
#define BRIDGE_DECOMPOSITION_H

#include <vector>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include "common.hpp"


class BridgeDecomposition{
  std::vector<std::vector<int> >  G;
  std::vector<int> ord;
  std::vector<int> low;
  std::vector<int> par;
  void Dfs(int v, int &times);
  
public:
  std::vector<std::vector<int> > edge_group;
  BridgeDecomposition(const std::vector<std::pair<int, int> > &es);
  BridgeDecomposition(const std::vector<std::vector<int> > &G);
  std::vector<int> GetLargestBiconnectedComponent();
};



void BridgeDecomposition::Dfs(int v, int &times){
  ord[v] = low[v] = times++;
  for (int w : G[v]){
    if (ord[w] == -1){
      par[w] = v;
      Dfs(w, times);
      low[v] = std::min(low[v], low[w]);
    } else if (w != par[v]){
      low[v] = std::min(low[v], low[w]);
    }
  }
}

BridgeDecomposition::BridgeDecomposition(const std::vector<std::pair<int, int> > &es){
  int V = 0;
  for (const auto &e : es) V = std::max(V, std::max(e.fst, e.snd) + 1);
  
  ord = std::vector<int>(V, -1);
  low = std::vector<int>(V, -1);
  par = std::vector<int>(V, -1);
  G   = std::vector<std::vector<int> >(V);
  for (const auto &e : es){
    G[e.first].push_back(e.second);
    G[e.second].push_back(e.first);
  }

  int times = 0;
  REP(v, V) if (ord[v] == -1){
    Dfs(v, times);
  }
  
  G.clear();
  G = std::vector<std::vector<int> > (V);
  
  for (size_t i = 0; i < es.size(); i++){
    std::pair<int, int> e = es[i];
    if (par[e.snd] != e.fst) std::swap(e.fst, e.snd);
    if (par[e.snd] != e.fst) continue;
    if (low[e.snd] > ord[e.fst]){
      edge_group.push_back(std::vector<int>(1, i));
    } else {
      G[e.fst].push_back(e.snd);
      G[e.snd].push_back(e.fst);
    }
  }
  
  std::vector<int> visit(V, -1);
  
  int num_cc = edge_group.size();
  REP(s, V) if (visit[s] == -1){
    
    visit[s] = num_cc;
    int visit_count = 1;
    std::queue<int> que;
    que.push(s);
    while (!que.empty()){
      int v = que.front(); que.pop();
      for (int w : G[v]){
        if (visit[w] == -1){
          visit[w] = num_cc;
          visit_count++;
          que.push(w);
        }
      }
    }
    num_cc++;
  }
  edge_group.resize(num_cc);
  REP(i, es.size()){
    std::pair<int, int> e = es[i];
    if (visit[e.fst] == visit[e.snd]) {
      edge_group[visit[e.fst]].push_back(i);
    }
  }  
}

std::vector<int> BridgeDecomposition::GetLargestBiconnectedComponent(){
  const auto &eg = this->edge_group;
  int best_id = 0;
  REP(i, eg.size()){
    if (eg[i].size() > eg[best_id].size()) best_id = i;
  }
  return eg[best_id];
}

#endif
