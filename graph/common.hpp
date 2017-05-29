#ifndef COMMON_H
#define COMMON_H

#define REP2(i, m, n) for(int i = (int)(m); i < (int)(n); i++)
#define REP(i, n) REP2(i, 0, n)
#define ALL(c) (c).begin(), (c).end()
#define fst first
#define snd second
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>


template <typename S, typename T> std::ostream &operator<<(std::ostream &out, const std::pair<S, T> &p) {
  out << "(" << p.first << ", " << p.second << ")";
  return out;
}

template <typename T> std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << "[";
  REP(i, v.size()){
    if (i > 0) out << ", ";
    out << v[i];
  }
  out << "]";
  return out;
}


int readGraph(std::ifstream &ifs, std::vector<std::pair<int, int> > &es) {
  int u, v, V = 0;
  while (ifs >> u >> v) {
    V = std::max({u, v, V + 1});
    es.push_back(std::make_pair(u, v));
  }
  return V;
}



#endif /* COMMON_H */
