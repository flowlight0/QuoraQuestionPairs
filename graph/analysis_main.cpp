#include <string>
#include <fstream>
#include <queue>
#include <cassert>
#include <set>

#include "common.hpp"
#include "union_find.hpp"
#include "bridge_decomposition.hpp"
using namespace std;

typedef long long ll;


int readGraph(ifstream &ifs, vector<pair<int, int> > &es) {
  int u, v, V = 0;
  while (ifs >> u >> v) {
    V = max({u, v, V + 1});
    es.push_back(make_pair(u, v));
  }
  return V;
}

class TopKDistanceSolver {
  const size_t K;
  vector<vector<int> > G;
  BridgeDecomposition decomp;
  vector<size_t> visit_count;
  set<pair<int, int>> bridges;
public:
  TopKDistanceSolver(int V, const vector<pair<int, int> > &es, size_t K) :
    K(K), decomp(es), visit_count(V, 0)
  {
    G.clear();
    G.resize(V);
    for (const auto &e : es) {
      G[e.fst].push_back(e.snd);
      G[e.snd].push_back(e.fst);
    }
    
    bridges.clear();
    REP(i, decomp.edge_group.size()) {
      const auto &eg = decomp.edge_group[i];
      if (eg.size() == 1) {
        for (const int &ei: eg) {
          bridges.insert(es[ei]);
        }
      }
    }
  }
  
  vector<int> solve(int s, int t) {
    if (s == t) {
     return vector<int>(K, 0);
    }

    if (bridges.count(make_pair(s, t))) {
      return vector<int>(K, G.size());
    }

    
    vector<int> res;
    vector<int> visited_vs;
    queue<pair<int, int> > que;
    que.push(make_pair(s, 0));
    visit_count[s]++;
    visited_vs.push_back(s);
    
    while (!que.empty()) {
      const auto &p = que.front(); que.pop();
      const int v = p.fst;
      const int d = p.snd;
      if (res.size() == K) {
        break;
      }
      if (v == t) {
        res.push_back(d);
      }

      for (const int w : G[v]) {
        // We assume there is no edge between a queried vertex pair.
        if (s == v && t == w) continue;
        
        if (visit_count[w] < K) {
          if (visit_count[w]++ == 0) {
            visited_vs.push_back(w);
          }
          que.push(make_pair(w, d + 1));
        }
      }
    }

    while (res.size() < K) {
      res.push_back(G.size());
    }

    for (const int v : visited_vs) {
      visit_count[v] = 0;
    }

    assert(!bridges.count(make_pair(s, t)));
    return res;
  }
};


int main(int argc, char *argv[])
{
  const string edge_file = argv[1];
  const string output_file = argv[2];
  const string feature_prefix = argv[3];
  ifstream ifs(edge_file);
  vector<pair<int, int> > es;
  cerr << "Started loading a graph" << endl;
  int V = readGraph(ifs, es);
  cerr << "Finished loading a graph" << endl;
  
  const size_t K = 32;
  cerr << "Started initializing a solver" << endl;
  TopKDistanceSolver solver(V, es, K);
  cerr << "Finished initializing a solver" << endl;
  ofstream ofs(output_file);
  for (size_t i = 0; i < K; i++) {
    const string feature_name = feature_prefix + "." + to_string(i);
    ofs << feature_name << (i + 1 == K ? '\n' : ',');
    
  }

  int count = 0;
  for (const auto &e: es) {
    if (++count % 10000 == 0) {
      cerr << "Processing " << to_string(count) << "-th edge" << endl;
    }
    const int s = e.fst;
    const int t = e.snd;
    const vector<int> k_distance = solver.solve(s, t);
    for (size_t i = 0; i < k_distance.size(); i++) {
      ofs << k_distance[i] << (i + 1 == K ? '\n' : ',');
    }
  }
  return 0;
}
