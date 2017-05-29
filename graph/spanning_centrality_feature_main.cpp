#include "spanning_centrality.hpp"
#include "common.hpp"
#include <map>
using namespace spanning_centrality;
using namespace std;
  
int main(int argc, char *argv[])
{
  const string edge_file = argv[1];
  const string output_file = argv[2];
  const string feature_prefix = argv[3];
  ifstream ifs(edge_file);
  vector<pair<int, int> > es;
  cerr << "Started loading a graph" << endl;
  int V = readGraph(ifs, es);
  
  auto new_es = es;
  internal::ConvertToUndirectedGraph(new_es);
  cerr << "Finished loading a graph" << endl;
  
  cerr << "Started feature creation" << endl;
  const std::vector<double> centrality = EstimateEdgeCentrality(new_es, 3);
  cerr << "Finished feature creation" << endl;

  map<pair<int, int>, double>  centrality_map;
  for (size_t i = 0; i < centrality.size(); i++) {
    centrality_map[make_pair(new_es[i].fst, new_es[i].snd)] = centrality[i];
    centrality_map[make_pair(new_es[i].snd, new_es[i].fst)] = centrality[i];
  }

  for (const auto &e: es) {
    cout << centrality_map[make_pair(e.fst, e.snd)] << endl;
  }
  return 0;
}
