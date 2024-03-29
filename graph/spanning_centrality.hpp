#ifndef SPANNING_CENTRALITY_H
#define SPANNING_CENTRALITY_H

#include "articulation_point_decomposition.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <random>
#include <queue>
#include <chrono>

namespace spanning_centrality {
  typedef std::pair<int, int> pair_int;
  
  namespace internal {
    
    struct Edge {
      int src;
      int dst;
      size_t edge_id;
      double centrality;
      Edge(int src_, int dst_, size_t edge_id_)
        : src(src_), dst(dst_), edge_id(edge_id_), centrality(0) {}
    };
    
    
    std::vector<std::vector<pair_int> > BuildCompressedGraph(std::vector<internal::Edge> &es){
      size_t V = 0;
      std::unordered_map<int, int> vertex2id;
      for (auto &e : es){
        if (vertex2id.count(e.src) == 0) vertex2id[e.src] = V++;
        if (vertex2id.count(e.dst) == 0) vertex2id[e.dst] = V++;
        e.src = vertex2id[e.src];
        e.dst = vertex2id[e.dst];
      }

      std::vector<std::vector<pair_int> > g(V);
      for (size_t i = 0; i < es.size(); i++){
        g[es[i].src].emplace_back(es[i].dst, i);
        g[es[i].dst].emplace_back(es[i].src, i);
      }
      return g;
    }

    
    bool ReadGraph(const std::string &graph_file, std::vector<pair_int> &es){
      es.clear();
  
      std::ifstream ifs(graph_file);
      if (!ifs.good()){
        std::cerr << "Error: open graph_file." << std::endl;
        return false;
      }
    
      for (int u, v; ifs >> u >> v;) {
        if (u != v) es.emplace_back(u, v);
      }
      ifs.close();
      return true;
    }


    void ConvertToUndirectedGraph(std::vector<pair_int> &es){
      // remove self loop.
      size_t m = 0;
      for (size_t i = 0; i < es.size(); i++){
        if (es[i].first != es[i].second) es[m++] = es[i];
      }
      es.resize(m);
    
      // remove redundant edges.
      for (auto &e : es){
        if (e.first > e.second) std::swap(e.first, e.second);
      }
      std::sort(es.begin(), es.end());
      es.erase(unique(es.begin(), es.end()), es.end());
    }
    

    std::vector<int> DistanceOrdering(const std::vector<std::vector<pair_int> > &g){
      int root = 0;
      for (size_t v = 0; v < g.size(); v++) {
        if (g[v].size() > g[root].size()) root = v;
      }
      
      std::queue<int>  que;
      std::vector<int> dist(g.size(), std::numeric_limits<int>::max());
      que.push(root);
      dist[root] = 0;
      while (!que.empty()){
        int v = que.front(); que.pop();
        for (const auto &p : g[v]){
          int w = p.first;
          if (dist[w] > dist[v] + 1){
            dist[w] = dist[v] + 1;
            que.push(w);
          }
        }
      }
    
      std::vector<pair_int> order;
      for (size_t v = 0; v < g.size(); v++) {
        order.push_back(std::make_pair(dist[v], v));
      }
      std::sort(order.begin(), order.end());
      std::vector<int> res;
      for (size_t v = 0; v < g.size(); v++) {
        res.push_back(order[v].second);
      }
      assert(res[0] == root);
      return res;
    }
  }

  
  class SpanningCentrality {
  private:
    std::vector<double> edge_centrality;
    std::vector<double> vertex_centrality;
    std::vector<double> aggregated_centrality;
    std::vector<std::pair<int, int> > original_edges;
    std::mt19937  mt;

  public:

    SpanningCentrality() {
      std::random_device rd;
      this->mt = std::mt19937(rd());
    }
      
    inline double GetEdgeCentrality(size_t edge_id) const {
      return edge_centrality.at(edge_id);
    }
      
    inline double GetVertexCentrality(size_t vertex_id) const {
      return vertex_centrality.at(vertex_id);
    }

    inline double GetAggregatedCentrality(size_t vertex_id) const {
      return aggregated_centrality.at(vertex_id);
    }

    inline size_t GetNumVertices() const {
      return vertex_centrality.size();
    }

    inline size_t GetNumEdges() const {
      return edge_centrality.size();
    }

    inline std::pair<int, int> GetEdge(size_t edge_id) const {
      return original_edges.at(edge_id);
    }

    bool Construct(const std::string &graph_file, int num_samples) {
      std::vector<std::pair<int, int> > es;
      if (internal::ReadGraph(graph_file, es)) {
        internal::ConvertToUndirectedGraph(es);
        return Construct(es, num_samples);
      } else {
        return false;
      }
    }

    bool Construct(const std::vector<std::pair<int, int> > &es, int num_samples) {
      this->original_edges = es;
      
      int V = 0;
      for (const auto &e : es){
        V = std::max({V, e.first + 1, e.second + 1});
      }

        
      this->edge_centrality = std::vector<double>(es.size());
      this->vertex_centrality = std::vector<double>(V);
      this->aggregated_centrality = std::vector<double>(V);
      
      
      cerr << "Started articulation point decomposition" << endl;
      auto start_time = chrono::system_clock::now();
      const ArticulationPointDecomposition decomposition = ArticulationPointDecomposition(es);
      cerr << "Finished articulation point decomposition: " << chrono::duration_cast<std::chrono::seconds>(chrono::system_clock::now() - start_time).count() << " [s]" << endl;

      const std::vector<std::vector<int> > edge_groups = decomposition.edge_groups;
      const std::vector<int> articulation_points = decomposition.articulation_points;
      cerr << "Number of bi-connected components: " << edge_groups.size() << endl;
      cerr << "NUmber of samples: " << num_samples << endl;

      // process each bi-connected component one by one
      int iter = 0;
      std::vector<int8_t> visit(V, false);
      std::vector<int>  next_edges(V, -1);
      std::vector<int>  degree(V);
      
      for (const auto &edge_group : edge_groups) {
        if (edge_group.empty()) continue;
        if (iter++ % 10000 == 0) {
          cerr << "Started processing " << iter << "-th bi-connected component" << endl;
        }
        
        // build a subgraph from an edge group and shuffle vertices on distances.
        std::vector<internal::Edge> ccomp_es;
        for (int edge_id : edge_group){
          ccomp_es.emplace_back(es[edge_id].first, es[edge_id].second, edge_id);
        }
        std::vector<std::vector<pair_int> > g(BuildCompressedGraph(ccomp_es));
        std::vector<int>  order = internal::DistanceOrdering(g);
        

        // 簡単のため全体の頂点数分の領域を確保している.二重連結成分が多いと効率が悪くなる.
        for (int trial = 0; trial < num_samples; trial++) {
          SampleSpanningTree(next_edges, visit, g, order);
            
          for (size_t s = 1; s < g.size(); s++) {
            int v = order[s];
            int e = ccomp_es[g[v][next_edges[v]].second].edge_id;

            this->edge_centrality[e]+= 1.0 / num_samples;

            if (++degree[es[e].first] == 2){
              this->vertex_centrality[es[e].first] += 1.0 / num_samples;
            }
            if (++degree[es[e].second] == 2){
              this->vertex_centrality[es[e].second] += 1.0 / num_samples;
            }
            
            this->aggregated_centrality[es[e].first] += 1.0 / num_samples;
            this->aggregated_centrality[es[e].second] += 1.0 / num_samples;
          }

          for (const auto &e: ccomp_es) {
            visit[e.src] = visit[e.dst] = false;
            degree[e.src] = degree[e.dst] = 0;
          }
        }
      }

      for (int v : articulation_points) {
        vertex_centrality[v] = 1.0;
      }

      for (int v = 0; v < V; v++) {
        assert(vertex_centrality[v] < 1 + 1e-9);
      }
      return true;
    }

  private:
    void SampleSpanningTree(std::vector<int > &next_edges,
                            std::vector<int8_t> &visit,
                            const std::vector<std::vector<pair_int> > &g,
                            const std::vector<int> &order)
    {
      visit[order[0]] = true;
        
      for (size_t s = 1; s < g.size(); s++) {
        if (visit[order[s]]) continue;
          
        int u = order[s];
        while (!visit[u]){
          int next = mt() % g[u].size();
          next_edges[u] = next;
          u = g[u][next].first;
        }
        
        u = order[s];
        while (!visit[u]){
          visit[u] = true;
          u = g[u][next_edges[u]].first;
        }
      }
    }
  };
  
  
  // helper functions
  std::vector<double> EstimateEdgeCentrality(const std::vector<pair_int> &es, int num_samples){
    SpanningCentrality spanning_centrality;
    spanning_centrality.Construct(es, num_samples);

    size_t E = spanning_centrality.GetNumEdges();
    std::vector<double> centrality(E);
    for (size_t e = 0; e < E; e++) {
      centrality[e] = spanning_centrality.GetEdgeCentrality(e);
    }
    return centrality;
  }
  
  
  std::vector<double> EstimateVertexCentrality(const std::vector<pair_int> &es, int num_samples){
    SpanningCentrality spanning_centrality;
    spanning_centrality.Construct(es, num_samples);

    size_t V = spanning_centrality.GetNumVertices();
    std::vector<double> centrality(V);
    for (size_t v = 0; v < V; v++) {
      centrality[v] = spanning_centrality.GetVertexCentrality(v);
    }
    return centrality;
  }
  

  std::vector<double> EstimateAggregatedCentrality(const std::vector<pair_int> &es, int num_samples){
    SpanningCentrality spanning_centrality;
    spanning_centrality.Construct(es, num_samples);

    size_t V = spanning_centrality.GetNumVertices();
    std::vector<double> centrality(V);
    for (size_t v = 0; v < V; v++) {
      centrality[v] = spanning_centrality.GetAggregatedCentrality(v);
    }
    return centrality;
  }
}

#endif /* SPANNING_CENTRALITY_H */
