// One cell graph in C++: grid and walls, cells, the cell graph, its minimum spanning tree, and the persistent
// factor tree materialised along it (path copying), with one root and one value per cell.
//
// The same structure as overlay.Overlay and cellindex.CellIndex(layout="path"): the same cuts, cells, edges and
// edge weights (the number of internal nodes copied), the same Kruskal order (weight, then edge index), so the
// same tree weight and the same number of nodes.  Values are accumulated along the tree with a compensated sum
// as in cellindex._Neumaier.  Build: g++ -O3 -std=c++17 -shared -fPIC -ffp-contract=off -pthread.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Neumaier {
    double s = 0.0, c = 0.0;
    void add(double x) {
        double t = s + x;
        if (std::fabs(s) >= std::fabs(x)) c += (s - t) + x;
        else c += (x - t) + s;
        s = t;
    }
    double value() const { return s + c; }
};

// open addressing, keys are stored +1 so that 0 means empty
class FlatMap {
public:
    explicit FlatMap(size_t cap = 1024) { size_t c = 16; while (c < cap) c <<= 1; k_.assign(c, 0); v_.assign(c, 0); }
    // returns true when inserted
    bool insert(uint64_t key, uint64_t value) {
        if ((n_ + 1) * 10 >= k_.size() * 6) grow();
        size_t i = slot(key);
        if (k_[i]) return false;
        k_[i] = key + 1; v_[i] = value; ++n_;
        return true;
    }
    bool find(uint64_t key, uint64_t& value) const {
        size_t i = slot(key);
        if (!k_[i]) return false;
        value = v_[i];
        return true;
    }
    template <class F> void each(F f) const {
        for (size_t i = 0; i < k_.size(); ++i) if (k_[i]) f(k_[i] - 1, v_[i]);
    }
    size_t size() const { return n_; }
private:
    size_t slot(uint64_t key) const {
        uint64_t h = (key + 1) * 0x9E3779B97F4A7C15ull;
        size_t i = (h ^ (h >> 29)) & (k_.size() - 1);
        while (k_[i] && k_[i] != key + 1) i = (i + 1) & (k_.size() - 1);
        return i;
    }
    void grow() {
        std::vector<uint64_t> ok, ov;
        ok.swap(k_); ov.swap(v_);
        k_.assign(ok.size() * 2, 0); v_.assign(ok.size() * 2, 0);
        for (size_t i = 0; i < ok.size(); ++i) if (ok[i]) { size_t j = slot(ok[i] - 1); k_[j] = ok[i]; v_[j] = ov[i]; }
    }
    std::vector<uint64_t> k_, v_;
    size_t n_ = 0;
};

struct Graph {
    int d = 0;
    int64_t K = 1, n_faces = 0, n_edges = 0, labels = 0, labelled = 0, roots = 0, width = 2;
    double tree_weight = 0.0;
    std::vector<std::vector<int64_t>> cuts;
    std::vector<int64_t> shape, strides;
    std::vector<uint32_t> face_of_cell;
    std::vector<double> face_value;
    std::vector<int32_t> face_root;
    std::vector<uint8_t> face_empty;
    std::vector<int32_t> left, right;
    std::vector<double> node_sum;   // of the leaf values under a node of the persistent tree
};

// calls fn(flat cell) for the cells at coordinate `coord` on axis k inside the item's range on the other axes
template <class F>
void face_cells(const Graph& g, const int64_t* loc, int k, int64_t coord, std::vector<int64_t>& at, F fn) {
    const int d = g.d;
    int64_t base = coord * g.strides[k];
    for (int j = 0; j < d; ++j) {
        at[j] = loc[2 * j];
        if (j != k) base += at[j] * g.strides[j];
    }
    for (;;) {
        fn(base);
        int j = d - 1;
        for (; j >= 0; --j) {
            if (j == k) continue;
            if (++at[j] < loc[2 * j + 1]) { base += g.strides[j]; break; }
            base -= (at[j] - 1 - loc[2 * j]) * g.strides[j];
            at[j] = loc[2 * j];
        }
        if (j < 0) return;
    }
}

template <class F>
void parallel_items(int64_t n, int threads, F fn) {
    std::atomic<int64_t> next{0};
    auto work = [&]() {
        for (;;) {
            int64_t s = next.fetch_add(256);
            if (s >= n) return;
            for (int64_t i = s; i < std::min(n, s + 256); ++i) fn(i);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
}

struct Change { int32_t slot, in_a, in_b; };
struct Raw { int64_t edge; int32_t slot, side, item; };

class Pft {
public:
    Pft(Graph& g, int height, const double* values) : g_(g), height_(height), values_(values) {}
    int32_t build(const std::vector<int32_t>& state) {
        std::vector<int32_t> level(g_.width, -1);
        for (size_t i = 0; i < state.size(); ++i) level[i] = encode(state[i]);
        while (level.size() > 1) {
            std::vector<int32_t> up(level.size() / 2);
            for (size_t i = 0; i < up.size(); ++i) up[i] = make(level[2 * i], level[2 * i + 1]);
            level.swap(up);
        }
        return level[0];
    }
    // changes sorted by slot, new payload in `to`
    int32_t derive(int32_t root, const std::vector<std::pair<int32_t, int32_t>>& to) {
        if (to.empty()) return root;
        return rec(root, 0, 0, 0, to.size(), to);
    }
    bool full() const { return g_.left.size() >= 2147483000ull; }
private:
    static int32_t encode(int32_t item) { return -(item + 2); }
    double total(int32_t child) const {
        if (child >= 0) return g_.node_sum[child];
        int32_t item = -child - 2;
        return item < 0 ? 0.0 : values_[item];
    }
    int32_t make(int32_t l, int32_t r) {
        const double t = total(l) + total(r);
        g_.left.push_back(l);
        g_.right.push_back(r);
        g_.node_sum.push_back(t);
        return static_cast<int32_t>(g_.left.size() - 1);
    }
    int32_t rec(int32_t node, int depth, int64_t lo, size_t i, size_t j, const std::vector<std::pair<int32_t, int32_t>>& to) {
        if (depth == height_) return encode(to[i].second);
        int64_t mid = lo + (g_.width >> (depth + 1));
        size_t m = i;
        while (m < j && to[m].first < mid) ++m;
        int32_t l = g_.left[node], r = g_.right[node];
        if (i < m) l = rec(l, depth + 1, lo, i, m, to);
        if (m < j) r = rec(r, depth + 1, mid, m, j, to);
        return make(l, r);
    }
    Graph& g_;
    int height_;
    const double* values_;
};

}  // namespace

extern "C" {

// status: 0 ok, 1 more than max_cells grid cells, 2 more persistent nodes than fit in 32 bits
void* tw_cell_graph(int d, int64_t n, const int64_t* ranges, const int64_t* slots, const double* values,
                    const int64_t* slab_counts, double offset, int threads, int64_t max_cells, int64_t* status) {
    auto* gp = new Graph();
    Graph& g = *gp;
    g.d = d;
    *status = 0;
    int64_t n_slots = 1;
    for (int64_t i = 0; i < n; ++i) n_slots = std::max(n_slots, slots[i] + 1);
    g.width = 2;
    while (g.width < n_slots) g.width *= 2;
    int height = 0;
    for (int64_t w = g.width; w > 1; w >>= 1) ++height;

    // the local grid: the items' own boundaries only
    g.cuts.resize(d);
    g.shape.resize(d);
    g.strides.resize(d);
    __int128 K128 = 1;
    for (int k = 0; k < d; ++k) {
        auto& c = g.cuts[k];
        c.reserve(2 * n);
        for (int64_t i = 0; i < n; ++i) {
            int64_t lo = ranges[(i * d + k) * 2], hi = ranges[(i * d + k) * 2 + 1];
            if (lo > 0) c.push_back(lo);
            if (hi < slab_counts[k]) c.push_back(hi);
        }
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        g.shape[k] = static_cast<int64_t>(c.size()) + 1;
        K128 *= g.shape[k];
    }
    if (K128 > max_cells || K128 >= (static_cast<__int128>(1) << 32) - 1) {
        *status = 1;
        return gp;
    }
    g.K = static_cast<int64_t>(K128);
    const int64_t K = g.K;
    int64_t stride = 1;
    for (int k = d - 1; k >= 0; --k) { g.strides[k] = stride; stride *= g.shape[k]; }
    std::vector<int64_t> local(n * d * 2);
    for (int64_t i = 0; i < n; ++i)
        for (int k = 0; k < d; ++k) {
            const auto& c = g.cuts[k];
            int64_t lo = ranges[(i * d + k) * 2], hi = ranges[(i * d + k) * 2 + 1];
            local[(i * d + k) * 2] = std::upper_bound(c.begin(), c.end(), lo) - c.begin();
            local[(i * d + k) * 2 + 1] = (std::upper_bound(c.begin(), c.end(), hi - 1) - c.begin()) + 1;
        }

    // 1. walls: which grid edges are labelled; edge id = k * K + flat(upper cell)
    const size_t words = (static_cast<size_t>(d) * K + 63) / 64 + 1;
    std::vector<std::atomic<uint64_t>> wall(words);
    for (auto& w : wall) w.store(0, std::memory_order_relaxed);
    std::atomic<int64_t> label_count{0};
    parallel_items(n, threads, [&](int64_t i) {
        thread_local std::vector<int64_t> at;
        at.resize(d);
        const int64_t* loc = &local[i * d * 2];
        int64_t cnt = 0;
        for (int k = 0; k < d; ++k) {
            auto mark = [&](int64_t cell) {
                uint64_t e = static_cast<uint64_t>(k) * K + cell;
                wall[e >> 6].fetch_or(1ull << (e & 63), std::memory_order_relaxed);
                ++cnt;
            };
            if (loc[2 * k] > 0) face_cells(g, loc, k, loc[2 * k], at, mark);
            if (loc[2 * k + 1] < g.shape[k]) face_cells(g, loc, k, loc[2 * k + 1], at, mark);
        }
        label_count.fetch_add(cnt, std::memory_order_relaxed);
    });
    g.labels = label_count.load();
    auto is_wall = [&](int k, int64_t cell) {
        uint64_t e = static_cast<uint64_t>(k) * K + cell;
        return (wall[e >> 6].load(std::memory_order_relaxed) >> (e & 63)) & 1;
    };

    // 2. cells of the graph = components of the unlabelled grid edges; the root of a component is its first
    //    grid cell, so numbering them in order of the root is the numbering of the Python overlay
    std::vector<uint32_t> uf(K);
    for (int64_t i = 0; i < K; ++i) uf[i] = static_cast<uint32_t>(i);
    auto find = [&](uint32_t x) {
        while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
        return x;
    };
    {
        std::vector<int64_t> at(d, 0);
        for (int64_t u = 0; u < K; ++u) {
            for (int k = 0; k < d; ++k) {
                if (at[k] > 0 && !is_wall(k, u)) {
                    uint32_t a = find(static_cast<uint32_t>(u)), b = find(static_cast<uint32_t>(u - g.strides[k]));
                    if (a != b) { if (a < b) uf[b] = a; else uf[a] = b; }
                }
            }
            for (int k = d - 1; k >= 0; --k) { if (++at[k] < g.shape[k]) break; at[k] = 0; }
        }
    }
    for (int64_t u = 0; u < K; ++u) uf[u] = uf[uf[u]];   // parents have smaller indices: fully compressed
    std::vector<int64_t> representative;
    for (int64_t u = 0; u < K; ++u) {
        if (uf[u] == static_cast<uint32_t>(u)) {
            uf[u] = static_cast<uint32_t>(representative.size());
            representative.push_back(u);
        } else {
            uf[u] = uf[uf[u]];
        }
    }
    g.face_of_cell.swap(uf);
    g.n_faces = static_cast<int64_t>(representative.size());
    const auto& fc = g.face_of_cell;

    // 3. the cell graph: one representative wall per pair of adjacent cells
    FlatMap pairs(1024);
    int64_t labelled = 0;
    for (int k = 0; k < d; ++k) {
        for (int64_t u = 0; u < K; ++u) {
            uint64_t e = static_cast<uint64_t>(k) * K + u;
            uint64_t word = wall[e >> 6].load(std::memory_order_relaxed);
            if (word == 0 && (e & 63) == 0) {   // skip an empty word
                u += 63;
                if (u >= K) break;
                continue;
            }
            if ((word >> (e & 63)) & 1) {
                ++labelled;
                uint64_t a = fc[u - g.strides[k]], b = fc[u];
                uint64_t key = a < b ? (a << 32 | b) : (b << 32 | a);
                pairs.insert(key, e);
            }
        }
    }
    g.labelled = labelled;
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    edges.reserve(pairs.size());
    pairs.each([&](uint64_t key, uint64_t e) { edges.emplace_back(key, e); });
    { FlatMap().size(); }
    std::sort(edges.begin(), edges.end());
    g.n_edges = static_cast<int64_t>(edges.size());
    const int64_t F = g.n_edges;
    FlatMap rep(static_cast<size_t>(F) * 2 + 16);
    std::vector<uint64_t> rep_bits(words, 0);
    for (int64_t i = 0; i < F; ++i) {
        rep.insert(edges[i].second, static_cast<uint64_t>(i));
        rep_bits[edges[i].second >> 6] |= 1ull << (edges[i].second & 63);
    }
    std::vector<std::atomic<uint64_t>>().swap(wall);

    // the change set of every edge: a second pass over the walls of the items
    std::vector<std::vector<Raw>> raws(std::max(1, threads));
    {
        std::atomic<int64_t> next{0};
        auto work = [&](int t) {
            std::vector<int64_t> at(d);
            auto& out = raws[t];
            for (;;) {
                int64_t s = next.fetch_add(256);
                if (s >= n) return;
                for (int64_t i = s; i < std::min(n, s + 256); ++i) {
                    const int64_t* loc = &local[i * d * 2];
                    for (int k = 0; k < d; ++k) {
                        for (int side = 0; side < 2; ++side) {
                            // side 1: the item is above the wall (its lower face), 0: below (its upper face)
                            int64_t coord = side ? loc[2 * k] : loc[2 * k + 1];
                            if (side ? coord <= 0 : coord >= g.shape[k]) continue;
                            face_cells(g, loc, k, coord, at, [&](int64_t cell) {
                                uint64_t e = static_cast<uint64_t>(k) * K + cell;
                                if ((rep_bits[e >> 6] >> (e & 63)) & 1) {
                                    uint64_t idx = 0;
                                    rep.find(e, idx);
                                    out.push_back(Raw{static_cast<int64_t>(idx), static_cast<int32_t>(slots[i]), side,
                                                      static_cast<int32_t>(i)});
                                }
                            });
                        }
                    }
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; ++t) pool.emplace_back(work, t);
        work(0);
        for (auto& t : pool) t.join();
    }
    std::vector<uint64_t>().swap(rep_bits);
    std::vector<Raw> raw;
    for (auto& r : raws) { raw.insert(raw.end(), r.begin(), r.end()); std::vector<Raw>().swap(r); }
    std::sort(raw.begin(), raw.end(), [](const Raw& x, const Raw& y) {
        return x.edge != y.edge ? x.edge < y.edge : x.slot < y.slot;
    });
    std::vector<int64_t> eptr(F + 1, 0);
    std::vector<Change> changes;
    changes.reserve(raw.size() / 2 + 1);
    {
        size_t r = 0;
        for (int64_t e = 0; e < F; ++e) {
            eptr[e] = static_cast<int64_t>(changes.size());
            const uint64_t eid = edges[e].second;
            const int k = static_cast<int>(eid / K);
            const int64_t upper = static_cast<int64_t>(eid % K);
            const bool below_is_a = fc[upper - g.strides[k]] < fc[upper];
            while (r < raw.size() && raw[r].edge == e) {
                const int32_t slot = raw[r].slot;
                int32_t below = -1, above = -1;
                for (; r < raw.size() && raw[r].edge == e && raw[r].slot == slot; ++r)
                    (raw[r].side ? above : below) = raw[r].item;
                changes.push_back(below_is_a ? Change{slot, below, above} : Change{slot, above, below});
            }
        }
        eptr[F] = static_cast<int64_t>(changes.size());
    }
    std::vector<Raw>().swap(raw);

    // edge weights: internal nodes copied by a path copy of the changed slots
    std::vector<int32_t> weight(F);
    {
        std::vector<int64_t> cur;
        for (int64_t e = 0; e < F; ++e) {
            cur.clear();
            for (int64_t c = eptr[e]; c < eptr[e + 1]; ++c) cur.push_back(changes[c].slot);
            int cost = 0;
            for (int64_t w = g.width; w > 1 && !cur.empty(); w /= 2) {
                for (auto& v : cur) v /= 2;
                cur.erase(std::unique(cur.begin(), cur.end()), cur.end());
                cost += static_cast<int>(cur.size());
            }
            weight[e] = cost;
        }
    }

    // 4. minimum spanning forest: Kruskal, by weight and then edge index
    std::vector<int64_t> order(F);
    for (int64_t i = 0; i < F; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return weight[x] < weight[y]; });
    std::vector<uint32_t> dsu(g.n_faces);
    for (int64_t i = 0; i < g.n_faces; ++i) dsu[i] = static_cast<uint32_t>(i);
    auto dfind = [&](uint32_t x) {
        while (dsu[x] != x) { dsu[x] = dsu[dsu[x]]; x = dsu[x]; }
        return x;
    };
    std::vector<int64_t> tree_edges;
    tree_edges.reserve(g.n_faces);
    for (int64_t i : order) {
        uint32_t a = static_cast<uint32_t>(edges[i].first >> 32), b = static_cast<uint32_t>(edges[i].first & 0xFFFFFFFFull);
        uint32_t ra = dfind(a), rb = dfind(b);
        if (ra != rb) {
            dsu[ra] = rb;
            tree_edges.push_back(i);
            g.tree_weight += weight[i];
        }
    }
    std::vector<int64_t>().swap(order);
    std::vector<uint32_t>().swap(dsu);
    std::vector<int64_t> aptr(g.n_faces + 1, 0);
    for (int64_t e : tree_edges) {
        ++aptr[(edges[e].first >> 32) + 1];
        ++aptr[(edges[e].first & 0xFFFFFFFFull) + 1];
    }
    for (int64_t i = 0; i < g.n_faces; ++i) aptr[i + 1] += aptr[i];
    std::vector<int64_t> adj(aptr[g.n_faces]);
    {
        std::vector<int64_t> pos(aptr.begin(), aptr.end() - 1);
        for (int64_t e : tree_edges) {
            adj[pos[edges[e].first >> 32]++] = e;
            adj[pos[edges[e].first & 0xFFFFFFFFull]++] = e;
        }
    }

    // 5. materialise along the tree: one root and one value per cell
    g.face_value.assign(g.n_faces, 0.0);
    g.face_root.assign(g.n_faces, -1);
    g.face_empty.assign(g.n_faces, 0);
    std::vector<uint8_t> seen(g.n_faces, 0);
    Pft pft(g, height, values);
    struct Frame { uint32_t face; int64_t next; int64_t parent_edge; Neumaier acc; int64_t active; };
    std::vector<Frame> stack;
    std::vector<std::pair<int32_t, int32_t>> to;
    auto val = [&](int32_t item) { return item < 0 ? 0.0 : values[item]; };
    std::vector<int64_t> at(d);
    for (int64_t r = 0; r < g.n_faces; ++r) {
        if (seen[r]) continue;
        ++g.roots;
        // the source: the items of the root's first grid cell
        int64_t cell = representative[r];
        for (int k = 0; k < d; ++k) at[k] = (cell / g.strides[k]) % g.shape[k];
        std::vector<int32_t> state(n_slots, -1);
        Neumaier acc;
        acc.add(offset);
        int64_t active = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t* loc = &local[i * d * 2];
            bool in = true;
            for (int k = 0; k < d && in; ++k) in = loc[2 * k] <= at[k] && at[k] < loc[2 * k + 1];
            if (in) { state[slots[i]] = static_cast<int32_t>(i); acc.add(values[i]); ++active; }
        }
        seen[r] = 1;
        g.face_root[r] = pft.build(state);
        g.face_value[r] = acc.value();
        g.face_empty[r] = active == 0;
        stack.push_back(Frame{static_cast<uint32_t>(r), aptr[r], -1, acc, active});
        while (!stack.empty()) {
            Frame& top = stack.back();
            if (top.next >= aptr[top.face + 1]) { stack.pop_back(); continue; }
            int64_t e = adj[top.next++];
            uint32_t a = static_cast<uint32_t>(edges[e].first >> 32), b = static_cast<uint32_t>(edges[e].first & 0xFFFFFFFFull);
            uint32_t v = top.face, w = v == a ? b : a;
            if (seen[w]) continue;
            seen[w] = 1;
            const bool forward = v == a;
            Neumaier cacc = top.acc;
            int64_t cact = top.active;
            to.clear();
            for (int64_t c = eptr[e]; c < eptr[e + 1]; ++c) {
                int32_t oldv = forward ? changes[c].in_a : changes[c].in_b;
                int32_t newv = forward ? changes[c].in_b : changes[c].in_a;
                cacc.add(val(newv) - val(oldv));
                cact += (newv >= 0) - (oldv >= 0);
                to.emplace_back(changes[c].slot, newv);
            }
            if (pft.full()) { *status = 2; return gp; }
            int32_t root = pft.derive(g.face_root[v], to);
            g.face_root[w] = root;
            g.face_value[w] = cacc.value();
            g.face_empty[w] = cact == 0;
            stack.push_back(Frame{w, aptr[w], e, cacc, cact});
        }
    }
    return gp;
}

void tw_cg_info(void* h, int64_t* out, double* weight) {
    auto* g = static_cast<Graph*>(h);
    out[0] = g->K; out[1] = g->n_faces; out[2] = g->n_edges; out[3] = g->labels; out[4] = g->labelled;
    out[5] = g->roots; out[6] = static_cast<int64_t>(g->left.size()); out[7] = g->width;
    *weight = g->tree_weight;
}
int64_t tw_cg_ncuts(void* h, int k) { return static_cast<int64_t>(static_cast<Graph*>(h)->cuts[k].size()); }
void tw_cg_cuts(void* h, int k, int64_t* out) {
    auto& c = static_cast<Graph*>(h)->cuts[k];
    if (!c.empty()) std::memcpy(out, c.data(), c.size() * sizeof(int64_t));
}
// each copy releases the C++ array, so that memory is not held twice
void tw_cg_take(void* h, uint32_t* face_of_cell, double* face_value, int32_t* face_root, uint8_t* face_empty,
                int32_t* left, int32_t* right, double* node_sum) {
    auto* g = static_cast<Graph*>(h);
    std::memcpy(face_of_cell, g->face_of_cell.data(), g->face_of_cell.size() * sizeof(uint32_t));
    std::vector<uint32_t>().swap(g->face_of_cell);
    std::memcpy(face_value, g->face_value.data(), g->face_value.size() * sizeof(double));
    std::memcpy(face_root, g->face_root.data(), g->face_root.size() * sizeof(int32_t));
    std::memcpy(face_empty, g->face_empty.data(), g->face_empty.size());
    std::memcpy(left, g->left.data(), g->left.size() * sizeof(int32_t));
    std::memcpy(right, g->right.data(), g->right.size() * sizeof(int32_t));
    std::memcpy(node_sum, g->node_sum.data(), g->node_sum.size() * sizeof(double));
}
void tw_cg_free(void* h) { delete static_cast<Graph*>(h); }

}  // extern "C"
