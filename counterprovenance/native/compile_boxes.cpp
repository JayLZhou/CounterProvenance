// Leaf boxes of a tree ensemble over the intervention axes, through mediators.
//
// The same algorithm as scm._compile_trees and scm._Mediator.split, in the same order, so that the boxes
// are identical to the Python ones, bit for bit: the stack discipline, the choice of the cut (median of the
// distinct boundaries strictly inside the box, on the axis with the most of them, first axis on ties), the
// correctly rounded sum (a port of CPython's math.fsum), and the way a model reads its input (rounding to
// float32, and LightGBM's zero threshold).  Build: g++ -O3 -std=c++17 -shared -fPIC -ffp-contract=off -pthread.
//
// Two things make it faster than a line-by-line port, and neither changes the output.  (1) A box inside a box
// only meets mediator boxes that its parent meets, so every box carries, per mediator, the mediator boxes that
// meet it, and a split filters that list rather than all of them (a sub-sequence keeps the order).  (2) Trees
// are independent: they are compiled by several threads and concatenated in tree order.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace {

// math.fsum of CPython: Shewchuk's partials, then a correctly rounded (half-even) sum.
double fsum(const std::vector<double>& xs, std::vector<double>& p) {
    size_t n = 0;
    if (p.size() < xs.size() + 2) p.resize(xs.size() + 2);
    for (double x : xs) {
        size_t i = 0;
        for (size_t j = 0; j < n; ++j) {
            double y = p[j];
            if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
            double hi = x + y, yr = hi - x, lo = y - yr;
            if (lo != 0.0) p[i++] = lo;
            x = hi;
        }
        n = i;
        if (x != 0.0) {
            if (n >= p.size()) p.resize(2 * n + 2);
            p[n++] = x;
        }
    }
    double hi = 0.0;
    if (n > 0) {
        double lo = 0.0;
        hi = p[--n];
        while (n > 0) {
            double x = hi, y = p[--n];
            hi = x + y;
            double yr = hi - x;
            lo = y - yr;
            if (lo != 0.0) break;
        }
        if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
            double y = lo * 2.0, x = hi + y, yr = x - hi;
            if (y == yr) hi = x;
        }
    }
    return hi;
}

struct Mediator {
    int64_t n;
    const double* lo;      // n x d
    const double* hi;      // n x d
    const int64_t* tree;   // n, non-decreasing
    const double* diff;    // n
    double observed;
};

struct Cast {
    bool f32;
    double zero;
    double operator()(double v) const {
        double out = f32 ? static_cast<double>(static_cast<float>(v)) : v;
        if (zero > 0.0 && std::fabs(out) <= zero) out = 0.0;
        return out;
    }
};

using Ids = std::vector<int32_t>;
using IdsPtr = std::shared_ptr<const Ids>;

struct Piece {
    bool go_left;
    std::vector<double> lo, hi;
    IdsPtr idx;   // the mediator boxes that meet this piece
};

struct SplitItem {
    std::vector<double> lo, hi;
    Ids idx;
};

class Splitter {
public:
    Splitter(int d, Cast cast) : d_(d), cast_(cast) {}

    // false: no boundary inside a straddling box (cannot happen for a consistent mediator)
    bool split(const Mediator& m, double theta, const std::vector<double>& lo0, const std::vector<double>& hi0,
               const IdsPtr& cand, std::vector<Piece>& out) {
        out.clear();
        std::vector<SplitItem> stack;
        SplitItem first{lo0, hi0, {}};
        if (cand) {
            first.idx = *cand;
        } else {
            first.idx.resize(m.n);
            for (int64_t i = 0; i < m.n; ++i) first.idx[i] = static_cast<int32_t>(i);
        }
        stack.push_back(std::move(first));
        while (!stack.empty()) {
            SplitItem it = std::move(stack.back());
            stack.pop_back();
            const double* blo = it.lo.data();
            const double* bhi = it.hi.data();
            size_t w = 0;
            for (int32_t i : it.idx) {
                const double* l = m.lo + static_cast<int64_t>(i) * d_;
                const double* h = m.hi + static_cast<int64_t>(i) * d_;
                bool in = true;
                for (int a = 0; a < d_; ++a)
                    if (!(l[a] < bhi[a]) || !(h[a] > blo[a])) { in = false; break; }
                if (in) it.idx[w++] = i;
            }
            it.idx.resize(w);
            double vlo = m.observed, vhi = m.observed;
            if (!it.idx.empty()) {
                mins_.clear();
                maxs_.clear();
                size_t s = 0;
                while (s < it.idx.size()) {
                    int64_t tr = m.tree[it.idx[s]];
                    double mn = m.diff[it.idx[s]], mx = mn;
                    size_t e = s + 1;
                    for (; e < it.idx.size() && m.tree[it.idx[e]] == tr; ++e) {
                        double v = m.diff[it.idx[e]];
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                    }
                    mins_.push_back(mn);
                    maxs_.push_back(mx);
                    s = e;
                }
                vlo = m.observed + fsum(mins_, partials_);
                vhi = m.observed + fsum(maxs_, partials_);
            }
            vlo = cast_(vlo);
            vhi = cast_(vhi);
            if (vhi <= theta) {
                out.push_back(Piece{true, std::move(it.lo), std::move(it.hi), std::make_shared<const Ids>(std::move(it.idx))});
                continue;
            }
            if (vlo > theta) {
                out.push_back(Piece{false, std::move(it.lo), std::move(it.hi), std::make_shared<const Ids>(std::move(it.idx))});
                continue;
            }
            int best_axis = -1;
            size_t best_count = 0;
            double best_cut = 0.0;
            for (int a = 0; a < d_; ++a) {
                cuts_.clear();
                for (int32_t i : it.idx) {
                    double l = m.lo[static_cast<int64_t>(i) * d_ + a], h = m.hi[static_cast<int64_t>(i) * d_ + a];
                    if (l > blo[a] && l < bhi[a]) cuts_.push_back(l);
                    if (h > blo[a] && h < bhi[a]) cuts_.push_back(h);
                }
                if (cuts_.empty()) continue;
                std::sort(cuts_.begin(), cuts_.end());
                cuts_.erase(std::unique(cuts_.begin(), cuts_.end()), cuts_.end());
                if (best_axis < 0 || cuts_.size() > best_count) {
                    best_axis = a;
                    best_count = cuts_.size();
                    best_cut = cuts_[cuts_.size() / 2];
                }
            }
            if (best_axis < 0) return false;
            SplitItem right{it.lo, it.hi, it.idx};
            right.lo[best_axis] = best_cut;
            it.hi[best_axis] = best_cut;
            stack.push_back(std::move(right));
            stack.push_back(std::move(it));
        }
        return true;
    }

private:
    int d_;
    Cast cast_;
    std::vector<double> mins_, maxs_, cuts_, partials_;
};

struct Result {
    int d = 0;
    std::vector<int64_t> tree, node;
    std::vector<double> lo, hi;
};

struct NodeItem {
    int64_t node;
    std::vector<double> lo, hi;
    std::vector<IdsPtr> cand;   // per mediator; null: all its boxes
};

}  // namespace

extern "C" {

// status: 0 ok, 1 more than max_boxes boxes, 2 a straddling box without a boundary inside
void* tw_compile(int d, int64_t n_trees, const int64_t* tree_ptr, const int64_t* feature, const double* threshold,
                 const int64_t* left, const int64_t* right, const int32_t* kind, const double* fixed,
                 const int64_t* arg, int64_t n_med, const int64_t* med_n, const double* const* med_lo,
                 const double* const* med_hi, const int64_t* const* med_tree, const double* const* med_diff,
                 const double* med_observed, int is_float32, double zero_threshold, int64_t max_boxes,
                 int threads, int64_t* status) {
    std::vector<Mediator> meds(n_med);
    for (int64_t i = 0; i < n_med; ++i)
        meds[i] = Mediator{med_n[i], med_lo[i], med_hi[i], med_tree[i], med_diff[i], med_observed[i]};
    Cast cast{is_float32 != 0, zero_threshold};
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<Result> parts(n_trees);
    std::atomic<int64_t> next{0}, total{0};
    std::atomic<int> stop{0};

    auto worker = [&]() {
        Splitter splitter(d, cast);
        std::vector<NodeItem> stack;
        std::vector<Piece> pieces;
        for (;;) {
            int64_t m = next.fetch_add(1);
            if (m >= n_trees || stop.load()) return;
            Result& res = parts[m];
            const int64_t base = tree_ptr[m];
            stack.clear();
            stack.push_back(NodeItem{0, std::vector<double>(d, -inf), std::vector<double>(d, inf),
                                     std::vector<IdsPtr>(n_med)});
            while (!stack.empty()) {
                NodeItem it = std::move(stack.back());
                stack.pop_back();
                int64_t f = feature[base + it.node];
                if (f < 0) {
                    res.tree.push_back(m);
                    res.node.push_back(it.node);
                    res.lo.insert(res.lo.end(), it.lo.begin(), it.lo.end());
                    res.hi.insert(res.hi.end(), it.hi.begin(), it.hi.end());
                    if (total.fetch_add(1) + 1 > max_boxes) {
                        stop.store(1);
                        return;
                    }
                    continue;
                }
                double theta = threshold[base + it.node];
                int64_t l = left[base + it.node], r = right[base + it.node];
                if (kind[f] == 0) {  // fixed
                    it.node = fixed[f] <= theta ? l : r;
                    stack.push_back(std::move(it));
                } else if (kind[f] == 1) {  // axis
                    int64_t a = arg[f];
                    if (theta >= it.hi[a]) {
                        it.node = l;
                        stack.push_back(std::move(it));
                    } else if (theta <= it.lo[a]) {
                        it.node = r;
                        stack.push_back(std::move(it));
                    } else {
                        NodeItem rr{r, it.lo, it.hi, it.cand};
                        rr.lo[a] = theta;
                        it.hi[a] = theta;
                        it.node = l;
                        stack.push_back(std::move(rr));
                        stack.push_back(std::move(it));
                    }
                } else {  // mediator
                    int64_t k = arg[f];
                    if (!splitter.split(meds[k], theta, it.lo, it.hi, it.cand[k], pieces)) {
                        stop.store(2);
                        return;
                    }
                    for (Piece& p : pieces) {
                        NodeItem ch{p.go_left ? l : r, std::move(p.lo), std::move(p.hi), it.cand};
                        ch.cand[k] = std::move(p.idx);
                        stack.push_back(std::move(ch));
                    }
                }
            }
        }
    };

    int n_threads = std::max(1, threads);
    std::vector<std::thread> pool;
    for (int t = 1; t < n_threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    auto* res = new Result();
    res->d = d;
    *status = stop.load();
    if (*status != 0) return res;
    size_t n = static_cast<size_t>(total.load());
    res->tree.reserve(n);
    res->node.reserve(n);
    res->lo.reserve(n * d);
    res->hi.reserve(n * d);
    for (Result& p : parts) {   // tree order, whatever thread compiled the tree
        res->tree.insert(res->tree.end(), p.tree.begin(), p.tree.end());
        res->node.insert(res->node.end(), p.node.begin(), p.node.end());
        res->lo.insert(res->lo.end(), p.lo.begin(), p.lo.end());
        res->hi.insert(res->hi.end(), p.hi.begin(), p.hi.end());
        std::vector<int64_t>().swap(p.tree);
        std::vector<int64_t>().swap(p.node);
        std::vector<double>().swap(p.lo);
        std::vector<double>().swap(p.hi);
    }
    return res;
}

int64_t tw_count(void* handle) { return static_cast<int64_t>(static_cast<Result*>(handle)->tree.size()); }

void tw_copy(void* handle, int64_t* tree, int64_t* node, double* lo, double* hi) {
    auto* r = static_cast<Result*>(handle);
    size_t n = r->tree.size();
    if (!n) return;
    std::memcpy(tree, r->tree.data(), n * sizeof(int64_t));
    std::memcpy(node, r->node.data(), n * sizeof(int64_t));
    std::memcpy(lo, r->lo.data(), r->lo.size() * sizeof(double));
    std::memcpy(hi, r->hi.data(), r->hi.size() * sizeof(double));
}

void tw_free(void* handle) { delete static_cast<Result*>(handle); }

}  // extern "C"
