"""repo_pft64 = repo_abl with 64-bit node numbers in the persistent tree: only for the airline models without grouping."""
from pathlib import Path
R = Path("/mnt/data/yingli/counterprovenance/repo_lean64/counterprovenance")
def edit(p, pairs):
    s = p.read_text()
    for a, b, n in pairs:
        assert s.count(a) == n, (a, s.count(a))
        s = s.replace(a, b)
    p.write_text(s)
edit(R / "native/cell_graph.cpp", [
 ("    std::vector<int32_t> face_root;", "    std::vector<int64_t> face_root;", 1),
 ("    std::vector<int32_t> left, right;", "    std::vector<int64_t> left, right;", 1),
 ("    int32_t build(const std::vector<int32_t>& state) {\n        std::vector<int32_t> level(g_.width, -1);", "    int64_t build(const std::vector<int32_t>& state) {\n        std::vector<int64_t> level(g_.width, -1);", 1),
 ("            std::vector<int32_t> up(level.size() / 2);", "            std::vector<int64_t> up(level.size() / 2);", 1),
 ("    int32_t derive(int32_t root, const", "    int64_t derive(int64_t root, const", 1),
 ("    bool full() const { return g_.left.size() >= 2147483000ull; }", "    bool full() const { return false; }", 1),
 ("    static int32_t encode(int32_t item) { return -(item + 2); }", "    static int64_t encode(int32_t item) { return -(static_cast<int64_t>(item) + 2); }", 1),
 ("    double total(int32_t child) const {\n        if (child >= 0) return g_.node_sum[child];\n        int32_t item = -child - 2;", "    double total(int64_t child) const {\n        if (child >= 0) return g_.node_sum[child];\n        int64_t item = -child - 2;", 1),
 ("    int32_t make(int32_t l, int32_t r) {", "    int64_t make(int64_t l, int64_t r) {", 1),
 ("        return static_cast<int32_t>(g_.left.size() - 1);", "        return static_cast<int64_t>(g_.left.size() - 1);", 1),
 ("    int32_t rec(int32_t node, int depth,", "    int64_t rec(int64_t node, int depth,", 1),
 ("        int32_t l = g_.left[node], r = g_.right[node];", "        int64_t l = g_.left[node], r = g_.right[node];", 1),
 ("            int32_t root = pft.derive(g.face_root[v], to);", "            int64_t root = pft.derive(g.face_root[v], to);", 1),
 ("double* face_value, int32_t* face_root, uint8_t* face_empty,\n                int32_t* left, int32_t* right, double* node_sum) {", "double* face_value, int64_t* face_root, uint8_t* face_empty,\n                int64_t* left, int64_t* right, double* node_sum) {", 1),
 ("g->face_root.size() * sizeof(int32_t));", "g->face_root.size() * sizeof(int64_t));", 1),
 ("g->left.size() * sizeof(int32_t));", "g->left.size() * sizeof(int64_t));", 1),
 ("g->right.size() * sizeof(int32_t));", "g->right.size() * sizeof(int64_t));", 1),
])
edit(R / "native_cells.py", [
 ("lib.tw_cg_take.argtypes = [C.c_void_p, _U32, _F64, _I32, _U8, _I32, _I32, _F64]", "lib.tw_cg_take.argtypes = [C.c_void_p, _U32, _F64, _I64, _U8, _I64, _I64, _F64]", 1),
 ("f = np.array([root], dtype=np.int32)", "f = np.array([root], dtype=np.int64)", 1),
 ("self.face_version = np.empty(faces, dtype=np.int32)", "self.face_version = np.empty(faces, dtype=np.int64)", 1),
 ("left, right = np.empty(nodes, dtype=np.int32), np.empty(nodes, dtype=np.int32)", "left, right = np.empty(nodes, dtype=np.int64), np.empty(nodes, dtype=np.int64)", 1),
 ("self.face_version.ctypes.data_as(_I32), self.face_empty.ctypes.data_as(_U8),", "self.face_version.ctypes.data_as(_I64), self.face_empty.ctypes.data_as(_U8),", 1),
 ("left.ctypes.data_as(_I32), right.ctypes.data_as(_I32), sums.ctypes.data_as(_F64))", "left.ctypes.data_as(_I64), right.ctypes.data_as(_I64), sums.ctypes.data_as(_F64))", 1),
])
edit(R / "index.py", [("            8 * self.pft_nodes\n", "            16 * self.pft_nodes\n", 1)])
print("patched")
