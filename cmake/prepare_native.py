"""Stage native headers with narrow fixes; never edit the source checkout.

See docs/native-build.md for the defects in Schmate e2e7d9a addressed here.
"""

from pathlib import Path
import shutil
import sys

source, destination = map(Path, sys.argv[1:])

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)

patched_headers = {"hnswlib/space_ip.h", "hnswlib/space_l2.h", "hnswlib/space_quantized.h", "unified_hnsw.hpp"}
for original in source.rglob("*"):
    if not original.is_file():
        continue
    relative = original.relative_to(source)
    if relative.as_posix() in patched_headers:
        continue
    output = destination / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or original.read_bytes() != output.read_bytes():
        shutil.copy2(original, output)

path = destination / "hnswlib/space_ip.h"
text = (source / "hnswlib/space_ip.h").read_text()
if "static float InnerProductSIMD4Ext(const void*" in text:
    start = text.index("static float InnerProductSIMD4Ext(const void*")
    end = text.index("#endif // USE_AVX", start)
    text = text[:start] + """static float InnerProductSIMD4ExtAVX(const void* av, const void* bv, const void* qty) {
    const auto* a = static_cast<const float*>(av);
    const auto* b = static_cast<const float*>(bv);
    const size_t n = *static_cast<const size_t*>(qty);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = 0;
    for (float lane : lanes) result += lane;
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
static float InnerProductDistanceSIMD4ExtAVX(const void* a, const void* b, const void* qty) {
    return 1.0f - InnerProductSIMD4ExtAVX(a, b, qty);
}

""" + text[end:]
if "static float InnerProductSIMD4ExtSSE(" not in text:
    anchor = "#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)"
    if text.count(anchor) != 1:
        raise RuntimeError("Schmate's SIMD header changed; review the compatibility fix")
    text = text.replace(anchor, """#if defined(USE_SSE)
static float InnerProductSIMD4ExtSSE(const void* av, const void* bv, const void* qty) {
    const auto* a = static_cast<const float*>(av);
    const auto* b = static_cast<const float*>(bv);
    const size_t n = *static_cast<const size_t*>(qty);
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(a+i), _mm_loadu_ps(b+i)));
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}
static float InnerProductDistanceSIMD4ExtSSE(const void* a, const void* b, const void* qty) {
    return 1.0f - InnerProductSIMD4ExtSSE(a, b, qty);
}
#endif

""" + anchor)
text = text.replace("        dim_ = dim;\n", "        fstdistfunc_ = InnerProductDistance;\n        dim_ = dim;\n")
write(path, text)

logger = (source.parent / "src/Logger.cpp").read_text()
logger = logger.replace("openlog(prefix_,", "openlog(prefix_.c_str(),")
logger = logger.replace("file_stream.open(filename,", "file_stream.open(std::string(filename),")
output = destination.parent / "native-src/Logger.cpp"
write(output, logger)

sharded = (source.parent / "src/ShardedIndex.cpp").read_text()
sharded = sharded.replace("    shard->set_storage_path_dir (base_dir);",
                          "    if (!base_dir.empty()) shard->set_storage_path_dir(base_dir);")
sharded = sharded.replace("        shards.push_back(make_unique<BertIndex>(embedder, cfg,newname, false, opaque_ptr));",
    "        add_shard(shards.size());")
sharded = sharded.replace("if (cfg.parallel_merge) return merge_two_parallel(n);", "// Python builds use the repaired serial publication path.")
start = sharded.index("bool ShardedIndex::merge_two_serial(size_t n) {")
end = sharded.index("\n\n/*", start)
sharded = sharded[:start] + (Path(__file__).parents[1] / "bindings/merge_serial.inc").read_text() + sharded[end:]
write(destination.parent / "native-src/ShardedIndex.cpp", sharded)

unified = (source / "unified_hnsw.hpp").read_text()
unified = unified.replace("    void markDelete(labeltype label) {", """    bool is_deleted(labeltype label) const {
        if (!index_) return true;
        const auto found = index_->label_lookup_.find(label);
        return found == index_->label_lookup_.end() || index_->isMarkedDeleted(found->second);
    }
    void markDelete(labeltype label) {""")
unified = unified.replace("if (index_) index_->markDelete(label);", "if (index_) { index_->markDelete(label); ++additions_since_flush_; }")
unified = unified.replace("if (index_) index_->unmarkDelete(label);", "if (index_) { index_->unmarkDelete(label); ++additions_since_flush_; }")
write(destination / "unified_hnsw.hpp", unified)

l2 = (source / "hnswlib/space_l2.h").read_text()
l2 = l2.replace("//        fstdistfunc_ = L2Sqr;", "        fstdistfunc_ = L2Sqr;")
write(destination / "hnswlib/space_l2.h", l2)

offset = (source.parent / "src/OffsetFile.cpp").read_text()
offset = offset.replace("    if (length < header_size) {",
    "    this->max_entries = max_entries;\n    if (length < header_size) {")
write(destination.parent / "native-src/OffsetFile.cpp", offset)

index = (source.parent / "src/BertIndex.cpp").read_text()
# Scores are similarities after conversion; the old filters compare them as if
# they were raw distances. Keep the native candidate searches and gap policy.
index = index.replace("return score <= r;", "return score >= r;")
index = index.replace("float best = topk.front().first;", "float best = score_from_dist(topk.front().first);")
index = index.replace("float threshold = accepted.back();", "float threshold = score_from_dist(accepted.back());")
index = index.replace("return score <= threshold;", "return score >= threshold;")
index = index.replace("return std::clamp(dist, -1.0f, 1.0f);", "return std::clamp(1.0f - dist, -1.0f, 1.0f);")
index = index.replace("    size_t max_elements = cfg.max_elements;", "    metric = cfg.metric();\n    size_t max_elements = cfg.max_elements;")
index = index.replace("  offsets = std::make_unique<OffsetFile>(offsets_path, max_elements);",
    "  offsets = std::make_unique<OffsetFile>(offsets_path, max_elements);\n  next_label = index->getCurrentElementCount();")
index = index.replace("""    auto it = chunk_sentence_map.find(label);
    if (it == chunk_sentence_map.end())
        return -1;
    return it->second;""", """    return offsets->get(label).sid;""")
index = index.replace("""    // clear offset entry
    OffsetEntry empty{};
    offsets->set(label, empty);

    // recycle label
    free_labels.push(label);""", """    // Keep offset metadata for undelete; the graph owns the tombstone.
    // Do not reuse this label while its deletion can be reversed.""")
start = index.index("void BertIndex::flush() {")
end = index.index("\nvoid BertIndex::save() {", start)
index = index[:start] + """void BertIndex::flush() {
    if (searchOnly) return;
    save();
    if (offsets) offsets->flush();
    if (sentences) sentences->flush();
}

""" + index[end:]
index = index.replace("    search_ctrl.save(name);", "    if (!searchOnly) search_ctrl.save(name);")
write(destination.parent / "native-src/BertIndex.cpp", index)

path = destination / "hnswlib/space_quantized.h"
text = (source / "hnswlib/space_quantized.h").read_text()
# This native branch calls a function that does not exist. Preserve an explicit
# unsupported-feature error instead of inventing a packed-distance algorithm.
text = text.replace("return compute_dist_L2_pass(bits, a, b, dim);",
    'throw std::runtime_error("This Schmate revision does not implement L2 PASS storage");')
write(path, text)
