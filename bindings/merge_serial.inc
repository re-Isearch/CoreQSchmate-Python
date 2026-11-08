// Replacement for the current native merge implementation. Rebuild one chunk
// at a time, retain IDs/token spans, close files before publication, then reopen
// against the manager's long-lived config. Use with exclusive index access.
bool ShardedIndex::merge_two_serial(size_t n) {
    namespace fs = std::filesystem;
    if (n < 2) return false;
    const size_t first = n - 2, second = n - 1;
    const std::string temporary = base_name + IndexFileExtensions::merge;
    const std::string target = shard_basename(first);
    const std::string backup = target + "_merge_backup";
    const std::vector<std::string> extensions = {
        IndexFileExtensions::sentences, IndexFileExtensions::offsets,
        IndexFileExtensions::hnsw
    };
    for (const auto& ext : extensions)
        if (fs::exists(temporary + ext) || fs::exists(backup + ext))
            throw std::runtime_error("Existing merge files require inspection: " + temporary);

    flush();
    HnswConfig merged_config = cfg;
    merged_config.max_elements = shards[first]->size() + shards[second]->size() + 16;
    auto merged = std::make_unique<BertIndex>(embedder, merged_config, temporary, false, opaque_ptr);
    try {
        for (size_t s : {first, second}) {
            auto& original = *shards[s];
            original.offsets->for_each([&](size_t label, const OffsetEntry& entry) {
                if (entry.sid == 0 || entry.file_end <= entry.file_start ||
                    original.index->is_deleted(label)) return;
                const auto text = original.get_text_by_label(label);
                const auto before = merged->size();
                const auto new_label = merged->append(text, entry.sid, entry.span);
                if (merged->size() != before + 1)
                    throw std::runtime_error("Merge would change chunk boundaries; source files preserved");
                auto saved = merged->offsets->get(new_label);
                saved.start_tok = entry.start_tok;
                saved.end_tok = entry.end_tok;
                saved.span = entry.span;
                merged->offsets->set(new_label, saved);
            });
        }
        merged->flush();
        const auto expected = merged->size();
        merged.reset();
        // An empty compacted shard has no .hix; retain one marked-deleted point
        // rather than publish files that native discovery cannot reopen.
        if (expected == 0)
            throw std::runtime_error("No live chunks to merge; clear the index instead");
    } catch (...) {
        merged.reset();
        for (const auto& ext : extensions) fs::remove(temporary + ext);
        throw;
    }

    // No object may retain the old filenames or a reference to merged_config.
    shards[first].reset();
    shards[second].reset();
    std::vector<std::string> backed_up, published;
    try {
        for (const auto& ext : extensions) {
            fs::rename(target + ext, backup + ext);
            backed_up.push_back(ext);
        }
        for (const auto& ext : extensions) {
            fs::rename(temporary + ext, target + ext);
            published.push_back(ext);
        }
        shards[first] = std::make_unique<BertIndex>(embedder, cfg, target, false, opaque_ptr);
    } catch (...) {
        const auto failure = std::current_exception();
        shards[first].reset();
        for (auto it = published.rbegin(); it != published.rend(); ++it)
            fs::rename(target + *it, temporary + *it);
        for (auto it = backed_up.rbegin(); it != backed_up.rend(); ++it)
            fs::rename(backup + *it, target + *it);
        shards[first] = std::make_unique<BertIndex>(embedder, cfg, target, false, opaque_ptr);
        shards[second] = std::make_unique<BertIndex>(embedder, cfg, shard_basename(second), false, opaque_ptr);
        std::rethrow_exception(failure);
    }
    shards.pop_back();
    shard_tuners.pop_back();
    for (const auto& ext : extensions) {
        fs::remove(backup + ext);
        fs::remove(shard_basename(second) + ext);
    }
    return true;
}
