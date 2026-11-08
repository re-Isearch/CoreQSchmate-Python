#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include "BertIndexManager.hpp"
#include "ConfigBuilder.hpp"
#include "StringEmbedding.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace hnswlib;

namespace {
void text_ok(const std::string& text, bool empty = false) {
    if ((!empty && text.empty()) || text.find('\0') != std::string::npos)
        throw nb::value_error("A nonempty string without NUL bytes is required");
}
void finite(float value) {
    if (!std::isfinite(value)) throw nb::value_error("Numeric options must be finite");
}
void validate(const HnswConfig& config) {
    for (float v : {config.overlap_percent, config.default_radius, config.default_alpha,
                   config.default_gapDelta, config.default_epsilon, config.default_epsilonL2,
                   config.default_epsilonIP, config.deletion_threshold_pc}) finite(v);
    if (!config.validate()) throw nb::value_error("Invalid HNSW configuration");
}
void copy_config(HnswConfig& destination, const HnswConfig& source) {
    destination = source;
    // Native SpecificationString assignment currently omits this flag.
    destination.specification.rescore_ = source.specification.rescore_;
}
void set_config(HnswConfig& config, const std::string& key, const std::string& value) {
    HnswConfig candidate(config);
    if (!candidate.set(key, value)) throw nb::value_error("Unknown configuration key or invalid value");
    validate(candidate);
    copy_config(config, candidate);
}

// Keep the native model/encoding implementation, while honoring the Python
// constructor's explicit thread count. Native SBertGGML otherwise auto-selects
// its encoding thread count independently of that constructor argument.
struct PythonEmbedder : SBertGGML {
    size_t threads;
    std::mutex gate;
    PythonEmbedder(const std::string& path, size_t count) : SBertGGML(path, count), threads(count) {}
    explicit PythonEmbedder(size_t dim) : SBertGGML(dim), threads(1) {}
    std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts, bool = false) override {
        if (!ctx) throw nb::value_error("A precomputed-vector model cannot encode text");
        std::lock_guard<std::mutex> lock(gate);
        std::vector<std::vector<float>> output(texts.size(), std::vector<float>(n_embd, 0.0f));
        bert_batch batch;
        std::vector<size_t> positions;
        for (size_t i = 0; i < texts.size(); ++i) {
            text_ok(texts[i], true);
            if (texts[i].empty()) continue;
            auto tokens = bert_tokenize(ctx, texts[i].c_str(), bert_n_max_tokens(ctx));
            if (!tokens.empty()) { batch.push_back(std::move(tokens)); positions.push_back(i); }
        }
        if (!batch.empty()) {
            std::vector<float> flat(batch.size() * n_embd);
            bert_forward_batch(ctx, batch, flat.data(), static_cast<int>(threads));
            for (size_t i = 0; i < positions.size(); ++i)
                output[positions[i]].assign(flat.data()+i*n_embd, flat.data()+(i+1)*n_embd);
        }
        return output;
    }
    std::vector<float> encode_text(const std::string& text, bool debug = false) override {
        if (schmate_util::isFloatArrayVector(text, n_embd))
            return schmate_util::floatArrayToFloat32(text);
        return encode_batch({text}, debug).front();
    }
};

struct ModelState { std::unique_ptr<PythonEmbedder> value; };
struct Model {
    std::shared_ptr<ModelState> state;
    Model(const std::string& path, size_t threads) {
        text_ok(path);
        if (!threads || threads > 1024) throw nb::value_error("threads must be between 1 and 1024");
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Model file does not exist: " + path);
        state = std::make_shared<ModelState>();
        state->value = std::make_unique<PythonEmbedder>(path, threads);
    }
    explicit Model(size_t dim) {
        if (!dim || dim > 65536) throw nb::value_error("dimension must be between 1 and 65536");
        state = std::make_shared<ModelState>();
        state->value = std::make_unique<PythonEmbedder>(dim);
    }
    PythonEmbedder& get() const {
        if (!state) throw std::runtime_error("Embedding model is closed");
        return *state->value;
    }
};

struct Manager {
    std::shared_ptr<ModelState> model;
    HnswConfig config;
    std::filesystem::path directory;
    bool search_only;
    size_t cache_size;
    std::unique_ptr<BertIndexManager> value;
    Manager(Model& embedding, const HnswConfig& cfg, const std::string& path,
            size_t cache_size, bool reading) : model(embedding.state), config(cfg), search_only(reading), cache_size(cache_size) {
        embedding.get();
        text_ok(path);
        if (!cache_size) throw nb::value_error("cache_size must be positive");
        validate(config);
        config.bert_n_threads = model->value->threads;
        directory = std::filesystem::absolute(path).lexically_normal();
        if (!reading) std::filesystem::create_directories(directory);
        if (!std::filesystem::is_directory(directory)) throw std::runtime_error("Index directory does not exist");
        value = std::make_unique<BertIndexManager>(*model->value, config, cache_size, reading);
    }
    BertIndexManager& get() const {
        if (!value) throw std::runtime_error("Index manager is closed");
        return *value;
    }
    std::string name(const std::string& name) const {
        get(); text_ok(name);
        if (name == "." || name == ".." || name.find_first_of("/\\") != std::string::npos)
            throw nb::value_error("An index name must be a single filename component");
        auto path = (directory / name).string();
        if (search_only && !ShardedIndex::Exists(path)) throw std::runtime_error("Index does not exist");
        return path;
    }
    void writer() const {
        get();
        if (search_only) throw std::runtime_error("Manager was opened with search_only=True");
    }
    std::string input(nb::handle object) const {
        get();
        if (nb::isinstance<nb::str>(object)) {
            auto text = nb::cast<std::string>(object); text_ok(text);
            if (!model->value->ctx) throw nb::value_error("Precomputed-vector mode requires a vector sequence");
            return text;
        }
        auto values = nb::cast<std::vector<float>>(object);
        if (values.size() != static_cast<size_t>(model->value->n_embd))
            throw nb::value_error("Vector dimension does not match the model");
        if (std::all_of(values.begin(), values.end(), [](float v){return v == 0.0f;}))
            throw nb::value_error("This native backend requires a nonzero vector");
        std::ostringstream stream;
        stream << '[' << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (size_t i = 0; i < values.size(); ++i) {
            finite(values[i]);
            if (i) stream << ',';
            stream << values[i];
        }
        stream << ']';
        return stream.str();
    }
    void close() {
        if (value && !search_only) value->flush_all();
        value.reset();
        model.reset();
    }
    void clear(const std::string& index_name) {
        writer();
        const auto target = name(index_name);
        value->flush_all();
        value->clear(target);
        // Native clear empties a ShardedIndex but leaves it in the LRU. Reopen
        // the manager cache so the cleared index is usable on the next append.
        value.reset();
        value = std::make_unique<BertIndexManager>(*model->value, config, cache_size, false);
    }
};
}

NB_MODULE(_core, m) {
    m.doc() = "Standalone Schmate embeddings and sharded HNSW indexes";
    m.attr("schmate_revision") = COREQSCHMATE_schmate_REVISION;
    m.attr("bert_revision") = COREQSCHMATE_bert_REVISION;
    m.attr("ggml_revision") = COREQSCHMATE_ggml_REVISION;
    nb::enum_<Metric>(m, "Metric").value("L1",Metric::L1).value("L2",Metric::L2)
        .value("IP",Metric::IP).value("COSINE",Metric::Cosine);
    nb::enum_<SearchModes>(m, "SearchMode").value("KNN",SearchModes::Knn)
        .value("RADIUS",SearchModes::Radius).value("RELATIVE",SearchModes::Relative)
        .value("ADAPTIVE",SearchModes::Adaptive).value("EPSILON",SearchModes::Epsilon);
    nb::enum_<QuantMode>(m, "Quantization").value("NONE",QuantMode::NONE).value("BIN1",QuantMode::BIN1)
        .value("INT158",QuantMode::INT158).value("INT4",QuantMode::INT4).value("INT8",QuantMode::INT8)
        .value("INT16",QuantMode::INT16).value("FP16",QuantMode::FP16).value("BF16",QuantMode::BF16);
    nb::enum_<OptBinMode>(m, "QuantizationAlgorithm").value("PASS",OptBinMode::PASS)
        .value("STANDARD",OptBinMode::STANDARD).value("BETTER",OptBinMode::BETTER)
        .value("CENTROID",OptBinMode::CENTROID).value("ROTATIONAL",OptBinMode::ROTATIONAL)
        .value("RABITQ",OptBinMode::RABITQ).value("RABITQ_EXTENDED",OptBinMode::RABITQ_EXTENDED);
    auto storage = nb::enum_<StorageType>(m, "StorageType");
#define STORAGE(name) storage.value(#name, StorageType::name)
    STORAGE(BIN1); STORAGE(INT2); STORAGE(INT3); STORAGE(INT4); STORAGE(FP4); STORAGE(INT5);
    STORAGE(INT6); STORAGE(INT8); STORAGE(INT16); STORAGE(INT32); STORAGE(INT64);
    STORAGE(FP16); STORAGE(BF16); STORAGE(FLOAT32); STORAGE(FLOAT64);
#undef STORAGE

    auto cfg = nb::class_<HnswConfig>(m,"Config").def(nb::init<>())
        .def("copy",[](const HnswConfig& c){ return HnswConfig(c); })
        .def("validate",&HnswConfig::validate)
        .def("set",&set_config,"key"_a,"value"_a)
        .def("get",&HnswConfig::get,"key"_a)
        .def_static("keys",&HnswConfig::get_all_keys)
        .def("__str__",[](const HnswConfig& c){std::ostringstream s;c.print(s);return s.str();})
        .def("save",[](const HnswConfig& c,const std::string& path){validate(c);if(!c.save_to_file(path))throw std::runtime_error("Cannot save configuration");})
        .def_static("load",[](const std::string& path){HnswConfig c;if(!c.load_from_file(path))throw std::runtime_error("Cannot load configuration");validate(c);return c;})
        .def_prop_rw("metric",&HnswConfig::metric,&HnswConfig::set_metric)
        .def_prop_rw("storage_type",&HnswConfig::storage_type,&HnswConfig::set_storage_type)
        .def_prop_rw("quantization",&HnswConfig::quantization,&HnswConfig::set_quantization)
        .def_prop_rw("quantization_algorithm",&HnswConfig::mode,&HnswConfig::set_mode)
        .def_prop_rw("enable_rescoring",&HnswConfig::enable_rescoring,&HnswConfig::set_enable_rescoring)
        .def("set_autotune",&HnswConfig::set_autotune,"enabled"_a=true);
#define CONFIG(name) cfg.def_rw(#name,&HnswConfig::name)
    CONFIG(model_name); CONFIG(max_elements); CONFIG(M); CONFIG(ef_construction); CONFIG(ef_search);
    CONFIG(matryoshka_dim); CONFIG(bert_n_threads); CONFIG(max_tokens_per_chunk); CONFIG(overlap_percent);
    CONFIG(lock_on_append); CONFIG(debug); CONFIG(default_search_mode); CONFIG(default_k);
    CONFIG(default_radius); CONFIG(default_alpha); CONFIG(default_minN); CONFIG(default_lookahead);
    CONFIG(default_gapDelta); CONFIG(default_epsilon); CONFIG(default_epsilonL2); CONFIG(default_epsilonIP);
    CONFIG(deletion_threshold_pc); CONFIG(min_candidates); CONFIG(max_candidates_cap);
    CONFIG(unified_index); CONFIG(knn_lookahead_scale); CONFIG(flush_threshold); CONFIG(flush_offsets_each);
    CONFIG(parallel_merge); CONFIG(merge_threads); CONFIG(normalize_embeddings); CONFIG(auto_tune_ef); CONFIG(auto_tune_eps);
#undef CONFIG
    nb::class_<ConfigLoader>(m,"ConfigLoader")
        .def(nb::init<const std::string&,const std::string&>(),"global_path"_a="/etc/ib/hnsw_config.bin","local_path"_a="~/.ib/hnsw_config.bin")
        .def("load",&ConfigLoader::load,"debug"_a=false)
        .def("load_with_project",&ConfigLoader::load_with_project,"directory"_a);

    nb::class_<Model>(m,"EmbeddingModel")
        .def(nb::init<const std::string&,size_t>(),"path"_a,"threads"_a=4)
        .def_static("for_vectors",[](size_t dim){return std::make_unique<Model>(dim);},"dimension"_a)
        .def_prop_ro("dimension",[](const Model& e){return e.get().n_embd;})
        .def_prop_ro("model_name",[](const Model& e){return std::string(e.get().model_name());})
        .def_prop_ro("architecture",[](const Model& e){return std::string(e.get().model_architecture());})
        .def_prop_ro("threads",[](const Model& e){return e.get().threads;})
        .def("encode",[](Model& e,const std::string& t){text_ok(t,true);if(!e.get().ctx)throw nb::value_error("No text model loaded");return e.get().encode_text(t);},"text"_a)
        .def("encode_batch",[](Model& e,const std::vector<std::string>& texts){return e.get().encode_batch(texts);},"texts"_a)
        .def("tokenize",[](Model& e,const std::string& t){text_ok(t,true);auto&v=e.get();if(!v.ctx)throw nb::value_error("No text model loaded");return bert_tokenize(v.ctx,t.c_str(),bert_n_max_tokens(v.ctx));})
        .def("close",[](Model&e){e.state.reset();})
        .def_prop_ro("closed",[](const Model&e){return !e.state;})
        .def("__enter__",[](Model&e)->Model&{e.get();return e;},nb::rv_policy::reference)
        .def("__exit__",[](Model&e,nb::handle,nb::handle,nb::handle){e.state.reset();},
             "exc_type"_a.none(),"exc_value"_a.none(),"traceback"_a.none());

    auto result = nb::class_<SearchResult>(m,"SearchResult");
#define RESULT(name) result.def_ro(#name,&SearchResult::name)
    RESULT(score); RESULT(file_start); RESULT(file_end); RESULT(label); RESULT(token_start); RESULT(token_end);
    RESULT(sentence_id); RESULT(span); RESULT(shard); RESULT(text);
#undef RESULT

    auto manager = nb::class_<Manager>(m,"IndexManager")
        .def(nb::init<Model&,const HnswConfig&,const std::string&,size_t,bool>(),
             "model"_a,"config"_a,"directory"_a=".","cache_size"_a=3,"search_only"_a=false)
        .def("close",&Manager::close).def_prop_ro("closed",[](const Manager&v){return !v.value;})
        .def_prop_ro("config",[](const Manager&v){v.get();return HnswConfig(v.config);})
        .def("set_config",[](Manager&v,const std::string&key,const std::string&value){
            v.get();
            const std::vector<std::string> fixed={"model","metric","specification","max_elements","M","ef_construction","normalize_embeddings","max_tokens_per_chunk","overlap_percent","enable_rescoring","bert_n_threads","unified_index"};
            if(std::find(fixed.begin(),fixed.end(),key)!=fixed.end())throw nb::value_error("Set structural options before constructing the manager");
            set_config(v.config,key,value);
        },"key"_a,"value"_a)
        .def("use",[](Manager&v,const std::string&n){v.get().getOrCreate(v.name(n));},"name"_a)
        .def("append",[](Manager&v,const std::string&n,nb::handle input,nb::object sid,uint32_t span){
            v.writer();auto name=v.name(n);auto text=v.input(input);
            if(sid.is_none() && span)throw nb::value_error("span requires an explicit sentence_id");
            if(sid.is_none())v.get().append(name,text);
            else {auto id=nb::cast<int64_t>(sid);if(id==0)throw nb::value_error("sentence_id 0 is reserved by the native offset store");v.get().append(name,text,id,span);}
        },"name"_a,"text_or_vector"_a,"sentence_id"_a=nb::none(),"span"_a=0)
        .def("search",[](Manager&v,const std::string&n,nb::handle q){auto hits=v.get().search(v.name(n),v.input(q));if(v.config.default_search_mode==SearchModes::Knn && hits.size()>v.config.default_k)hits.resize(v.config.default_k);return hits;},"name"_a,"query"_a)
        .def("knn",[](Manager&v,const std::string&n,nb::handle q,size_t k){auto hits=v.get().knn(v.name(n),v.input(q),k);auto limit=k?k:v.config.default_k;if(hits.size()>limit)hits.resize(limit);return hits;},"name"_a,"query"_a,"k"_a=0)
        .def("radius",[](Manager&v,const std::string&n,nb::handle q,float s){finite(s);return v.get().radius(v.name(n),v.input(q),s);},"name"_a,"query"_a,"min_score"_a=-1.0f)
        .def("relative",[](Manager&v,const std::string&n,nb::handle q,float a){finite(a);return v.get().relative(v.name(n),v.input(q),a);},"name"_a,"query"_a,"alpha"_a=-1.0f)
        .def("adaptive",[](Manager&v,const std::string&n,nb::handle q,float a,size_t mn,size_t la,float gap){finite(a);finite(gap);return v.get().adaptive(v.name(n),v.input(q),a,mn,la,gap);},"name"_a,"query"_a,"alpha"_a=-1.0f,"min_n"_a=0,"lookahead"_a=0,"gap_delta"_a=-1.0f)
        .def("epsilon_search",[](Manager&v,const std::string&n,nb::handle q,float e){finite(e);return v.get().epsilon_search(v.name(n),v.input(q),e);},"name"_a,"query"_a,"epsilon"_a=-1.0f)
        .def("get_text",[](Manager&v,const std::string&n,const SearchResult&r,bool full){auto name=v.name(n);auto& index=v.get().getOrCreate(name);if(r.shard>=index.shard_count())throw nb::value_error("Result shard is not present in this index");return index.get_shard(r.shard).get_text(r,full);},"name"_a,"result"_a,"full_sentence"_a=false)
        .def("shard_count",[](Manager&v,const std::string&n){return v.get().shard_count(v.name(n));})
        .def("find_labels_by_sid",[](Manager&v,const std::string&n,int64_t sid,size_t shard){return v.get().find_labels_by_sid(v.name(n),sid,shard);},"name"_a,"sentence_id"_a,"shard"_a=0)
        .def("get_sentence_id",[](Manager&v,const std::string&n,size_t label,size_t shard){return v.get().get_sentence_id(v.name(n),label,shard);},"name"_a,"label"_a,"shard"_a=0)
        .def("remove",[](Manager&v,const std::string&n,size_t label,size_t shard){v.writer();v.get().remove(v.name(n),label,shard);},"name"_a,"label"_a,"shard"_a=0)
        .def("undelete",[](Manager&v,const std::string&n,size_t label,size_t shard){v.writer();v.get().undelete(v.name(n),label,shard);},"name"_a,"label"_a,"shard"_a=0)
        .def("delete_by_address",[](Manager&v,const std::string&n,int64_t address,size_t shard){v.writer();v.get().delete_byAddress(v.name(n),address,shard);},"name"_a,"address"_a,"shard"_a=0)
        .def("undelete_by_address",[](Manager&v,const std::string&n,int64_t address,size_t shard){v.writer();v.get().undelete_byAddress(v.name(n),address,shard);},"name"_a,"address"_a,"shard"_a=0)
        .def("reconstruct_sid",[](Manager&v,const std::string&n,int64_t sid){return v.get().reconstruct_sid(v.name(n),sid);})
        .def("reconstruct_label",[](Manager&v,const std::string&n,size_t label){return v.get().reconstruct_label(v.name(n),label);})
        .def("flush",[](Manager&v,const std::string&n){v.writer();v.get().flush(v.name(n));})
        .def("flush_all",[](Manager&v){v.writer();v.get().flush_all();})
        .def("merge",[](Manager&v,const std::string&n){v.writer();v.get().merge(v.name(n));})
        .def("clear",&Manager::clear,"name"_a)
        .def("__enter__",[](Manager&v)->Manager&{v.get();return v;},nb::rv_policy::reference)
        .def("__exit__",[](Manager&v,nb::handle,nb::handle,nb::handle){v.close();},
             "exc_type"_a.none(),"exc_value"_a.none(),"traceback"_a.none());
    m.def("model_quantization",[](const std::string&path){text_ok(path);return get_model_quant(path);},"path"_a);
}
