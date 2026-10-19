#include "moe_stream.hpp"

#include "prefill_plan.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "llama-model.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "cancellation.hpp"
#include "expert_layout.hpp"
#include "file_source.hpp"
#include "ggml-moe.h"
#include "ggml.h"
#include "store.hpp"

namespace ame::backend {
namespace {

using ame::model::ByteCount;
using ame::model::ExpertKey;
using ame::model::FileOffset;
using ame::residency::ExpertEntry;
using ame::residency::FileSource;
using ame::residency::Lease;
using ame::residency::Store;
using ame::residency::StoreOptions;

struct TensorSpec {
  std::string name;
  std::uint32_t table_id{0};
  std::uint32_t layer{0};
  std::string role;
  std::uint32_t type_id{0};
  std::string type_name;
  std::uint32_t n_expert{0};
  std::uint64_t expert_bytes{0};
  std::uint64_t stride{0};
  std::uint64_t offset{0};
  std::uint64_t nbytes{0};
};

struct AlignedArena {
  std::unique_ptr<std::byte, void (*)(void*)> memory{nullptr, nullptr};
  std::span<std::byte> view{};
};

struct Cache {
  std::unique_ptr<Store> store;
  Lease lease;
  std::span<std::byte> arena{};
};

std::string g_model_path;
const Artifact* g_artifact = nullptr;
std::map<std::string, TensorSpec> g_specs;
std::set<std::string> g_contract_tensors;
std::vector<ame::prefill::ConsumerBound> g_contract_bounds;
bool g_contract_bound = false;
std::map<std::uint64_t, std::vector<ExpertEntry>> g_entries_by_size;
std::map<std::uint64_t, AlignedArena> g_arenas;
std::map<std::uint64_t, Cache> g_caches;
std::vector<std::int32_t> g_remapped;
ggml_tensor g_staged_weights{};
ggml_tensor g_staged_ids{};
std::uint32_t g_slots = 0;
std::uint64_t g_arena_limit = 0;
bool g_installed = false;
std::string g_last_tensor;
MoeStreamStats g_stats{};
std::ofstream g_trace;
std::uint64_t g_trace_call = 0;
ame::Cancellation g_cancellation;

int fail_outcome(std::string message) {
  g_stats.error = std::move(message);
  g_stats.failed = true;
  std::fprintf(stderr, "ame-moe: %s\n", g_stats.error.c_str());
  std::fflush(stderr);
  return GGML_MOE_FAILED;
}

int cancel_outcome(std::string message) {
  g_stats.error = std::move(message);
  g_stats.cancelled = true;
  return GGML_MOE_CANCELLED;
}

bool manifest_error(std::string message) {
  g_stats.error = std::move(message);
  return false;
}

std::string layout_hash(std::string_view canonical) {
  std::uint64_t value = 0xCBF29CE484222325ULL;
  for (const char character : canonical) {
    const auto byte = static_cast<unsigned char>(character);
    value ^= byte;
    value *= 0x100000001B3ULL;
  }
  char buffer[17] = {};
  std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(value));
  return buffer;
}

std::string canonical_row(const TensorSpec& spec) {
  return spec.name + "|" + std::to_string(spec.layer) + "|" + spec.role + "|" +
         std::to_string(spec.type_id) + "|" + spec.type_name + "|" + std::to_string(spec.n_expert) +
         "|" + std::to_string(spec.expert_bytes) + "|" + std::to_string(spec.stride) + "|" +
         std::to_string(spec.offset) + "|" + std::to_string(spec.nbytes);
}

bool read_manifest(const std::string& path, std::uint64_t model_bytes) {
  std::ifstream stream(path);
  if (!stream) {
    return manifest_error("cannot open manifest: " + path);
  }
  std::string line;
  std::uint32_t version = 0;
  std::uint32_t table_id = 0;
  bool have_artifact_bytes = false;
  bool have_artifact_sha = false;
  bool have_tensor_count = false;
  bool have_layout = false;
  std::uint64_t artifact_bytes = 0;
  std::uint64_t tensor_count = 0;
  std::string declared_sha;
  std::string expected_layout;
  std::string canonical;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    if (line.front() == '#') {
      std::istringstream header(line.substr(1));
      std::string key;
      header >> key;
      if (key == "ame-expert-manifest") {
        std::uint32_t parsed = 0;
        if (!(header >> parsed) || (parsed != 1 && parsed != 2)) {
          return manifest_error("unsupported manifest version");
        }
        version = parsed;
      } else if (key == "artifact_bytes") {
        if (!(header >> artifact_bytes)) {
          return manifest_error("invalid artifact_bytes header");
        }
        have_artifact_bytes = true;
      } else if (key == "artifact_sha256") {
        if (!(header >> declared_sha)) {
          return manifest_error("invalid artifact_sha256 header");
        }
        have_artifact_sha = true;
      } else if (key == "tensor_count") {
        if (!(header >> tensor_count)) {
          return manifest_error("invalid tensor_count header");
        }
        have_tensor_count = true;
      } else if (key == "layout_fnv1a64") {
        if (!(header >> expected_layout)) {
          return manifest_error("invalid layout_fnv1a64 header");
        }
        have_layout = true;
      }
      continue;
    }
    std::istringstream fields(line);
    std::string tag;
    TensorSpec spec;
    fields >> tag >> spec.name >> spec.layer >> spec.role >> spec.type_id >> spec.type_name >>
        spec.n_expert >> spec.expert_bytes >> spec.stride >> spec.offset >> spec.nbytes;
    if (tag != "tensor" || spec.name.empty() || spec.expert_bytes == 0 || spec.n_expert == 0) {
      return manifest_error("invalid manifest row: " + line);
    }
    if (spec.n_expert > UINT64_MAX / spec.expert_bytes ||
        spec.nbytes != spec.expert_bytes * spec.n_expert) {
      return manifest_error("expert byte arithmetic overflow or mismatch: " + spec.name);
    }
    if (spec.stride < spec.expert_bytes || spec.offset > model_bytes ||
        spec.nbytes > model_bytes - spec.offset) {
      return manifest_error("manifest extent outside model: " + spec.name);
    }
    if (spec.offset % 32 != 0 || spec.expert_bytes % 32 != 0 || spec.stride % 32 != 0) {
      return manifest_error("unaligned manifest extent: " + spec.name);
    }
    spec.table_id = table_id++;
    if (!g_specs.emplace(spec.name, spec).second) {
      return manifest_error("duplicate expert tensor: " + spec.name);
    }
    if (!canonical.empty()) {
      canonical += '\n';
    }
    canonical += canonical_row(spec);
    auto& entries = g_entries_by_size[spec.expert_bytes];
    entries.reserve(entries.size() + spec.n_expert);
    for (std::uint32_t expert = 0; expert < spec.n_expert; ++expert) {
      entries.push_back(
          ExpertEntry{ExpertKey{spec.table_id, expert},
                      FileOffset{spec.offset + static_cast<std::uint64_t>(expert) * spec.stride},
                      ByteCount{spec.expert_bytes}});
    }
  }
  if (version == 0) {
    return manifest_error("missing manifest version header");
  }
  if (version == 2) {
    if (!have_artifact_bytes || !have_artifact_sha || !have_tensor_count || !have_layout) {
      return manifest_error("version 2 manifest is missing identity headers");
    }
    if (artifact_bytes != model_bytes) {
      return manifest_error("manifest artifact_bytes does not match the opened file");
    }
    if (tensor_count != g_specs.size()) {
      return manifest_error("manifest tensor_count does not match its rows");
    }
    const std::string actual_layout = layout_hash(canonical);
    if (actual_layout != expected_layout) {
      return manifest_error("manifest layout hash does not match its rows");
    }
    g_stats.manifest_identity_checked = true;
    g_stats.manifest_layout_hash = actual_layout;
  } else {
    g_stats.manifest_identity_checked = false;
    g_stats.manifest_layout_hash = layout_hash(canonical);
  }
  g_stats.manifest_version = version;
  g_stats.declared_artifact_sha256 = declared_sha;
  return !g_specs.empty();
}

AlignedArena* ensure_arena(std::uint64_t expert_bytes) {
  auto found = g_arenas.find(expert_bytes);
  if (found != g_arenas.end()) {
    return &found->second;
  }
  const std::uint64_t bytes = expert_bytes * g_slots;
  if (bytes > g_arena_limit || g_stats.arena_bytes > g_arena_limit - bytes) {
    (void)fail_outcome("arena request " + std::to_string(bytes) + " exceeds limit " +
                       std::to_string(g_arena_limit));
    return nullptr;
  }
  void* raw = nullptr;
  if (::posix_memalign(&raw, 64, static_cast<std::size_t>(bytes)) != 0 || raw == nullptr) {
    (void)fail_outcome("arena allocation failed");
    return nullptr;
  }
  AlignedArena arena;
  arena.memory = std::unique_ptr<std::byte, void (*)(void*)>(
      static_cast<std::byte*>(raw), [](void* pointer) { std::free(pointer); });
  arena.view = std::span<std::byte>(arena.memory.get(), static_cast<std::size_t>(bytes));
  g_stats.arena_bytes += bytes;
  auto [inserted, ok] = g_arenas.emplace(expert_bytes, std::move(arena));
  (void)ok;
  return &inserted->second;
}

Cache* cache_for(std::uint64_t expert_bytes) {
  auto found = g_caches.find(expert_bytes);
  if (found != g_caches.end()) {
    return &found->second;
  }
  AlignedArena* arena = ensure_arena(expert_bytes);
  if (arena == nullptr) {
    return nullptr;
  }
  auto source = g_artifact != nullptr ? FileSource::open_fd(g_artifact->fd())
                                      : FileSource::open(g_model_path);
  if (!source.ok()) {
    (void)fail_outcome("cannot open model for cache");
    return nullptr;
  }
  auto entries = g_entries_by_size.find(expert_bytes);
  if (entries == g_entries_by_size.end() || entries->second.empty()) {
    (void)fail_outcome("no entries for cache size class");
    return nullptr;
  }
  StoreOptions options;
  options.slot_count = g_slots;
  options.alignment = 64;
  options.external_arena = arena->view;
  auto store = Store::create(std::move(source).value(), entries->second, options);
  if (!store.ok()) {
    (void)fail_outcome("store creation failed for cache size class");
    return nullptr;
  }
  Cache cache;
  cache.store = std::move(store).value();
  cache.arena = arena->view;
  auto [inserted, ok] = g_caches.emplace(expert_bytes, std::move(cache));
  (void)ok;
  return &inserted->second;
}

int materialize(const ggml_tensor* weights, const ggml_tensor* ids) {
  if (!g_installed || weights == nullptr || ids == nullptr || weights->name[0] == '\0' ||
      ids->data == nullptr) {
    return GGML_MOE_PASS_THROUGH;
  }
  if (g_cancellation.requested()) {
    return cancel_outcome("cancelled before materialization");
  }
  if (g_artifact != nullptr) {
    std::string recheck_error;
    if (!g_artifact->recheck(recheck_error)) {
      return fail_outcome(recheck_error);
    }
  }
  const auto found = g_specs.find(weights->name);
  if (found == g_specs.end()) {
    const std::string unindexed(weights->name);
    if (unindexed.find("_exps") != std::string::npos ||
        unindexed.find("expert") != std::string::npos) {
      return fail_outcome("unindexed expert tensor: " + unindexed);
    }
    return GGML_MOE_PASS_THROUGH;
  }
  const TensorSpec& spec = found->second;
  if (ids->type != GGML_TYPE_I32 || weights->type != static_cast<ggml_type>(spec.type_id) ||
      weights->ne[2] != spec.n_expert || weights->nb[2] != spec.expert_bytes) {
    return fail_outcome("tensor metadata mismatch for " + spec.name);
  }
  const int64_t n_ids = ids->ne[0];
  const int64_t tokens = ids->ne[1];
  if (n_ids <= 0 || tokens <= 0 || n_ids * tokens > 1048576) {
    return fail_outcome("invalid ids shape for " + spec.name);
  }
  const auto* ids_bytes = static_cast<const char*>(ids->data);
  std::array<ExpertKey, ame::residency::kMaxLeaseExperts> unique{};
  std::uint32_t unique_count = 0;
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t rank = 0; rank < n_ids; ++rank) {
      std::int32_t expert = 0;
      const std::size_t id_offset = static_cast<std::size_t>(token) * ids->nb[1] +
                                    static_cast<std::size_t>(rank) * ids->nb[0];
      std::memcpy(&expert, ids_bytes + id_offset, sizeof(expert));
      if (expert < 0 || static_cast<std::uint32_t>(expert) >= spec.n_expert) {
        return fail_outcome("expert id outside tensor: " + std::to_string(expert));
      }
      const ExpertKey key{spec.table_id, static_cast<std::uint32_t>(expert)};
      bool seen = false;
      for (std::uint32_t index = 0; index < unique_count; ++index) {
        if (unique[index] == key) {
          seen = true;
          break;
        }
      }
      if (seen) {
        continue;
      }
      if (unique_count >= ame::residency::kMaxLeaseExperts) {
        return fail_outcome("expert union exceeds lease capacity");
      }
      unique[unique_count++] = key;
    }
  }
  if (unique_count > g_slots) {
    return fail_outcome("expert union " + std::to_string(unique_count) + " exceeds " +
                        std::to_string(g_slots) + " slots for " + spec.name);
  }
  g_stats.max_union = std::max<std::uint64_t>(g_stats.max_union, unique_count);
  if (g_trace.is_open()) {
    g_trace << g_trace_call << ' ' << spec.name << ' ' << unique_count;
    for (std::uint32_t index = 0; index < unique_count; ++index) {
      g_trace << ' ' << unique[index].expert;
    }
    g_trace << '\n';
    g_trace.flush();
  }
  ++g_trace_call;
  if (std::getenv("AME_MOE_TRACE_ONLY") != nullptr) {
    ++g_stats.materialize_calls;
    return GGML_MOE_PASS_THROUGH;
  }

  Cache* cache = cache_for(spec.expert_bytes);
  if (cache == nullptr) {
    return GGML_MOE_FAILED;
  }
  cache->lease = Lease{};
  auto lease = cache->store->acquire(std::span<const ExpertKey>(unique.data(), unique_count),
                                     g_cancellation);
  if (!lease.ok()) {
    if (lease.error() == ame::residency::StoreError::Cancelled || g_cancellation.requested()) {
      return cancel_outcome("cancelled during materialization: " + spec.name);
    }
    return fail_outcome("store acquire failed for " + spec.name);
  }
  cache->lease = std::move(lease).value();
#ifdef AME_TEST_BREAK_MATERIALIZER
  // Deliberately corrupt one staged byte of the first selected expert.
  if (unique_count != 0) {
    const auto broken_slot = cache->lease.slot_index(unique[0]);
    if (broken_slot.has_value()) {
      const std::size_t offset =
          static_cast<std::size_t>(*broken_slot) * static_cast<std::size_t>(spec.expert_bytes);
      if (offset < cache->arena.size()) {
        std::memset(cache->arena.data() + offset, 0xFF,
                    static_cast<std::size_t>(spec.expert_bytes));
      }
    }
  }
#endif

  g_remapped.resize(static_cast<std::size_t>(n_ids * tokens));
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t rank = 0; rank < n_ids; ++rank) {
      std::int32_t expert = 0;
      const std::size_t id_offset = static_cast<std::size_t>(token) * ids->nb[1] +
                                    static_cast<std::size_t>(rank) * ids->nb[0];
      std::memcpy(&expert, ids_bytes + id_offset, sizeof(expert));
      const auto slot =
          cache->lease.slot_index(ExpertKey{spec.table_id, static_cast<std::uint32_t>(expert)});
      if (!slot.has_value()) {
        cache->lease = Lease{};
        return fail_outcome("remap failed for expert " + std::to_string(expert));
      }
      g_remapped[static_cast<std::size_t>(token * n_ids + rank)] = static_cast<std::int32_t>(*slot);
    }
  }

  g_staged_weights = *weights;
  g_staged_weights.data = cache->arena.data();
  g_staged_weights.ne[2] = g_slots;
  g_staged_weights.nb[2] = spec.expert_bytes;
  g_staged_weights.nb[3] = spec.expert_bytes * g_slots;
  g_staged_ids = *ids;
  g_staged_ids.data = g_remapped.data();
  g_staged_ids.nb[0] = sizeof(std::int32_t);
  g_staged_ids.nb[1] = sizeof(std::int32_t) * static_cast<std::size_t>(n_ids);
  if (g_last_tensor != spec.name) {
    ++g_stats.tensor_switches;
    g_last_tensor = spec.name;
  }
  ++g_stats.materialize_calls;
  return GGML_MOE_STAGED;
}

bool bind_model_contract_impl(const llama_model* model, std::string& error) {
  if (model == nullptr || g_artifact == nullptr) {
    error = "contract requires a loaded model and verified artifact";
    return false;
  }
  if (g_specs.empty()) {
    error = "contract requires a populated manifest";
    return false;
  }
  const auto consumers = llama_internal_get_expert_consumers(model);
  g_contract_bounds.clear();
  std::set<std::string> seen;
  std::set<std::string> routed_manifest;
  std::uint64_t routed = 0;
  std::uint64_t shared = 0;
  const auto& tensors = g_artifact->tensors();
  for (const auto& consumer : consumers) {
    if (consumer.tensor == nullptr) {
      error = "null expert consumer";
      return false;
    }
    const std::string name = consumer.tensor->name;
    if (!seen.insert(name).second) {
      error = "duplicate expert consumer: " + name;
      return false;
    }
    std::uint64_t companion_count = 0;
    for (const auto& [tensor_name, info] : tensors) {
      (void)info;
      if (tensor_name.rfind(name + ".", 0) == 0) {
        const std::string suffix = tensor_name.substr(name.size() + 1);
        if (suffix != "scale" && suffix != "input_scale") {
          error = "unowned companion tensor: " + tensor_name;
          return false;
        }
        ++companion_count;
      }
    }
    if (consumer.shared) {
      ++shared;
      continue;
    }
    if (consumer.max_selected == 0 ||
        consumer.max_selected > static_cast<uint32_t>(consumer.tensor->ne[2])) {
      error = "unsupported source selection bound: " + name;
      return false;
    }
    ++routed;
    routed_manifest.insert(name);
    const auto spec = g_specs.find(name);
    if (spec == g_specs.end()) {
      error = "source consumer missing from manifest: " + name;
      return false;
    }
    const auto& s = spec->second;
    g_contract_bounds.push_back({static_cast<std::uint32_t>(consumer.tensor->ne[2]),
                                 consumer.max_selected, g_slots, ame::residency::kMaxLeaseExperts});
    if (consumer.tensor->type != s.type_id ||
        consumer.tensor->ne[2] != static_cast<int64_t>(s.n_expert) ||
        consumer.tensor->nb[2] != s.expert_bytes || ggml_nbytes(consumer.tensor) != s.nbytes) {
      error = "consumer layout mismatch: " + name;
      return false;
    }
  }
  for (const auto& [name, spec] : g_specs) {
    (void)spec;
    if (routed_manifest.find(name) == routed_manifest.end()) {
      error = "manifest routed tensor has no source consumer: " + name;
      return false;
    }
  }
  if (routed == 0) {
    error = "no routed expert consumers";
    return false;
  }
  g_contract_tensors = std::move(routed_manifest);
  g_contract_bound = true;
  g_stats.contract_routed = routed;
  g_stats.contract_shared = shared;
  return true;
}

bool validate_artifact_layout() {
  if (g_artifact == nullptr) {
    return manifest_error("no verified artifact");
  }
  if (g_specs.empty()) {
    return manifest_error("manifest has no expert tensors");
  }
  if (g_stats.manifest_version == 2 && !g_stats.declared_artifact_sha256.empty() &&
      g_stats.declared_artifact_sha256 != "-" &&
      g_stats.declared_artifact_sha256 != g_artifact->sha256()) {
    return manifest_error("manifest SHA does not match the verified artifact");
  }
  for (const auto& [name, spec] : g_specs) {
    if (name.find("_exps") == std::string::npos) {
      return manifest_error("manifest contains a non-routed tensor: " + name);
    }
    const TensorInfo* info = g_artifact->find(name);
    if (info == nullptr) {
      return manifest_error("manifest tensor is absent from GGUF: " + name);
    }
    if (info->type != spec.type_id) {
      return manifest_error("GGUF type mismatch for " + name);
    }
    if (info->ne[2] != static_cast<std::int64_t>(spec.n_expert) || info->ne[3] != 1 ||
        info->ne[0] <= 0 || info->ne[1] <= 0) {
      return manifest_error("GGUF shape mismatch for " + name);
    }
    if (info->absolute_offset != spec.offset || info->nbytes != spec.nbytes) {
      return manifest_error("GGUF extent mismatch for " + name);
    }
    if (spec.stride != spec.expert_bytes) {
      return manifest_error("non-contiguous expert stride is unsupported for " + name);
    }
    const std::uint64_t row_size = static_cast<std::uint64_t>(
        ggml_row_size(static_cast<ggml_type>(spec.type_id), info->ne[0]));
    if (row_size == 0 || static_cast<std::uint64_t>(info->ne[1]) > UINT64_MAX / row_size ||
        spec.expert_bytes != row_size * static_cast<std::uint64_t>(info->ne[1])) {
      return manifest_error("quantization row arithmetic mismatch for " + name);
    }
  }
  for (const auto& [name, info] : g_artifact->tensors()) {
    (void)info;
    const bool is_routed = name.find("_exps") != std::string::npos && name.size() >= 7 &&
                           name.compare(name.size() - 7, 7, ".weight") == 0;
    if (is_routed && g_specs.find(name) == g_specs.end()) {
      return manifest_error("routed GGUF tensor is missing from manifest: " + name);
    }
  }
  g_stats.artifact_verified = true;
  g_stats.verified_artifact_sha256 = g_artifact->sha256();
  g_stats.artifact_hash_seconds = g_artifact->hash_seconds();
  return true;
}

}  // namespace

bool bind_model_contract(const llama_model* model, std::string& error) {
  return bind_model_contract_impl(model, error);
}

bool plan_model_prefill(std::uint32_t batch_size, std::uint32_t lease_limit,
                        std::uint32_t& unit_tokens, std::uint32_t& expert_count,
                        std::uint32_t& max_selected, std::string& error) {
  if (!g_contract_bound || g_contract_bounds.empty()) {
    error = "no bound consumer contract";
    return false;
  }
  std::vector<ame::prefill::ConsumerBound> bounds = g_contract_bounds;
  for (auto& bound : bounds) {
    bound.lease_limit = lease_limit;
  }
  const auto plan = ame::prefill::make_contract_plan(bounds, batch_size);
  if (!plan.ok()) {
    switch (plan.error()) {
      case ame::prefill::PlanError::InvalidExpertCount:
        error = "consumer expert cardinality is invalid";
        break;
      case ame::prefill::PlanError::InvalidExpertUsed:
        error = "consumer selection bound is missing or invalid";
        break;
      case ame::prefill::PlanError::InvalidSlots:
      case ame::prefill::PlanError::InvalidBatch:
        error = "invalid prefill admission shape";
        break;
      case ame::prefill::PlanError::Capacity:
        error = "consumer pool capacity is below its selection bound";
        break;
      case ame::prefill::PlanError::LeaseCapacity:
        error = "consumer lease limit is below its selection bound";
        break;
    }
    return false;
  }
  unit_tokens = plan.value().unit_tokens;
  expert_count = plan.value().expert_count;
  max_selected = plan.value().expert_used;
  return true;
}
}  // namespace ame::backend

extern "C" {
const struct ggml_tensor* ggml_moe_staged_weights(void) { return &ame::backend::g_staged_weights; }
const struct ggml_tensor* ggml_moe_staged_ids(void) { return &ame::backend::g_staged_ids; }
uint32_t ggml_moe_max_slots(void) { return ame::backend::g_slots; }
}

namespace ame::backend {

bool install_moe_stream(const Artifact& artifact, const std::string& manifest_path,
                        std::uint32_t slots, std::uint64_t arena_limit_bytes) {
  uninstall_moe_stream();
  g_stats = MoeStreamStats{};
  if (slots == 0 || arena_limit_bytes == 0) {
    g_stats.error = "invalid stream options";
    return false;
  }
  g_artifact = &artifact;
  std::string recheck_error;
  if (!artifact.recheck(recheck_error)) {
    g_stats.error = recheck_error;
    return false;
  }
  g_model_path.clear();
  g_arena_limit = arena_limit_bytes;
  if (!read_manifest(manifest_path, artifact.size())) {
    if (g_stats.error.empty()) {
      g_stats.error = "manifest has no expert tensors";
    }
    return false;
  }
  if (!validate_artifact_layout()) {
    return false;
  }
  g_slots = slots;
  g_cancellation.reset();
  g_stats.slots = g_slots;
  g_trace_call = 0;
  if (const char* trace_path = std::getenv("AME_MOE_TRACE");
      trace_path != nullptr && trace_path[0] != '\0') {
    g_trace.open(trace_path, std::ios::trunc);
    if (!g_trace) {
      g_stats.error = "cannot open routing trace";
      return false;
    }
    g_trace << "# call tensor n_experts experts...\n";
    g_trace.flush();
  }
  ggml_moe_set_materializer(materialize);
  g_installed = true;
  return true;
}

bool install_moe_stream(const std::string& model_path, const std::string& manifest_path,
                        std::uint32_t slots, std::uint64_t arena_limit_bytes) {
  if (std::getenv("AME_ALLOW_UNVERIFIED_ARTIFACT") == nullptr) {
    g_stats.error =
        "unverified artifact install refused; set AME_ALLOW_UNVERIFIED_ARTIFACT=1 only for "
        "diagnostics";
    return false;
  }
  uninstall_moe_stream();
  g_stats = MoeStreamStats{};
  g_artifact = nullptr;
  std::error_code size_error;
  const auto model_bytes = std::filesystem::file_size(model_path, size_error);
  if (size_error) {
    g_stats.error = "cannot stat model";
    return false;
  }
  if (slots == 0 || arena_limit_bytes == 0) {
    g_stats.error = "invalid stream options";
    return false;
  }
  g_model_path = model_path;
  g_arena_limit = arena_limit_bytes;
  if (!read_manifest(manifest_path, model_bytes)) {
    if (g_stats.error.empty()) {
      g_stats.error = "manifest has no expert tensors";
    }
    return false;
  }
  g_slots = slots;
  g_cancellation.reset();
  g_stats.slots = g_slots;
  g_trace_call = 0;
  if (const char* trace_path = std::getenv("AME_MOE_TRACE");
      trace_path != nullptr && trace_path[0] != '\0') {
    g_trace.open(trace_path, std::ios::trunc);
    if (!g_trace) {
      g_stats.error = "cannot open routing trace";
      return false;
    }
    g_trace << "# call tensor n_experts experts...\n";
    g_trace.flush();
  }
  ggml_moe_set_materializer(materialize);
  g_installed = true;
  return true;
}

void request_moe_stream_cancel() { g_cancellation.request(); }

bool moe_stream_failed() { return g_stats.failed; }

bool moe_stream_cancelled() { return g_stats.cancelled; }

std::string moe_stream_error() { return g_stats.error; }

void uninstall_moe_stream() {
  ggml_moe_set_materializer(nullptr);
  if (g_trace.is_open()) {
    g_trace.close();
  }
  g_trace_call = 0;
  g_installed = false;
  for (auto& [key, cache] : g_caches) {
    (void)key;
    cache.lease = Lease{};
    cache.store.reset();
  }
  g_caches.clear();
  g_arenas.clear();
  g_specs.clear();
  g_contract_tensors.clear();
  g_contract_bounds.clear();
  g_contract_bound = false;
  g_entries_by_size.clear();
  g_remapped.clear();
  g_model_path.clear();
  g_last_tensor.clear();
  g_artifact = nullptr;
  g_cancellation.reset();
}

MoeStreamStats moe_stream_stats() {
  MoeStreamStats result = g_stats;
  for (const auto& [key, cache] : g_caches) {
    (void)key;
    if (cache.store) {
      const auto store_stats = cache.store->stats();
      result.hits += store_stats.hits;
      result.misses += store_stats.misses;
      result.evictions += store_stats.evictions;
      result.reads += store_stats.read_calls;
      result.read_bytes += store_stats.read_bytes;
      result.read_seconds += store_stats.read_seconds;
    }
  }
  return result;
}

}  // namespace ame::backend