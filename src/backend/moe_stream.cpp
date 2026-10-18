#include "moe_stream.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
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
std::map<std::string, TensorSpec> g_specs;
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

[[noreturn]] void fail(const std::string& message) {
  g_stats.error = message;
  std::fprintf(stderr, "ame-moe: %s\n", message.c_str());
  std::fflush(stderr);
  std::abort();
}

bool read_manifest(const std::string& path, std::uint64_t model_bytes) {
  std::ifstream stream(path);
  if (!stream) {
    fail("cannot open manifest: " + path);
  }
  std::string line;
  std::uint32_t minimum_experts = UINT32_MAX;
  std::uint32_t table_id = 0;
  while (std::getline(stream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string tag;
    TensorSpec spec;
    fields >> tag >> spec.name >> spec.layer >> spec.role >> spec.type_id >> spec.type_name >>
        spec.n_expert >> spec.expert_bytes >> spec.stride >> spec.offset >> spec.nbytes;
    if (tag != "tensor" || spec.name.empty() || spec.expert_bytes == 0 || spec.n_expert == 0 ||
        spec.stride < spec.expert_bytes || spec.nbytes != spec.expert_bytes * spec.n_expert) {
      fail("invalid manifest line: " + line);
    }
    if (spec.offset > model_bytes || spec.nbytes > model_bytes - spec.offset) {
      fail("manifest extent outside model: " + spec.name);
    }
    if (spec.offset % 32 != 0 || spec.expert_bytes % 32 != 0) {
      fail("unaligned manifest extent: " + spec.name);
    }
    spec.table_id = table_id++;
    minimum_experts = std::min(minimum_experts, spec.n_expert);
    auto& entries = g_entries_by_size[spec.expert_bytes];
    entries.reserve(entries.size() + spec.n_expert);
    for (std::uint32_t expert = 0; expert < spec.n_expert; ++expert) {
      entries.push_back(
          ExpertEntry{ExpertKey{spec.table_id, expert},
                      FileOffset{spec.offset + static_cast<std::uint64_t>(expert) * spec.stride},
                      ByteCount{spec.expert_bytes}});
    }
    g_specs.emplace(spec.name, std::move(spec));
  }
  (void)minimum_experts;
  return !g_specs.empty();
}

AlignedArena& ensure_arena(std::uint64_t expert_bytes) {
  auto found = g_arenas.find(expert_bytes);
  if (found != g_arenas.end()) {
    return found->second;
  }
  const std::uint64_t bytes = expert_bytes * g_slots;
  if (bytes > g_arena_limit || g_stats.arena_bytes > g_arena_limit - bytes) {
    fail("arena request " + std::to_string(bytes) + " exceeds limit " +
         std::to_string(g_arena_limit));
  }
  void* raw = nullptr;
  if (::posix_memalign(&raw, 64, static_cast<std::size_t>(bytes)) != 0 || raw == nullptr) {
    fail("arena allocation failed");
  }
  AlignedArena arena;
  arena.memory = std::unique_ptr<std::byte, void (*)(void*)>(
      static_cast<std::byte*>(raw), [](void* pointer) { std::free(pointer); });
  arena.view = std::span<std::byte>(arena.memory.get(), static_cast<std::size_t>(bytes));
  g_stats.arena_bytes += bytes;
  auto [inserted, ok] = g_arenas.emplace(expert_bytes, std::move(arena));
  (void)ok;
  return inserted->second;
}

Cache& cache_for(std::uint64_t expert_bytes) {
  auto found = g_caches.find(expert_bytes);
  if (found != g_caches.end()) {
    return found->second;
  }
  AlignedArena& arena = ensure_arena(expert_bytes);
  auto source = FileSource::open(g_model_path);
  if (!source.ok()) {
    fail("cannot open model for cache");
  }
  auto entries = g_entries_by_size.find(expert_bytes);
  if (entries == g_entries_by_size.end() || entries->second.empty()) {
    fail("no entries for cache size class");
  }
  StoreOptions options;
  options.slot_count = g_slots;
  options.alignment = 64;
  options.external_arena = arena.view;
  auto store = Store::create(std::move(source).value(), entries->second, options);
  if (!store.ok()) {
    fail("store creation failed for cache size class");
  }
  Cache cache;
  cache.store = std::move(store).value();
  cache.arena = arena.view;
  auto [inserted, ok] = g_caches.emplace(expert_bytes, std::move(cache));
  (void)ok;
  return inserted->second;
}

bool materialize(const ggml_tensor* weights, const ggml_tensor* ids) {
  if (!g_installed || weights == nullptr || ids == nullptr || weights->name[0] == '\0' ||
      ids->data == nullptr) {
    return false;
  }
  const auto found = g_specs.find(weights->name);
  if (found == g_specs.end()) {
    return false;
  }
  const TensorSpec& spec = found->second;
  if (ids->type != GGML_TYPE_I32 || weights->type != static_cast<ggml_type>(spec.type_id) ||
      weights->ne[2] != spec.n_expert || weights->nb[2] != spec.expert_bytes) {
    fail("tensor metadata mismatch for " + spec.name);
  }
  const int64_t n_ids = ids->ne[0];
  const int64_t tokens = ids->ne[1];
  if (n_ids <= 0 || tokens <= 0 || n_ids * tokens > 1048576) {
    fail("invalid ids shape for " + spec.name);
  }
  const auto* ids_bytes = static_cast<const char*>(ids->data);
  std::array<ExpertKey, ame::residency::kMaxLeaseExperts> unique{};
  std::uint32_t unique_count = 0;
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t rank = 0; rank < n_ids; ++rank) {
      std::int32_t expert = 0;
      std::memcpy(&expert, ids_bytes + token * ids->nb[1] + rank * ids->nb[0], sizeof(expert));
      if (expert < 0 || static_cast<std::uint32_t>(expert) >= spec.n_expert) {
        fail("expert id outside tensor: " + std::to_string(expert));
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
        fail("expert union exceeds lease capacity");
      }
      unique[unique_count++] = key;
    }
  }
  if (unique_count > g_slots) {
    fail("expert union " + std::to_string(unique_count) + " exceeds " + std::to_string(g_slots) +
         " slots for " + spec.name);
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

  Cache& cache = cache_for(spec.expert_bytes);
  cache.lease = Lease{};
  ame::Cancellation cancellation;
  auto lease =
      cache.store->acquire(std::span<const ExpertKey>(unique.data(), unique_count), cancellation);
  if (!lease.ok()) {
    fail("store acquire failed for " + spec.name);
  }
  cache.lease = std::move(lease).value();

  g_remapped.resize(static_cast<std::size_t>(n_ids * tokens));
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t rank = 0; rank < n_ids; ++rank) {
      std::int32_t expert = 0;
      std::memcpy(&expert, ids_bytes + token * ids->nb[1] + rank * ids->nb[0], sizeof(expert));
      const auto slot =
          cache.lease.slot_index(ExpertKey{spec.table_id, static_cast<std::uint32_t>(expert)});
      if (!slot.has_value()) {
        fail("remap failed for expert " + std::to_string(expert));
      }
      g_remapped[static_cast<std::size_t>(token * n_ids + rank)] = static_cast<std::int32_t>(*slot);
    }
  }

  g_staged_weights = *weights;
  g_staged_weights.data = cache.arena.data();
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
  return true;
}

}  // namespace
}  // namespace ame::backend

extern "C" {
const struct ggml_tensor* ggml_moe_staged_weights(void) { return &ame::backend::g_staged_weights; }
const struct ggml_tensor* ggml_moe_staged_ids(void) { return &ame::backend::g_staged_ids; }
uint32_t ggml_moe_max_slots(void) { return ame::backend::g_slots; }
}

namespace ame::backend {

bool install_moe_stream(const std::string& model_path, const std::string& manifest_path,
                        std::uint32_t slots, std::uint64_t arena_limit_bytes) {
  uninstall_moe_stream();
  g_stats = MoeStreamStats{};
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
    g_stats.error = "manifest has no expert tensors";
    return false;
  }
  g_slots = slots;
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
  g_entries_by_size.clear();
  g_remapped.clear();
  g_model_path.clear();
  g_last_tensor.clear();
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