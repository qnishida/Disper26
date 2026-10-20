#pragma once
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace disper26 { namespace detail {

// One immutable model/refinement policy per cache, owned by one calculation.
// Fixed capacity keeps returned references valid when another level is built.
// Failed construction leaves the level empty and may be retried.
template<class Mesh, class Builder>
class DispersionMeshCache {
    Builder builder_;
    std::vector<std::optional<Mesh>> levels_;
public:
    DispersionMeshCache(int max_level, Builder builder)
        : builder_(std::move(builder)) {
        if (max_level < 0) throw std::invalid_argument("max_level must be nonnegative");
        levels_.resize(static_cast<size_t>(max_level) + 1);
    }
    const Mesh& operator()(int level) {
        auto& entry = levels_.at(static_cast<size_t>(level));
        if (!entry) entry.emplace(builder_(level));
        return *entry;
    }
};

template<class Mesh, class Builder>
auto make_dispersion_mesh_cache(int max_level, Builder builder) {
    return DispersionMeshCache<Mesh, Builder>(max_level, std::move(builder));
}

}} // namespace disper26::detail
