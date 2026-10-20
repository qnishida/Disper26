#include "../src/dispersion_context.hpp"
#include <stdexcept>
#include <iostream>

int main() {
    int calls = 0;
    bool fail = true;
    auto cache = disper26::detail::make_dispersion_mesh_cache<int>(3, [&](int level) {
        ++calls;
        if (level == 2 && fail) throw std::runtime_error("construction failure");
        return level * 10;
    });
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("cache contract violated"); };
    check(calls == 0);
    const int* first = &cache(0);
    check(cache(3) == 30 && cache(0) == 0 && calls == 2 && first == &cache(0));
    bool failed = false;
    try { cache(2); } catch (const std::runtime_error&) { failed = true; }
    check(failed);
    fail = false;
    check(cache(2) == 20 && calls == 4);
    for (int level : {-1, 4}) {
        bool rejected = false;
        try { cache(level); } catch (const std::out_of_range&) { rejected = true; }
        check(rejected);
    }
    auto other = disper26::detail::make_dispersion_mesh_cache<int>(3, [](int n) { return n + 1; });
    check(other(0) == 1 && cache(0) == 0 && calls == 4);
    std::cout << "Lazy levels, stable references, retry and call isolation passed\n";
}
