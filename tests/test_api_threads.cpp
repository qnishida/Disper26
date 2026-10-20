#include "api.hpp"
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
// All workers enter each numerical calculation together. A worker that catches
// an exception still joins subsequent barriers, so failures cannot deadlock.
class Barrier {
    std::mutex mutex_;
    std::condition_variable condition_;
    const unsigned workers_;
    unsigned remaining_, generation_=0;
public:
    explicit Barrier(unsigned workers):workers_(workers),remaining_(workers) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const unsigned generation=generation_;
        if(--remaining_==0) {
            remaining_=workers_;
            ++generation_;
            condition_.notify_all();
        } else {
            condition_.wait(lock,[&]{return generation_!=generation;});
        }
    }
};
void same_result(const disper26::api::DerivativeResult& result,
                 const disper26::api::DerivativeResult& reference) {
    // Thread-local storage changes neither arithmetic nor reduction order.
    // Identical inputs must therefore preserve the serial result exactly.
    if(result.rows!=reference.rows || result.cols!=reference.cols ||
       result.hessian!=reference.hessian || result.gradients!=reference.gradients)
        throw std::runtime_error("Concurrent Full AD Hessian differs from serial result");
    if(result.parameters.size()!=reference.parameters.size())
        throw std::runtime_error("Concurrent Full AD parameter dimensions changed");
}
}

int main() {
    namespace api=disper26::api;
    const auto upper=api::create_isotropic_parameters(3.0,5.2,2.5);
    const auto middle=api::create_isotropic_parameters(3.4,5.9,2.7);
    const auto lower=api::create_isotropic_parameters(3.8,6.5,2.9);
    const auto halfspace=api::create_isotropic_parameters(4.5,8.0,3.3);
    const auto small=api::create_pancake_model({upper},{10.0},halfspace);
    const auto large=api::create_pancake_model({upper,middle,lower},{5.0,8.0,10.0},halfspace);
    api::Options options;
    options.method="ad";
    std::vector<api::ModeSolution> solutions;
    for(const auto& wave:{"Rayleigh","Love"}) {
        solutions.push_back(api::compute_dispersion(small,20.0,wave));
        solutions.push_back(api::compute_dispersion(large,20.0,wave));
    }
    std::vector<api::DerivativeResult> references;
    for(const auto& solution:solutions)
        references.push_back(api::compute_hessian(solution,"stiffness",options));
    if(references[0].rows!=12 || references[1].rows!=24)
        throw std::runtime_error("Test requires different active Full AD dimensions");
    const unsigned count=static_cast<unsigned>(solutions.size());
    Barrier barrier(count);
    std::vector<std::exception_ptr> errors(count);
    std::vector<std::thread> threads;
    for(unsigned worker=0;worker<count;++worker) {
        threads.emplace_back([&,worker] {
            for(unsigned repeat=0;repeat<6;++repeat) {
                barrier.wait();
                if(errors[worker]) continue;
                try {
                    // Each worker also alternates dimensions across calls.
                    const unsigned index=worker^(repeat%2);
                    const auto result=api::compute_hessian(solutions[index],"stiffness",options);
                    same_result(result,references[index]);
                } catch(...) {
                    errors[worker]=std::current_exception();
                }
            }
        });
    }
    for(auto& thread:threads) thread.join();
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
    // A main-thread call must also remain independent of worker dimensions.
    same_result(api::compute_hessian(solutions[0],"stiffness",options),references[0]);
    std::cout << "Concurrent Full AD Hessians preserve serial results across dimensions and waves\n";
}
