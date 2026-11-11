#ifndef OT_PHASE_METRICS_H__
#define OT_PHASE_METRICS_H__

#include <atomic>
#include <chrono>
#include <cstdint>

namespace sci {

extern bool ot_phase_metrics_enabled;
extern std::atomic<uint64_t> ot_phase_preprocess_calls;
extern std::atomic<uint64_t> ot_phase_preprocess_time_us;
extern std::atomic<uint64_t> ot_phase_preprocess_comm_bytes;
extern std::atomic<uint64_t> ot_phase_fallback_calls;
extern std::atomic<uint64_t> ot_phase_fallback_ots;
extern std::atomic<uint64_t> ot_phase_preprocess_ots;

inline void ResetOTPhaseMetrics() {
  ot_phase_preprocess_calls.store(0);
  ot_phase_preprocess_time_us.store(0);
  ot_phase_preprocess_comm_bytes.store(0);
  ot_phase_fallback_calls.store(0);
  ot_phase_fallback_ots.store(0);
  ot_phase_preprocess_ots.store(0);
}

inline void RecordOTPhaseFallback(int length) {
  if (!ot_phase_metrics_enabled) {
    return;
  }
  ot_phase_fallback_calls.fetch_add(1);
  if (length > 0) {
    ot_phase_fallback_ots.fetch_add(static_cast<uint64_t>(length));
  }
}

template <typename IO> class OTPhaseMetricScope {
public:
  OTPhaseMetricScope(IO *io, int length) : io_(io), length_(length) {
    enabled_ = ot_phase_metrics_enabled;
    if (!enabled_) {
      return;
    }
    comm_before_ = io_ == nullptr ? 0 : static_cast<uint64_t>(io_->counter);
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~OTPhaseMetricScope() {
    if (!enabled_) {
      return;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count();
    const uint64_t comm_after =
        io_ == nullptr ? comm_before_ : static_cast<uint64_t>(io_->counter);
    ot_phase_preprocess_calls.fetch_add(1);
    ot_phase_preprocess_time_us.fetch_add(
        elapsed_us < 0 ? 0 : static_cast<uint64_t>(elapsed_us));
    ot_phase_preprocess_comm_bytes.fetch_add(
        comm_after >= comm_before_ ? comm_after - comm_before_ : 0);
    if (length_ > 0) {
      ot_phase_preprocess_ots.fetch_add(static_cast<uint64_t>(length_));
    }
  }

private:
  IO *io_ = nullptr;
  int length_ = 0;
  bool enabled_ = false;
  uint64_t comm_before_ = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

} // namespace sci

#endif // OT_PHASE_METRICS_H__
