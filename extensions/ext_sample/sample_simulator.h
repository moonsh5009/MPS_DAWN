#pragma once

#include "core_simulate/simulator.h"
#include <string>

namespace mps { namespace system { class System; } }

namespace ext_sample {

class SampleSimulator : public mps::simulate::ISimulator {
public:
    explicit SampleSimulator(mps::system::System& system);

    [[nodiscard]] const std::string& GetName() const override;
    void Initialize() override;
    void Update() override;

private:
    mps::system::System& system_;
    static const std::string kName;
};

}  // namespace ext_sample
