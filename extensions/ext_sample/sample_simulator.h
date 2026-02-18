#pragma once

#include "core_simulate/simulator.h"
#include <string>

namespace ext_sample {

class SampleSimulator : public mps::simulate::ISimulator {
public:
    [[nodiscard]] const std::string& GetName() const override;
    void Initialize(mps::database::Database& db) override;
    void Update(mps::database::Database& db, mps::float32 dt) override;

private:
    static const std::string kName;
};

}  // namespace ext_sample
