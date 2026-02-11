#include "core_util/logger.h"
#include "core_util/timer.h"
#include "core_util/math.h"
#include "core_util/types.h"

using namespace mps::util;

int main() {
    // Test Logger
    LogInfo("MPS_DAWN started successfully!");
    LogInfo("Dawn library linked successfully!");

    // Test Timer
    Timer timer;
    timer.Start();

    // Test Math
    vec3 pos(1.0f, 2.0f, 3.0f);
    vec3 normalized = Normalize(pos);

    LogInfo("Position: (", pos.x, ", ", pos.y, ", ", pos.z, ")");
    LogInfo("Normalized: (", normalized.x, ", ", normalized.y, ", ", normalized.z, ")");
    LogInfo("Length: ", Length(pos));

    // Test Types
    uint32 count = 100;
    float32 value = 3.14f;

    LogInfo("Count: ", count, ", Value: ", value);

    timer.Stop();
    LogInfo("Execution time: ", timer.GetElapsedMilliseconds(), " ms");

    return 0;
}
