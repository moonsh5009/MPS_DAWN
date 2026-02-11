# core_database - Database management and transaction system

# Create core_database library
add_library(core_database STATIC
    # Source files will be added here
    # database.cpp
    # transaction.cpp
    # query.cpp
    # storage.cpp
    # cache.cpp
)

# Set target properties
set_target_properties(core_database PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(core_database PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(core_database PUBLIC
    mps::core_util
)

# Create alias for namespace-style usage
add_library(mps::core_database ALIAS core_database)
