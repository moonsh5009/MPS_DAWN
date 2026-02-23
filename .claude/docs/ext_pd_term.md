# ext_pd_term

> Shared PD constraint term implementations (spring + area) with both Chebyshev and ADMM support.

## Module Structure

```
extensions/ext_pd_term/
├── CMakeLists.txt                       # STATIC library → mps::ext_pd_term (depends: mps::core_system, mps::ext_dynamics, mps::ext_newton)
├── pd_term_extension.h / .cpp           # PDTermExtension (IExtension) — registers PD term providers
├── pd_spring_term.h / .cpp              # PDSpringTerm (IProjectiveTerm) — spring distance constraint (Chebyshev + ADMM)
├── pd_spring_term_provider.h / .cpp     # PDSpringTermProvider (IProjectiveTermProvider → PDSpringTerm)
├── pd_area_term.h / .cpp                # PDAreaTerm (IProjectiveTerm) — ARAP area constraint (Chebyshev + ADMM)
└── pd_area_term_provider.h / .cpp       # PDAreaTermProvider (IProjectiveTermProvider → PDAreaTerm)
```

## Key Types

| Type | Header | Description |
|------|--------|-------------|
| `PDTermExtension` | `pd_term_extension.h` | IExtension: registers PDSpringTermProvider + PDAreaTermProvider |
| `ADMMSpringParams` | `pd_spring_term.h` | GPU uniform: `alignas(16) {penalty_weight, stiffness}` — 16 bytes (ADMM ρ + physical k) |
| `PDSpringTerm` | `pd_spring_term.h` | IProjectiveTerm: spring distance constraint (LHS + ProjectRHS + ADMM methods) |
| `PDSpringTermProvider` | `pd_spring_term_provider.h` | IProjectiveTermProvider: creates PDSpringTerm from SpringConstraintData |
| `PDAreaTerm` | `pd_area_term.h` | IProjectiveTerm: ARAP area constraint (LHS + ProjectRHS + ADMM methods) |
| `PDAreaTermProvider` | `pd_area_term_provider.h` | IProjectiveTermProvider: creates PDAreaTerm from AreaConstraintData |

## API

### PDTermExtension

```cpp
explicit PDTermExtension(mps::system::System& system);
const std::string& GetName() const override;       // "ext_pd_term"
void Register(mps::system::System& system) override;
```

`Register()` does:
1. `RegisterPDTermProvider(SpringConstraintData → PDSpringTermProvider)`
2. `RegisterPDTermProvider(AreaConstraintData → PDAreaTermProvider)`

### PDSpringTerm

```cpp
PDSpringTerm(const std::vector<ext_dynamics::SpringEdge>& edges, float32 stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) override;
void AssembleLHS(WGPUCommandEncoder encoder) override;
void ProjectRHS(WGPUCommandEncoder encoder) override;   // fused local projection + RHS

// ADMM methods (z/u buffers per edge)
void InitializeADMM(const PDAssemblyContext& ctx) override;
void AssembleADMMLHS(WGPUCommandEncoder encoder) override;   // LHS with ρ (penalty) instead of k (stiffness)
void ProjectLocal(WGPUCommandEncoder encoder) override;      // z = proximal projection with ρ and k
void AssembleADMMRHS(WGPUCommandEncoder encoder) override;   // rhs += ρ*S^T*(z-u)
void UpdateDual(WGPUCommandEncoder encoder) override;        // u += S*q - z
void ResetDual(WGPUCommandEncoder encoder) override;         // z=S*s, u=0
void Shutdown() override;
```

### PDAreaTerm

```cpp
PDAreaTerm(const std::vector<ext_dynamics::AreaTriangle>& triangles, float32 stretch_stiffness, float32 shear_stiffness);
const std::string& GetName() const override;
void DeclareSparsity(SparsityBuilder& builder) override;
void Initialize(const SparsityBuilder& sparsity, const PDAssemblyContext& ctx) override;
void AssembleLHS(WGPUCommandEncoder encoder) override;
void ProjectRHS(WGPUCommandEncoder encoder) override;   // fused local projection + RHS

// ADMM methods (z/u: 2×vec4f per face for 3x2 rotation columns)
void InitializeADMM(const PDAssemblyContext& ctx) override;
void ProjectLocal(WGPUCommandEncoder encoder) override;      // z = SVD rotation of (F + u)
void AssembleADMMRHS(WGPUCommandEncoder encoder) override;   // rhs += w*S^T*(z-u)
void UpdateDual(WGPUCommandEncoder encoder) override;        // u += F - z
void ResetDual(WGPUCommandEncoder encoder) override;         // z=F(s), u=0
void Shutdown() override;
```

### PDSpringTermProvider / PDAreaTermProvider

```cpp
std::string_view GetTermName() const override;
bool HasConfig(const Database& db, Entity entity) const override;
std::unique_ptr<IProjectiveTerm> CreateTerm(
    const Database& db, Entity entity, uint32 node_count) override;
void DeclareTopology(uint32& out_edge_count, uint32& out_face_count) override;
void QueryTopology(const Database& db, Entity entity,
                   uint32& out_edge_count, uint32& out_face_count) const override;
```

## Shaders (`assets/shaders/ext_pd_term/`)

| Shader | Used by | Purpose |
|--------|---------|---------|
| `pd_spring_lhs.wgsl` | PDSpringTerm | Spring LHS assembly (w * S^T * S) |
| `pd_spring_project_rhs.wgsl` | PDSpringTerm | Fused spring local projection + RHS assembly |
| `pd_area_lhs.wgsl` | PDAreaTerm | Area LHS assembly |
| `pd_area_project_rhs.wgsl` | PDAreaTerm | Fused area ARAP local projection + RHS assembly |
| `admm_spring_project.wgsl` | PDSpringTerm | z = rest_len * normalize(S*q + u) |
| `admm_spring_rhs.wgsl` | PDSpringTerm | rhs += w*S^T*(z-u) |
| `admm_spring_dual.wgsl` | PDSpringTerm | u += S*q - z |
| `admm_spring_reset.wgsl` | PDSpringTerm | z=S*s, u=0 |
| `admm_area_project.wgsl` | PDAreaTerm | z = SVD rotation of (F + u) |
| `admm_area_rhs.wgsl` | PDAreaTerm | rhs += w*S^T*(z-u) |
| `admm_area_dual.wgsl` | PDAreaTerm | u += F - z |
| `admm_area_reset.wgsl` | PDAreaTerm | z=F(s), u=0 |

All shaders import `core_simulate/header/solver_params.wgsl` (root-relative). Shaders that use physics params also import `core_simulate/header/physics_params.wgsl`.
