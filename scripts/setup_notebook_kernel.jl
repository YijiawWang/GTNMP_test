#!/usr/bin/env julia
# One-shot setup for running demo/*.ipynb in Cursor / VS Code.
#
# Usage (from TNMP_test/):
#   julia --project=. scripts/setup_notebook_kernel.jl

using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)
Pkg.instantiate()
Pkg.precompile()

using IJulia

kernel_dir = joinpath(homedir(), ".local", "share", "jupyter", "kernels")
primary_dir = joinpath(kernel_dir, "julia-tnmp_test-1.11")
if !isfile(joinpath(primary_dir, "kernel.json"))
    IJulia.installkernel("Julia TNMP_test 1.11")
end

primary = joinpath(primary_dir, "kernel.json")
alias_dir = joinpath(kernel_dir, "julia")
mkpath(alias_dir)

alias_json = replace(read(primary, String), "Julia TNMP_test 1.11" => "Julia (TNMP_test)")
write(joinpath(alias_dir, "kernel.json"), alias_json)

# Drop duplicate specs created by repeated installkernel calls.
for d in readdir(kernel_dir)
    startswith(d, "julia-tnmp_test-1.11-") && rm(joinpath(kernel_dir, d); recursive = true)
end

println("Notebook kernel ready.")
println("  primary : julia-tnmp_test-1.11  (Julia TNMP_test 1.11)")
println("  alias   : julia                 (Julia (TNMP_test))")
println("In the notebook, pick kernel:  Julia (TNMP_test)")
