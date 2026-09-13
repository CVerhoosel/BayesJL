#!/usr/bin/env julia
using Coverage
using Printf

# Collect coverage for src (only files that produced .cov files)
cov = process_folder("src")

# Build a set of already covered file paths (normalized)
covered = Set(fc.filename for fc in cov)

# Enumerate all .jl source files under src
all_files = String[]
for (root, _, files) in walkdir("src")
    for f in files
        endswith(f, ".jl") || continue
        push!(all_files, joinpath(root, f))
    end
end

# Add zero-coverage entries for files not touched by tests so they appear in report
for f in all_files
    if !(f in covered)
        nlines = countlines(f)
        push!(cov, FileCoverage(f, fill(0, nlines)))
    end
end

LCOV.writefile("coverage.info", cov)

covered_lines, total_lines = get_summary(cov)
@printf("Full project coverage: %d/%d lines (%.1f%%)\n", covered_lines, total_lines, 100 * covered_lines / total_lines)
