#!/bin/bash
# Rebuild the full ahead-of-time default.metallib from the resolved mlx-swift checkout.
#
# MLX loads "default.metallib" from the working directory as its last fallback
# (METAL_PATH in mlx-swift's Package.swift). SwiftPM only ships the JIT-mode subset
# in mlx-swift_Cmlx.bundle, so the full library must be built by hand and kept in
# sync with the mlx-swift version in Package.resolved. A stale copy produces garbage
# output.
#
# Usage:
#   scripts/build-metallib.sh            # build, verify, install to ./ and Tools/
#   scripts/build-metallib.sh --no-install
#
# Requires: cmake, Xcode with the Metal toolchain, swiftc, python3.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

MLX_SRC="$ROOT/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
if [[ ! -d "$MLX_SRC/mlx/backend/metal/kernels" ]]; then
    echo "mlx source not found at $MLX_SRC. Run 'swift package resolve' first." >&2
    exit 1
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/build-metallib.XXXXXX")
BUILD="$WORK/build"
AIR="$WORK/air"
OUT="$WORK/default.metallib"
mkdir -p "$AIR"
echo "Work dir: $WORK"

# 1. Let mlx's CMake generate the exact per-kernel compile commands (flags, includes,
#    kernel list). We do not run its mlx-metallib target directly: on Xcode 27 the
#    Metal compiler's default language mode rejects steel/gemm/mma.h.
cmake -S "$MLX_SRC" -B "$BUILD" \
    -DMLX_BUILD_METAL=ON -DMLX_METAL_JIT=OFF \
    -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_BUILD_TYPE=Release \
    > "$WORK/cmake-configure.log" 2>&1

BUILD_MAKE="$BUILD/mlx/backend/metal/kernels/CMakeFiles/mlx-metallib.dir/build.make"

# 2. Compile every kernel. Regular kernels get -std=metal3.2; the NAX kernels (any name
#    with "_nax", e.g. quantized_nax or gated_delta_update_nax_vjp) need Metal 4 tensor
#    ops, so they get -std=metal4.0.
python3 - "$BUILD_MAKE" "$AIR" <<'EOF'
import re, subprocess, sys, concurrent.futures as cf

build_make, air_dir = sys.argv[1], sys.argv[2]
cmds = []
for line in open(build_make):
    if "metal -x metal" not in line:
        continue
    line = re.sub(r"^\s*cd [^&]*&& ", "", line.strip())
    name = re.search(r"-o (\w+)\.air", line).group(1)
    line = line.replace(f"-o {name}.air", f"-o {air_dir}/{name}.air")
    std = "-std=metal4.0" if "_nax" in name else "-std=metal3.2"
    cmds.append((name, line.replace("-x metal", f"-x metal {std}", 1)))

def run(item):
    name, cmd = item
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if "error:" in l]
    return name, r.returncode, errs[:3]

failed = 0
with cf.ThreadPoolExecutor(8) as ex:
    for name, rc, errs in ex.map(run, cmds):
        if rc != 0:
            failed += 1
            print(f"FAIL {name}")
            for e in errs:
                print("   ", e[:200])
print(f"compiled {len(cmds) - failed}/{len(cmds)} kernels")
sys.exit(1 if failed else 0)
EOF

# 3. Link.
xcrun -sdk macosx metallib "$AIR"/*.air -o "$OUT"

# 4. Verify: load the library and build a sample of compute pipelines on this GPU.
cat > "$WORK/check.swift" <<'EOF'
import Metal
import Foundation
let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let names = lib.functionNames
var ok = 0
for (i, n) in names.enumerated() where i % 200 == 0 {
    let f = lib.makeFunction(name: n)!
    // Functions with function constants need specialization first; skip them.
    if !f.functionConstantsDictionary.isEmpty { continue }
    _ = try! dev.makeComputePipelineState(function: f)
    ok += 1
}
print("\(names.count) functions, \(ok) sampled pipelines built")
EOF
swiftc -O -o "$WORK/check" "$WORK/check.swift" 2>/dev/null
"$WORK/check" "$OUT"
ls -la "$OUT"

# 5. Install.
if [[ "${1:-}" != "--no-install" ]]; then
    cp "$OUT" "$ROOT/default.metallib"
    cp "$OUT" "$ROOT/Tools/default.metallib"
    echo "Installed to $ROOT/default.metallib and $ROOT/Tools/default.metallib"
else
    echo "Built $OUT (not installed)"
fi
