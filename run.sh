#!/bin/bash
cd /local/scratch/a/paranjav/.cache/bazel/_bazel_paranjav/094499ec8a8b0cba839455cae1ffd6f0/execroot/heir/bazel-out/k8-dbg/bin/tools/heir-opt.runfiles/heir && \
  exec env \
    -u JAVA_RUNFILES \
    -u RUNFILES_DIR \
    -u RUNFILES_MANIFEST_FILE \
    -u RUNFILES_MANIFEST_ONLY \
    -u TEST_SRCDIR \
    BUILD_WORKING_DIRECTORY=/local/scratch/a/paranjav/coyote-project/paper/heir \
    BUILD_WORKSPACE_DIRECTORY=/local/scratch/a/paranjav/coyote-project/paper/heir \
  perf record -e cycles -a -- /local/scratch/a/paranjav/.cache/bazel/_bazel_paranjav/094499ec8a8b0cba839455cae1ffd6f0/execroot/heir/bazel-out/k8-dbg/bin/tools/heir-opt --unroll-secret-loops --yosys-optimizer --shrink-lut-constants --merge-luts /local/scratch/a/paranjav/coyote-project/paper/heir/temp/psi8/psi8.mlir --mlir-timing "$@"
