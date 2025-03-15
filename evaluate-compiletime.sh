#!/usr/bin/env bash
exec &> >(tee "evaluate-compiletime.log")

BENCHMARKPATH="/local/scratch/a/paranjav/coyote-project/paper/heir/benchmarks/"

set -e 
if ! [ -x "$(command -v numactl)" ]; then
  echo 'Error: numactl is not installed.' >&2
  exit 1
fi

if ! [ -x "$(command -v tee)" ]; then
  echo 'Error: numactl is not installed.' >&2
  exit 1
fi

echo "====== [1/2] Lowering MLIR benchmarks to FHE Code ======"
for suite in $BENCHMARKPATH/*/
do	
	su=$(basename "$suite")
	for benchmark in $BENCHMARKPATH/$su/*/
	do
		bmk=$(basename "$benchmark")
		echo "==== Lowering benchmark -> ${su}:${bmk} ===="
		python3 scripts/templates/benchmark.py compile_benchmark /local/scratch/a/paranjav/coyote-project/paper/heir/benchmarks/$su/ $bmk
	done
done

echo "====== [2/2] Compiling FHE Code to Binary ======"
for suite in $BENCHMARKPATH/*/
do	
	su=$(basename "$suite")
	for benchmark in $BENCHMARKPATH/$su/*/
	do
		bmk=$(basename "$benchmark")
		echo "==== Compiling benchmark -> ${su}:${bmk} ===="
		mkdir -p $benchmark/build
		cmake -B$benchmark/build -S$benchmark
		make -C $benchmark/build clean
		echo "==== Compiling unoptimized version ===="
		/usr/bin/time -f "${bmk}:unopt:runtime:%e" -- make -j -C $benchmark/build $bmk.unopt
		echo "==== Compiling optimized version ===="
		/usr/bin/time -f "${bmk}:opt:runtime:%e" -- make -j -C $benchmark/build $bmk.opt
	done
done