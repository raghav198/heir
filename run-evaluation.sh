#!/usr/bin/env bash
exec &> >(tee "run-evaluation.log")

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

echo "====== [1/4] Lowering MLIR benchmarks to FHE Code ======"
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

echo "====== [2/4] Compiling FHE Code to Binary ======"
for suite in $BENCHMARKPATH/*/
do	
	su=$(basename "$suite")
	for benchmark in $BENCHMARKPATH/$su/*/
	do
		bmk=$(basename "$benchmark")
		echo "==== Compiling benchmark -> ${su}:${bmk} ===="
		mkdir -p $benchmark/build
		cmake -B$benchmark/build -S$benchmark
		make -j -C $benchmark/build clean
		echo "==== Compiling unoptimized version ===="
		/usr/bin/time -f "${bmk}:unopt:runtime:%e" -- make -j -C $benchmark/build $bmk.unopt
		echo "==== Compiling optimized version ===="
		/usr/bin/time -f "${bmk}:opt:runtime:%e" -- make -j -C $benchmark/build $bmk.opt
	done
done

echo "====== [3/4] Running FHE Binaries ======"
for suite in $BENCHMARKPATH/*/
do	
	su=$(basename "$suite")
	for benchmark in $BENCHMARKPATH/$su/*/
	do
		bmk=$(basename "$benchmark")
		echo "==== Running benchmark -> ${su}:${bmk} ===="
		mkdir -p $benchmark/results
		echo "== Running unoptimized =="
		numactl --physcpubind=0 --membind=0 $benchmark/build/$bmk.unopt | tee $benchmark/results/$bmk.unopt.log
		echo "== Running optimized =="
		numactl --physcpubind=0 --membind=0 $benchmark/build/$bmk.opt | tee $benchmark/results/$bmk.opt.log
	done
done

echo "====== [4/4] Generating FHE Run Results ======"
touch $BENCHMARKPATH/results.csv
cat /dev/null > $BENCHMARKPATH/results.csv
echo -e "Benchmark;Run 1;Run 2;Run 3;Run 4;Run 5;Run 6;Run 7;Run 8;Run 9;Run 10;Run 11;Run 12;Run 13;Run 14;Run 15;Run 16;Run 17;Run 18;Run 19;Run 20;Run 21;Run 22;Run 23;Run 24;Run 25;Run 26;Run 27;Run 28;Run 29;Run 30" >> $BENCHMARKPATH/results.csv
for suite in $BENCHMARKPATH/*/
do	
	su=$(basename "$suite")
	for benchmark in $BENCHMARKPATH/$su/*/
	do
		bmk=$(basename "$benchmark")
		echo "==== Parsing benchmark results -> ${su}:${bmk} ===="
		sed -i 's/,/;/g' $benchmark/results/$bmk.unopt.log
		sed -i 's/,/;/g' $benchmark/results/$bmk.opt.log
		echo -e "$bmk.unopt;$(cat $benchmark/results/$bmk.unopt.log)" >> $BENCHMARKPATH/results.csv
		echo -e "$bmk.opt;$(cat $benchmark/results/$bmk.opt.log)" >> $BENCHMARKPATH/results.csv
	done
done
