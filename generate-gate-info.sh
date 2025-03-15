#!/usr/bin/env bash
exec &> >(tee "generate-gate-info.log")

BENCHMARKPATH="/local/scratch/a/paranjav/coyote-project/paper/heir/benchmarks/"

set -e 

echo "====== [1/1] Parsing MLIR generate code ======"
for suite in $BENCHMARKPATH/*/
do	
	su=$(basename "$suite")
	for benchmark in $BENCHMARKPATH/$su/*/
	do
		bmk=$(basename "$benchmark")
		# echo "==== Looking at benchmark -> ${su}:${bmk} ===="
        unopt_gate=`cat /local/scratch/a/paranjav/coyote-project/paper/heir/benchmarks/$su/$bmk/IR/unopt.mlir | grep eval_func | wc -l`
        opt_gate=`cat /local/scratch/a/paranjav/coyote-project/paper/heir/benchmarks/$su/$bmk/IR/opt.mlir | grep eval_func | wc -l`

        echo "${bmk},${unopt_gate},${opt_gate}"
	done
done