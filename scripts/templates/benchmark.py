import shutil
import subprocess
import time
from templates import render_all, try_create_dirs, copy_all
import fire  # type: ignore
import pathlib
import os


def get_bazel_args(
    tool: str, pipeline: list[str], benchmark_root: pathlib.Path | None = None
) -> list[str]:
    args = ["bazel", "run", f"@heir//tools:{tool}", "--"]
    for heir_pass in pipeline:
        args.append(f"--{heir_pass}")

    if benchmark_root is not None:
        benchmark_name = benchmark_root.stem
        benchmark_file = benchmark_root.absolute() / f"{benchmark_name}.mlir"
        args.append(str(benchmark_file))
    return args


def try_run(
    name: str, args: list[str], input: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        args, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(result.stderr.decode())
        raise RuntimeError(f"{name} failed with exit code: {result.returncode}")
    return result


class CLI:
    def __init__(self):
        git_root = pathlib.Path(__file__).parent.parent.parent
        if not os.path.isdir(git_root / ".git"):
            raise RuntimeError(f"Could not find git root, looked at {git_root}")
        self.root = git_root

    def verify(self, root_dir: str, benchmark_name: str):
        """Syntactically verify a benchmark without running any passes

        Args:
            root_dir: The `project` directory that holds the benchmark
            benchmark_name: The name of the benchmark to verify
        """

        benchmark_root = pathlib.Path(root_dir) / benchmark_name
        benchmark_ir = benchmark_root / f"{benchmark_name}.mlir"

        if not benchmark_ir.exists():
            raise ValueError(f"Could not find benchmark at {benchmark_ir}")

        try:
            try_run(
                f"Verifying {benchmark_name}",
                get_bazel_args("heir-opt", [], benchmark_root),
            )
            print(f"Successfully verified {benchmark_name}")
        except Exception as e:
            print(e)

    def new_benchmark(self, root_dir: str, benchmark_name: str, force: bool = False):
        """Initialize a new empty benchmark

        Args:
          root_dir: The `project` directory that will hold the benchmark
          benchmark_name: The name to use for the benchmark and associated files
          force: If True, overwrite existing files. If False, raise an error if
            any files already exist.
        """
        root = pathlib.Path(root_dir)
        if not os.path.isdir(root):
            raise RuntimeError(f"Could not find project root at {root}")
        benchmark_root = root / benchmark_name
        templ_root = self.root / "scripts" / "templates" / "Benchmark"
        path_mapping = {
            templ_root / "CMakeLists.txt.jinja": benchmark_root / "CMakeLists.txt",
            templ_root / "harness.cpp.jinja": benchmark_root / "harness.cpp",
            templ_root
            / "benchmark.mlir.jinja": benchmark_root
            / f"{benchmark_name}.mlir",
        }

        try:
            try_create_dirs(benchmark_root, force)
            try_create_dirs(benchmark_root / "IR", force)
            copy_all(path_mapping)
            render_all(benchmark_root, benchmark_name=benchmark_name)
        except:
            print("Hit unrecoverable error, cleaning up")
            shutil.rmtree(benchmark_root)
            raise

    def compile_benchmark(self, root_dir: str, benchmark_name: str):
        """Compile and codegen an existing benchmark

        Args:
          root_dir: The `project` directory that holds the benchmark
          benchmark_name: The name of the benchmark to compile
        """
        benchmark_root = pathlib.Path(root_dir) / benchmark_name
        benchmark_ir = benchmark_root / f"{benchmark_name}.mlir"
        benchmark_opt = benchmark_root / f"{benchmark_name}.opt.cpp"
        benchmark_unopt = benchmark_root / f"{benchmark_name}.unopt.cpp"
        benchmark_nosys = benchmark_root / f"{benchmark_name}.nosys.cpp"
        benchmark_hdrs = benchmark_root / f"{benchmark_name}.hpp"
        time_file = benchmark_root / "time.txt"

        ir_path = benchmark_root / "IR"
        if not benchmark_root.exists():
            raise ValueError(f"Could not find benchmark at {benchmark_root}")
        if not benchmark_ir.exists():
            raise ValueError(f"Could not find benchmark at {benchmark_ir}")

        optimized_pipeline = [
            "unroll-secret-loops",
            "yosys-optimizer=mode=LUT",
            "shrink-lut-constants",
            "merge-luts",
            "secret-distribute-generic",
            "canonicalize",
            "secret-to-cggi",
            "cggi-canonicalize-luts",
            "cse",
            "cggi-to-openfhe",
        ]

        unoptimized_pipeline = [
            "unroll-secret-loops",
            "yosys-optimizer=mode=LUT",
            "shrink-lut-constants",
            "secret-distribute-generic",
            "canonicalize",
            "secret-to-cggi",
            "cggi-canonicalize-luts",
            "cse",
            "cggi-to-openfhe",
        ]

        no_yosys_pipeline = [
            "unroll-secret-loops",
            "yosys-optimizer=mode=Boolean",
            "shrink-lut-constants",
            "secret-distribute-generic",
            "canonicalize",
            "secret-to-cggi",
            "cggi-canonicalize-luts",
            "cse",
            "cggi-to-openfhe",
        ]

        if not ir_path.exists():
            os.makedirs(ir_path)

        start = time.time()
        print(f"Compiling {benchmark_name}.nosys...", end="", flush=True)
        nosys_compiled = try_run(
            f"Compilation of {benchmark_name}.nosys",
            get_bazel_args("heir-opt", no_yosys_pipeline, benchmark_root),
        ).stdout

        (ir_path / "nosys.mlir").write_text(nosys_compiled.decode())

        print("codegen...", end="", flush=True)
        nosys_codegen = try_run(
            f"Codegen of {benchmark_name}.nosys",
            get_bazel_args("heir-translate", ["emit-openfhe-bin"]),
            nosys_compiled,
        ).stdout.decode()

        benchmark_nosys.write_text(nosys_codegen)
        end = time.time()
        nosys_time = end - start
        print(f"done! ({int(nosys_time)} sec)")

        start = time.time()
        print(f"Compiling {benchmark_name}.unopt...", end="", flush=True)
        unopt_compiled = try_run(
            f"Compilation of {benchmark_name}.unopt",
            get_bazel_args("heir-opt", unoptimized_pipeline, benchmark_root),
        ).stdout

        (ir_path / "unopt.mlir").write_text(unopt_compiled.decode())

        print("codegen...", end="", flush=True)
        unopt_codegen = try_run(
            f"Codegen of {benchmark_name}.unopt",
            get_bazel_args("heir-translate", ["emit-openfhe-bin"]),
            unopt_compiled,
        ).stdout.decode()

        benchmark_unopt.write_text(unopt_codegen)
        end = time.time()
        unopt_time = end - start
        print(f"done! ({int(unopt_time)} sec)")

        start = time.time()
        print(f"Compiling {benchmark_name}.opt...", end="", flush=True)
        opt_compiled = try_run(
            f"Compilation of {benchmark_name}.opt",
            get_bazel_args("heir-opt", optimized_pipeline, benchmark_root),
        ).stdout

        (ir_path / "opt.mlir").write_text(opt_compiled.decode())

        print("codegen...", end="", flush=True)
        opt_codegen = try_run(
            f"Codegen of {benchmark_name}.opt",
            get_bazel_args("heir-translate", ["emit-openfhe-bin"]),
            opt_compiled,
        ).stdout.decode()

        benchmark_opt.write_text(opt_codegen)

        end = time.time()
        opt_time = end - start
        print(f"done! ({int(opt_time)} sec)")

        header = try_run(
            f"Header generation of {benchmark_name}",
            get_bazel_args("heir-translate", ["emit-openfhe-bin-header"]),
            opt_compiled,
        ).stdout.decode()
        benchmark_hdrs.write_text(header)
        print(f"Wrote headers to {benchmark_hdrs}")

        time_file.write_text(f"Unopt: {unopt_time} seconds\nOpt: {opt_time} seconds")


if __name__ == "__main__":
    fire.Fire(CLI)
