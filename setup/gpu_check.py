#!/usr/bin/env python3
"""
GPU Check Script for silgymcu environment
Checks if PyTorch, JAX, and cuML are working with GPU
"""

import sys
import time
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def check_pytorch() -> Dict[str, Any]:
    """Check PyTorch GPU availability and functionality"""
    result = {
        "name": "PyTorch",
        "available": False,
        "gpu_available": False,
        "details": [],
        "error": None
    }

    try:
        import torch
        result["available"] = True
        result["details"].append(f"Version: {torch.__version__}")

        # Check CUDA availability
        if torch.cuda.is_available():
            result["gpu_available"] = True
            result["details"].append(f"CUDA Available: ✓")
            result["details"].append(f"CUDA Version: {torch.version.cuda}")
            result["details"].append(f"GPU Count: {torch.cuda.device_count()}")

            # Get GPU details
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                result["details"].append(f"GPU {i}: {gpu_name}")

            # Test GPU operation
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                _ = torch.matmul(x, y)
                result["details"].append("GPU Operation Test: ✓")
            except Exception as e:
                result["error"] = f"GPU operation failed: {str(e)}"
        else:
            result["details"].append("CUDA Available: ✗")
            result["error"] = "CUDA is not available"

    except ImportError as e:
        result["error"] = f"Import failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


def check_jax() -> Dict[str, Any]:
    """Check JAX GPU availability and functionality"""
    result = {
        "name": "JAX",
        "available": False,
        "gpu_available": False,
        "details": [],
        "error": None
    }

    try:
        import jax
        import jax.numpy as jnp
        result["available"] = True
        result["details"].append(f"Version: {jax.__version__}")

        # Check devices
        devices = jax.devices()
        result["details"].append(f"Devices: {len(devices)}")

        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            result["gpu_available"] = True
            result["details"].append(f"GPU Devices: {len(gpu_devices)}")

            for i, device in enumerate(gpu_devices):
                result["details"].append(f"GPU {i}: {device.device_kind}")

            # Test GPU operation
            try:
                x = jnp.ones((1000, 1000))
                y = jnp.ones((1000, 1000))
                z = jnp.dot(x, y)
                z.block_until_ready()  # Wait for computation to complete
                result["details"].append("GPU Operation Test: ✓")
                result["details"].append(f"Default Backend: {devices[0].platform}")
            except Exception as e:
                result["error"] = f"GPU operation failed: {str(e)}"
        else:
            result["details"].append(f"Default Backend: {devices[0].platform}")
            result["error"] = "No GPU devices found"

    except ImportError as e:
        result["error"] = f"Import failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


def check_cuml() -> Dict[str, Any]:
    """Check cuML GPU availability and functionality"""
    result = {
        "name": "cuML",
        "available": False,
        "gpu_available": False,
        "details": [],
        "error": None
    }

    try:
        import cuml
        result["available"] = True
        result["details"].append(f"Version: {cuml.__version__}")

        # Check CuPy and GPU availability
        try:
            import cupy as cp
            result["details"].append("CuPy Import: ✓")

            # Check GPU devices via CuPy
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count > 0:
                result["details"].append(f"GPU Count: {gpu_count}")

                # Get GPU info
                for i in range(gpu_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    gpu_name = device_props['name'].decode('utf-8')
                    result["details"].append(f"GPU {i}: {gpu_name}")

                # Test cuML operation on GPU
                try:
                    from cuml.manifold import UMAP

                    # Create test data on GPU (small dataset for quick test)
                    X = cp.random.rand(100, 20, dtype=cp.float32)

                    # Run UMAP with minimal epochs for quick test
                    umap = UMAP(n_components=2, n_neighbors=15, n_epochs=20, random_state=42, verbose=False)
                    embedding = umap.fit_transform(X)
                    _ = embedding  # Ensure operation completed

                    result["gpu_available"] = True
                    result["details"].append("GPU Operation Test: ✓")
                    result["details"].append("UMAP Dimensionality Reduction: ✓")

                except Exception as e:
                    result["error"] = f"cuML operation failed: {str(e)}"
            else:
                result["error"] = "No GPU devices found"

        except ImportError as e:
            result["error"] = f"CuPy import failed: {str(e)}"
        except Exception as e:
            result["error"] = f"GPU check failed: {str(e)}"

    except ImportError as e:
        result["error"] = f"cuML import failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"

    return result


def benchmark_pytorch() -> Dict[str, Any]:
    """Benchmark PyTorch CPU vs GPU performance"""
    result = {
        "name": "PyTorch",
        "cpu_time": None,
        "gpu_time": None,
        "speedup": None,
        "error": None
    }

    try:
        import torch

        if not torch.cuda.is_available():
            result["error"] = "GPU not available"
            return result

        # Warm up
        _ = torch.randn(10, 10, device='cuda')
        torch.cuda.synchronize()

        # CPU benchmark
        size = 10000
        try:
            x_cpu = torch.randn(size, size, device='cpu')
            y_cpu = torch.randn(size, size, device='cpu')

            start = time.time()
            _ = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start
            result["cpu_time"] = cpu_time
        except Exception as e:
            result["error"] = f"CPU benchmark failed: {str(e)}"
            return result

        # GPU benchmark
        try:
            x_gpu = torch.randn(size, size, device='cuda')
            y_gpu = torch.randn(size, size, device='cuda')
            torch.cuda.synchronize()

            start = time.time()
            _ = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            result["gpu_time"] = gpu_time

            result["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
        except Exception as e:
            result["error"] = f"GPU benchmark failed: {str(e)}"

    except Exception as e:
        result["error"] = f"Benchmark error: {str(e)}"

    return result


def benchmark_jax() -> Dict[str, Any]:
    """Benchmark JAX CPU vs GPU performance with JIT compilation"""
    result = {
        "name": "JAX (JIT)",
        "cpu_time": None,
        "gpu_time": None,
        "speedup": None,
        "error": None
    }

    try:
        import jax
        import jax.numpy as jnp
        from jax import devices

        gpu_devices = [d for d in devices() if d.platform == 'gpu']
        if not gpu_devices:
            result["error"] = "GPU not available"
            return result

        cpu_device = devices('cpu')[0]
        gpu_device = gpu_devices[0]

        # Define JIT-compiled function
        @jax.jit
        def matmul_fn(x, y):
            return jnp.dot(x, y)

        # Warm up JIT compilation
        x_warm = jnp.ones((10, 10))
        y_warm = jnp.ones((10, 10))
        _ = matmul_fn(x_warm, y_warm)
        jax.block_until_ready(_)

        # CPU benchmark
        size = 10000
        try:
            with jax.default_device(cpu_device):
                x_cpu = jnp.ones((size, size))
                y_cpu = jnp.ones((size, size))

                # Compile for CPU
                _ = matmul_fn(x_cpu, y_cpu)
                jax.block_until_ready(_)

                # Benchmark
                start = time.time()
                z = matmul_fn(x_cpu, y_cpu)
                jax.block_until_ready(z)
                cpu_time = time.time() - start
                result["cpu_time"] = cpu_time
        except Exception as e:
            result["error"] = f"CPU benchmark failed: {str(e)}"
            return result

        # GPU benchmark
        try:
            with jax.default_device(gpu_device):
                x_gpu = jnp.ones((size, size))
                y_gpu = jnp.ones((size, size))

                # Compile for GPU
                _ = matmul_fn(x_gpu, y_gpu)
                jax.block_until_ready(_)

                # Benchmark
                start = time.time()
                z = matmul_fn(x_gpu, y_gpu)
                jax.block_until_ready(z)
                gpu_time = time.time() - start
                result["gpu_time"] = gpu_time

                result["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
        except Exception as e:
            result["error"] = f"GPU benchmark failed: {str(e)}"

    except Exception as e:
        result["error"] = f"Benchmark error: {str(e)}"

    return result


def benchmark_cuml() -> Dict[str, Any]:
    """Benchmark cuML UMAP (GPU) vs umap-learn (CPU) performance"""
    result = {
        "name": "cuML (UMAP)",
        "cpu_time": None,
        "gpu_time": None,
        "speedup": None,
        "error": None
    }

    try:
        import cupy as cp
        from cuml.manifold import UMAP as cuUMAP

        # Check if umap-learn is available for comparison
        try:
            from umap import UMAP as cpuUMAP
            import numpy as np
        except ImportError:
            result["error"] = "umap-learn not available for comparison"
            return result

        # Warm up GPU (quick warm up)
        X_warm = cp.random.rand(100, 50, dtype=cp.float32)
        umap_warm = cuUMAP(n_components=2, n_neighbors=10, n_epochs=10, verbose=False)
        _ = umap_warm.fit_transform(X_warm)
        cp.cuda.Stream.null.synchronize()

        # Test data - optimized for speed while still showing GPU advantage
        # Note: UMAP's main bottleneck is neighbor graph construction, not n_epochs
        n_samples = 5000   # Reduced for faster neighbor graph construction
        n_features = 50    # Reduced from 100
        n_components = 2
        n_neighbors = 10   # Reduced from 15 for faster graph construction
        n_epochs = 10      # Minimal epochs for quick optimization

        # CPU benchmark (umap-learn)
        try:
            # Data generation is outside timing (fair comparison)
            X_cpu = np.random.rand(n_samples, n_features).astype(np.float32)

            start = time.time()
            umap_cpu = cpuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                n_epochs=n_epochs,
                random_state=42,
                verbose=False
            )
            umap_cpu.fit(X_cpu)
            cpu_time = time.time() - start
            result["cpu_time"] = cpu_time
        except Exception as e:
            result["error"] = f"CPU benchmark failed: {str(e)}"
            return result

        # GPU benchmark (cuML)
        try:
            # Data generation is outside timing (fair comparison)
            X_gpu = cp.random.rand(n_samples, n_features, dtype=cp.float32)

            start = time.time()
            umap_gpu = cuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                n_epochs=n_epochs,
                random_state=42,
                verbose=False
            )
            umap_gpu.fit(X_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start
            result["gpu_time"] = gpu_time

            result["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
        except Exception as e:
            result["error"] = f"GPU benchmark failed: {str(e)}"

    except Exception as e:
        result["error"] = f"Benchmark error: {str(e)}"

    return result


def create_status_panel(result: Dict[str, Any]) -> Panel:
    """Create a rich panel for a library check result"""

    # Determine status
    if not result["available"]:
        status_emoji = "❌"
        status_text = "NOT INSTALLED"
        status_color = "red"
    elif result["gpu_available"]:
        status_emoji = "✅"
        status_text = "GPU WORKING"
        status_color = "green"
    else:
        status_emoji = "⚠️"
        status_text = "CPU ONLY"
        status_color = "yellow"

    # Build content
    content = Text()
    content.append(f"{status_emoji} Status: ", style="bold")
    content.append(f"{status_text}\n\n", style=f"bold {status_color}")

    # Add details
    if result["details"]:
        for detail in result["details"]:
            content.append(f"  • {detail}\n", style="cyan")

    # Add error if present
    if result["error"]:
        content.append(f"\n  ⚠ Error: {result['error']}", style="bold red")

    # Choose border color
    border_style = status_color if result["available"] else "red"

    return Panel(
        content,
        title=f"[bold]{result['name']}[/bold]",
        border_style=border_style,
        box=box.ROUNDED
    )


def create_summary_table(results: list) -> Table:
    """Create a summary table of all checks"""
    table = Table(
        title="📊 GPU Check Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Library", style="cyan", no_wrap=True)
    table.add_column("Installed", justify="center")
    table.add_column("GPU Available", justify="center")
    table.add_column("Status", justify="center")

    for result in results:
        installed = "✅" if result["available"] else "❌"
        gpu = "✅" if result["gpu_available"] else "❌"

        if not result["available"]:
            status = "[red]Not Installed[/red]"
        elif result["gpu_available"]:
            status = "[green]Working[/green]"
        else:
            status = "[yellow]CPU Only[/yellow]"

        table.add_row(result["name"], installed, gpu, status)

    return table


def create_benchmark_table(benchmarks: list) -> Table:
    """Create a performance comparison table"""
    table = Table(
        title="⚡ Performance Comparison (CPU vs GPU)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Library", style="cyan", no_wrap=True)
    table.add_column("CPU Time", justify="right", style="yellow")
    table.add_column("GPU Time", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="bold cyan")
    table.add_column("Status", justify="center")

    for bench in benchmarks:
        if bench["error"]:
            table.add_row(
                bench["name"],
                "N/A",
                "N/A",
                "N/A",
                f"[red]{bench['error']}[/red]"
            )
        elif bench["cpu_time"] is not None and bench["gpu_time"] is not None:
            cpu_time_str = f"{bench['cpu_time']:.4f}s"
            gpu_time_str = f"{bench['gpu_time']:.4f}s"
            speedup_str = f"{bench['speedup']:.2f}x"

            # Color code speedup
            if bench["speedup"] > 10:
                speedup_color = "bold green"
            elif bench["speedup"] > 2:
                speedup_color = "green"
            elif bench["speedup"] > 1:
                speedup_color = "yellow"
            else:
                speedup_color = "red"

            table.add_row(
                bench["name"],
                cpu_time_str,
                gpu_time_str,
                f"[{speedup_color}]{speedup_str}[/{speedup_color}]",
                "[green]✓[/green]"
            )
        else:
            table.add_row(
                bench["name"],
                "N/A",
                "N/A",
                "N/A",
                "[yellow]Skipped[/yellow]"
            )

    return table


def main():
    """Main function to run all GPU checks"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]GPU Availability Check[/bold cyan]\n"
        "[dim]Checking PyTorch, JAX, and cuML GPU support[/dim]",
        border_style="cyan"
    ))
    console.print("\n")

    # Run checks
    with console.status("[bold green]Checking PyTorch...", spinner="dots"):
        pytorch_result = check_pytorch()
    console.print(create_status_panel(pytorch_result))
    console.print("\n")

    with console.status("[bold green]Checking JAX...", spinner="dots"):
        jax_result = check_jax()
    console.print(create_status_panel(jax_result))
    console.print("\n")

    with console.status("[bold green]Checking cuML...", spinner="dots"):
        cuml_result = check_cuml()
    console.print(create_status_panel(cuml_result))
    console.print("\n")

    # Print summary
    results = [pytorch_result, jax_result, cuml_result]
    console.print(create_summary_table(results))
    console.print("\n")

    # Final verdict
    all_gpu_working = all(r["gpu_available"] for r in results)

    if all_gpu_working:
        console.print(Panel(
            "[bold green]🎉 All libraries are working with GPU! 🎉[/bold green]",
            border_style="green"
        ))
        console.print("\n")

        # Run performance benchmarks
        console.print(Panel.fit(
            "[bold cyan]Performance Benchmarks[/bold cyan]\n"
            "[dim]Comparing CPU vs GPU performance[/dim]",
            border_style="cyan"
        ))
        console.print("\n")

        benchmarks = []

        # PyTorch benchmark
        if pytorch_result["gpu_available"]:
            with console.status("[bold green]Benchmarking PyTorch (10000x10000 matrix multiplication)...", spinner="dots"):
                pytorch_bench = benchmark_pytorch()
            benchmarks.append(pytorch_bench)
            if not pytorch_bench["error"]:
                console.print(f"[green]✓[/green] PyTorch: {pytorch_bench['speedup']:.2f}x speedup")
            else:
                console.print(f"[red]✗[/red] PyTorch: {pytorch_bench['error']}")

        # JAX benchmark
        if jax_result["gpu_available"]:
            with console.status("[bold green]Benchmarking JAX with JIT (10000x10000 matrix multiplication)...", spinner="dots"):
                jax_bench = benchmark_jax()
            benchmarks.append(jax_bench)
            if not jax_bench["error"]:
                console.print(f"[green]✓[/green] JAX (JIT): {jax_bench['speedup']:.2f}x speedup")
            else:
                console.print(f"[red]✗[/red] JAX (JIT): {jax_bench['error']}")

        # cuML benchmark
        if cuml_result["gpu_available"]:
            with console.status("[bold green]Benchmarking cuML UMAP (5k samples, 50→2 dims, 10 neighbors)...", spinner="dots"):
                cuml_bench = benchmark_cuml()
            benchmarks.append(cuml_bench)
            if not cuml_bench["error"]:
                console.print(f"[green]✓[/green] cuML UMAP: {cuml_bench['speedup']:.2f}x speedup")
            else:
                console.print(f"[red]✗[/red] cuML UMAP: {cuml_bench['error']}")

        console.print("\n")
        if benchmarks:
            console.print(create_benchmark_table(benchmarks))
            console.print("\n")
    else:
        failing = [r["name"] for r in results if not r["gpu_available"]]
        console.print(Panel(
            f"[bold yellow]⚠️  Some libraries are not using GPU: {', '.join(failing)}[/bold yellow]\n"
            "[dim]Skipping performance benchmarks[/dim]",
            border_style="yellow"
        ))
        console.print("\n")

    # Return exit code
    return 0 if all_gpu_working else 1


if __name__ == "__main__":
    sys.exit(main())
