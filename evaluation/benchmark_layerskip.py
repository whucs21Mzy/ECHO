"""Paired native SGLang AR/ECHO measurements for LayerSkip checkpoints."""

from .benchmark_looped import main as benchmark
from .looped_common import cleanup


def main():
    benchmark(default_backend="sglang")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
