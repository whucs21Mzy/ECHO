"""Single-prompt LayerSkip ECHO generation with native SGLang layers."""

from .inference_looped import main as generate
from .looped_common import cleanup


def main():
    generate(default_backend="sglang")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
