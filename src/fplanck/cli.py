"""CLI entry point for running fplanck examples."""

import sys
from pathlib import Path


def get_examples_dir() -> Path:
    """Get the examples directory, works both in dev and installed package."""
    # Examples are now part of the package
    examples_dir = Path(__file__).parent / "examples"
    if examples_dir.exists():
        return examples_dir

    raise FileNotFoundError("Could not find examples directory")


def main() -> None:
    """Run an fplanck example script by name."""
    if len(sys.argv) < 2:
        print("Usage: uvx fplanck <example-name>")
        print("\nAvailable examples:")
        try:
            examples_dir = get_examples_dir()
            for ex in sorted(examples_dir.glob("*.py")):
                print(f"  - {ex.stem}")
        except FileNotFoundError:
            print("  (examples not found)")
        sys.exit(1)

    example_name = sys.argv[1]

    try:
        examples_dir = get_examples_dir()
    except FileNotFoundError:
        print("Error: Examples directory not found")
        sys.exit(1)

    example_file = examples_dir / f"{example_name}.py"

    if not example_file.exists():
        print(f"Error: Example '{example_name}' not found")
        sys.exit(1)

    # Execute the example file directly
    with example_file.open() as f:
        code = compile(f.read(), str(example_file), "exec")
        exec(code, {"__name__": "__main__"})


if __name__ == "__main__":
    main()
