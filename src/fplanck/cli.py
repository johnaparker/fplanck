"""CLI entry point for running fplanck examples."""

import sys
from pathlib import Path


def main() -> None:
    """Run an fplanck example script by name."""
    if len(sys.argv) < 2:
        print("Usage: uvx fplanck <example-name>")
        print("\nAvailable examples:")
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        for ex in sorted(examples_dir.glob("*.py")):
            print(f"  - {ex.stem}")
        sys.exit(1)

    example_name = sys.argv[1]
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    example_file = examples_dir / f"{example_name}.py"

    if not example_file.exists():
        print(f"Error: Example '{example_name}' not found")
        sys.exit(1)

    # Execute the example file directly
    with open(example_file) as f:
        code = compile(f.read(), example_file, "exec")
        exec(code, {"__name__": "__main__"})


if __name__ == "__main__":
    main()
