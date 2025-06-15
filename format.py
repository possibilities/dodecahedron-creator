import subprocess
import sys


def main():
    try:
        # Run ruff check with fixes
        subprocess.run(["uvx", "ruff", "check", "--fix", "."], check=True)
        # Run ruff format
        subprocess.run(["uvx", "ruff", "format", "."], check=True)
        print("✓ Formatting complete")
    except subprocess.CalledProcessError as e:
        print(f"✗ Formatting failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
