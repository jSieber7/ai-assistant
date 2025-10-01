import re
import sys
from pathlib import Path


def bump_version(new_version):
    """Update version in all files to the new version."""

    # Update app/version.py
    version_file = Path("app/__init__.py")
    if version_file.exists():
        content = version_file.read_text()
        updated_content = re.sub(
            r'__version__ = ".*"', f'__version__ = "{new_version}"', content
        )
        version_file.write_text(updated_content)
        print(f"Updated app/__init__.py to version {new_version}")
    else:
        print("Warning: app/__init__.py not found")

    # Update docs/index.md
    docs_file = Path("docs/index.md")
    if docs_file.exists():
        docs_content = docs_file.read_text()
        updated_docs = re.sub(
            r"\*\*Current Version\*\*: .*",
            f"**Current Version**: {new_version}",
            docs_content,
        )
        docs_file.write_text(updated_docs)
        print(f"Updated docs/index.md to version {new_version}")
    else:
        print("Warning: docs/index.md not found")

    print(f"Successfully bumped version to {new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <new_version>")
        sys.exit(1)

    bump_version(sys.argv[1])
