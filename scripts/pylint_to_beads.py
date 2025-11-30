#!/usr/bin/env python3
"""Parse pylint output and create one bead per issue."""

import subprocess


def run_pylint():
    """Run pylint and return output lines."""
    result = subprocess.run(
        ["pylint", "lieutenant_of_poker"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def parse_pylint_output(output):
    """Parse pylint output into individual issues."""
    issues = []

    for line in output.splitlines():
        if not line.startswith("lieutenant_of_poker/"):
            continue

        # Parse: file.py:10:0: C0301: Line too long (115/88) (line-too-long)
        parts = line.split(": ", 2)
        if len(parts) < 3:
            continue

        location = parts[0]  # file.py:10:0
        code = parts[1]      # C0301
        message = parts[2]   # Line too long (115/88) (line-too-long)

        loc_parts = location.split(":")
        if len(loc_parts) >= 2:
            issues.append({
                "file": loc_parts[0],
                "line": loc_parts[1],
                "code": code,
                "message": message,
            })

    return issues


def create_bead(issue):
    """Create a bead for a single pylint issue."""
    filepath = issue["file"]
    line = issue["line"]
    code = issue["code"]
    message = issue["message"]

    # Get symbol from message: "Line too long (115/88) (line-too-long)" -> "line-too-long"
    symbol = code
    if "(" in message:
        symbol = message.split("(")[-1].rstrip(")")

    # Get short filename
    short_file = filepath.replace("lieutenant_of_poker/", "")

    title = f"{symbol}: {short_file}:{line}"

    description = f"""{message}

File: {filepath}
Line: {line}

Verify: pylint {filepath} 2>&1 | grep ":{line}:.*{code}" """

    result = subprocess.run(
        ["bd", "create", f"--title={title}", "--type=task", f"--description={description}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"✓ {short_file}:{line} {symbol}")
    else:
        print(f"✗ {short_file}:{line} {result.stderr.strip()}")


def main():
    print("Running pylint...")
    issues = parse_pylint_output(run_pylint())

    if not issues:
        print("No issues found!")
        return

    print(f"Found {len(issues)} issues\n")
    for issue in issues:
        create_bead(issue)

    print(f"\nCreated {len(issues)} beads")


if __name__ == "__main__":
    main()
