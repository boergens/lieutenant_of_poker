"""Progress bar utilities."""

import sys

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)


def create_progress(*extra_columns):
    """Create a standard progress bar with optional extra columns."""
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        *extra_columns,
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
    ]
    return Progress(*columns, console=None, transient=False, disable=not sys.stderr.isatty())
