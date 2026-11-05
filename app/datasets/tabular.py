"""Parse CSV / Excel files into typed tables, ready to load (docs/DESIGN.md section 15.3).

Everything here is pure Python on in-memory data: parsing, name cleaning and type inference.
Loading into Postgres is in tabular_loader.py.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

MAX_ROWS_PER_TABLE = 1_000_000
MAX_COLUMNS = 200
PG_TYPES = ("text", "integer", "bigint", "numeric", "boolean", "date", "timestamp")
_TRUE, _FALSE = {"true", "t", "yes", "y"}, {"false", "f", "no", "n"}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$")
_INT = re.compile(r"^[+-]?\d{1,18}$")
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_RESERVED = {"user", "order", "group", "table", "select", "from", "where", "end", "check", "column", "default",
             "references", "primary", "foreign", "limit", "offset", "all", "analyse", "analyze", "and", "any",
             "as", "asc", "desc", "both", "case", "cast", "create", "current_user", "distinct", "do", "else",
             "except", "false", "fetch", "for", "grant", "having", "in", "initially", "intersect", "into",
             "leading", "not", "null", "on", "only", "or", "placing", "returning", "session_user", "some",
             "symmetric", "then", "to", "trailing", "true", "union", "unique", "using", "variadic", "when",
             "window", "with"}


class TabularError(Exception):
    pass


@dataclass
class ParsedColumn:
    source_name: str          # header as it appeared in the file
    name: str                 # clean, unique, lowercase snake_case
    type: str                 # one of PG_TYPES (inferred; the user may override)


@dataclass
class ParsedTable:
    name: str                 # clean table name
    source: str               # "orders.csv" or "book.xlsx / Sheet1"
    columns: list[ParsedColumn]
    rows: list[list[str | None]] = field(default_factory=list)   # text cells; None = NULL

    def column(self, name: str) -> ParsedColumn:
        return next(c for c in self.columns if c.name == name)

    def values(self, name: str) -> list[str | None]:
        i = [c.name for c in self.columns].index(name)
        return [r[i] for r in self.rows]


# --------------------------------------------------------------------------- names

def clean_identifier(raw: str, *, fallback: str = "col") -> str:
    """'Order Date' -> 'order_date'. Always a safe lowercase identifier of at most 50 characters."""
    s = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"{fallback}_{s}"
    if s in _RESERVED:
        s += "_"
    return s[:50]


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            n = f"{n[:45]}_{seen[n]}"
        seen.setdefault(n, 1)
        out.append(n)
    return out


# --------------------------------------------------------------------------- reading

def parse_file(path: Path, *, max_bytes: int | None = None) -> list[ParsedTable]:
    path = Path(path)
    if not path.is_file():
        raise TabularError(f"File not found: {path}")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise TabularError(f"{path.name} is larger than {max_bytes // (1024 * 1024)} MB.")
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        return [_parse_csv(path)]
    if ext == ".xlsx":
        return _parse_xlsx(path)
    raise TabularError(f"Unsupported file type '{ext}'. Use .csv, .xlsx, .sql or .sql.gz.")


def _parse_csv(path: Path) -> ParsedTable:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text, newline=""), dialect) if any(c.strip() for c in r)]
    return _build(path.stem, path.name, rows)


def _parse_xlsx(path: Path) -> list[ParsedTable]:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(path, read_only=True, data_only=True, keep_vba=False)  # cached values; no macros
    except Exception as e:
        raise TabularError(f"Could not read {path.name}: {e}") from e
    tables = []
    try:
        for ws in wb.worksheets:
            rows = [[_cell_text(c) for c in row] for row in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c not in (None, "") for c in r)]
            if len(rows) >= 2:
                tables.append(_build(ws.title, f"{path.name} / {ws.title}", rows))
    finally:
        wb.close()
    if not tables:
        raise TabularError(f"{path.name} has no sheet with a header row and at least one data row.")
    return tables


def _cell_text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, (date, time)):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return str(v)


def _build(base_name: str, source: str, rows: list[list]) -> ParsedTable:
    if len(rows) < 2:
        raise TabularError(f"{source}: needs a header row and at least one data row.")
    header, data = rows[0], rows[1:]
    if len(data) > MAX_ROWS_PER_TABLE:
        raise TabularError(f"{source}: more than {MAX_ROWS_PER_TABLE:,} rows.")
    width = len(header)
    if width > MAX_COLUMNS:
        raise TabularError(f"{source}: more than {MAX_COLUMNS} columns.")
    names = _unique([clean_identifier(h if str(h or "").strip() else f"column_{i + 1}")
                     for i, h in enumerate(header)])
    cells = [[_norm(c) for c in (list(r) + [None] * width)[:width]] for r in data]
    columns = [ParsedColumn(str(header[i] or "").strip(), names[i], infer_type([r[i] for r in cells]))
               for i in range(width)]
    return ParsedTable(clean_identifier(base_name, fallback="table"), source, columns, cells)


def _norm(c) -> str | None:
    if c is None:
        return None
    s = str(c).strip()
    return s if s != "" and s.lower() not in ("null", "n/a", "na", "nan") else None


def unique_table_names(tables: list[ParsedTable]) -> None:
    for t, name in zip(tables, _unique([t.name for t in tables])):
        t.name = name


# --------------------------------------------------------------------------- types

def infer_type(values: list[str | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "text"
    lowered = {v.lower() for v in vals}
    if lowered <= {"true", "false"} or lowered <= {"yes", "no"}:
        return "boolean"
    if all(_INT.match(v) and not _leading_zero(v) for v in vals):
        return "integer" if all(abs(int(v)) < 2**31 for v in vals) else "bigint"
    if all(_NUM.match(v) and not _leading_zero(v) for v in vals):
        return "numeric"
    if all(_DATE.match(v) and _valid_date(v) for v in vals):
        return "date"
    if all(_TIMESTAMP.match(v) and _valid_date(v[:10]) for v in vals):
        return "timestamp"
    return "text"


def _leading_zero(v: str) -> bool:  # '007' and '0123' are codes, not numbers
    d = v.lstrip("+-")
    return len(d) > 1 and d[0] == "0" and d[1] != "."


def _valid_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def coerces(values: list[str | None], pg_type: str) -> bool:
    """Whether every non-null value can be stored as pg_type (used to validate user overrides)."""
    vals = [v for v in values if v is not None]
    if pg_type == "text":
        return True
    if pg_type in ("integer", "bigint"):
        lim = 2**31 if pg_type == "integer" else 2**63
        return all(_INT.match(v) and abs(int(v)) < lim for v in vals)
    if pg_type == "numeric":
        try:
            [Decimal(v) for v in vals]
            return True
        except InvalidOperation:
            return False
    if pg_type == "boolean":
        return all(v.lower() in _TRUE | _FALSE for v in vals)
    if pg_type == "date":
        return all(_DATE.match(v) and _valid_date(v) for v in vals)
    if pg_type == "timestamp":
        return all(_TIMESTAMP.match(v) and _valid_date(v[:10]) for v in vals)
    return False
