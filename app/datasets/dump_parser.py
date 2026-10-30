"""Parse a plain-SQL pg_dump file into statements the loader can execute safely.

We deliberately do NOT pipe dumps into psql: a plain dump may contain psql meta-commands
such as `\\! <shell command>`, which psql would run on the app machine. Here, any
meta-command is a hard rejection (except pg_dump's own `\\restrict` / `\\unrestrict`).

Each statement gets one of three verdicts:
- run      : execute as-is (or lightly rewritten)
- skip     : harmless for a dump, but must not run as a non-superuser in our setup
             (ownership, grants, session timeouts)
- reject   : the whole dump is refused (server-level objects, roles, external files/programs)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal


class DumpRejected(Exception):
    def __init__(self, line: int, reason: str):
        super().__init__(f"line {line}: {reason}")
        self.line = line
        self.reason = reason


@dataclass
class Statement:
    sql: str
    line: int  # line where the statement starts
    copy_data: list[str] | None = field(default=None, repr=False)  # set for COPY ... FROM stdin

    @property
    def is_copy(self) -> bool:
        return self.copy_data is not None


Verdict = Literal["run", "skip", "reject"]

_ALLOWED_META = {"\\restrict", "\\unrestrict"}

_SKIP = [
    re.compile(r"^ALTER\b.*\bOWNER TO\b"),
    re.compile(r"^(GRANT|REVOKE)\b"),
    re.compile(r"^ALTER DEFAULT PRIVILEGES\b"),
    re.compile(r"^COMMENT ON EXTENSION\b"),
    re.compile(r"^SECURITY LABEL\b"),
    re.compile(r"^SET (SESSION AUTHORIZATION|ROLE)\b"),
    # The loader enforces its own limits; a dump must not lift them.
    re.compile(r"^SET (STATEMENT_TIMEOUT|LOCK_TIMEOUT|IDLE_IN_TRANSACTION_SESSION_TIMEOUT|TRANSACTION_TIMEOUT)\b"),
]

_REJECT = [
    (re.compile(r"^(CREATE|ALTER|DROP) (DATABASE|ROLE|USER|GROUP|TABLESPACE|SUBSCRIPTION|PUBLICATION"
                r"|EVENT TRIGGER|SERVER|FOREIGN DATA WRAPPER|USER MAPPING)\b"),
     "server-level objects are not allowed in a dataset dump"),
    (re.compile(r"^ALTER SYSTEM\b"), "ALTER SYSTEM is not allowed"),
    (re.compile(r"^(LOAD|CHECKPOINT)\b"), "server administration commands are not allowed"),
    (re.compile(r"^DO\b"), "anonymous code blocks (DO) are not allowed"),
    (re.compile(r"^COPY\b(?!.*\bFROM STDIN\b)"), "COPY is only allowed FROM stdin"),
    (re.compile(r"^(CREATE|DROP) (OR REPLACE )?(TRUSTED )?(PROCEDURAL )?LANGUAGE\b"),
     "languages are not allowed"),
]

_CREATE_EXTENSION = re.compile(r"^CREATE EXTENSION (?!IF NOT EXISTS)", re.IGNORECASE)
_COPY_FROM_STDIN = re.compile(r"^COPY\b.*\bFROM STDIN\b")
_DOLLAR_TAG = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def classify(sql: str) -> tuple[Verdict, str | None]:
    """Returns (verdict, reason). reason is set for skip and reject."""
    head = re.sub(r"\s+", " ", sql).strip().upper()
    for pattern, reason in _REJECT:
        if pattern.search(head):
            return "reject", reason
    for pattern in _SKIP:
        if pattern.search(head):
            return "skip", "ownership/privilege/timeout statement"
    return "run", None


def rewrite(sql: str) -> str:
    # Extensions come pre-installed from template1; make re-creation a no-op.
    return _CREATE_EXTENSION.sub("CREATE EXTENSION IF NOT EXISTS ", sql.lstrip(), count=1)


def parse_dump(lines: Iterable[str]) -> Iterator[Statement]:
    """Yield statements in order. Raises DumpRejected on meta-commands or malformed input."""
    splitter = _StatementSplitter()
    it = iter(lines)
    lineno = 0
    for line in it:
        lineno += 1
        if splitter.at_statement_start() and line.startswith("\\"):
            command = line.split(maxsplit=1)[0]
            if command in _ALLOWED_META:
                continue
            raise DumpRejected(lineno, f"psql meta-command '{command}' is not allowed")

        for sql, start_line in splitter.feed(line, lineno):
            if _COPY_FROM_STDIN.search(re.sub(r"\s+", " ", sql).upper()):
                data: list[str] = []
                for data_line in it:
                    lineno += 1
                    if data_line.rstrip("\r\n") == "\\.":
                        break
                    data.append(data_line)
                else:
                    raise DumpRejected(start_line, "COPY data block is not terminated by '\\.'")
                yield Statement(sql=sql, line=start_line, copy_data=data)
            else:
                yield Statement(sql=sql, line=start_line)

    if not splitter.at_statement_start():
        raise DumpRejected(lineno, "dump ends in the middle of a statement")


class _StatementSplitter:
    """Splits SQL text on top-level semicolons, aware of quotes, dollar quotes and comments."""

    def __init__(self) -> None:
        self._buf: list[str] = []
        self._start_line = 0
        self._quote: str | None = None      # "'" or '"'
        self._escape_string = False         # inside E'...' (backslash escapes)
        self._dollar: str | None = None     # active dollar-quote tag, e.g. "$_$"
        self._block_comment = 0             # nesting depth of /* */

    def at_statement_start(self) -> bool:
        return (
            not "".join(self._buf).strip()
            and self._quote is None and self._dollar is None and self._block_comment == 0
        )

    def feed(self, line: str, lineno: int) -> list[tuple[str, int]]:
        done: list[tuple[str, int]] = []
        if self.at_statement_start():
            self._start_line = lineno
        start = i = 0
        n = len(line)
        while i < n:
            c = line[i]
            if self._dollar:
                if line.startswith(self._dollar, i):
                    i += len(self._dollar)
                    self._dollar = None
                else:
                    i += 1
            elif self._block_comment:
                if line.startswith("*/", i):
                    self._block_comment -= 1
                    i += 2
                elif line.startswith("/*", i):
                    self._block_comment += 1
                    i += 2
                else:
                    i += 1
            elif self._quote:
                if self._escape_string and c == "\\":
                    i += 2
                elif c == self._quote:
                    if line.startswith(self._quote * 2, i):
                        i += 2
                    else:
                        self._quote = None
                        i += 1
                else:
                    i += 1
            elif line.startswith("--", i):
                # Drop the rest of the line (a comment), keep the newline.
                self._buf.append(line[start:i] + "\n")
                start = i = n
            elif line.startswith("/*", i):
                self._block_comment = 1
                i += 2
            elif c in ("'", '"'):
                self._quote = c
                self._escape_string = c == "'" and _is_e_prefix(line, i)
                i += 1
            elif c == "$" and (m := _DOLLAR_TAG.match(line, i)) and not _is_word_char(line, i - 1):
                self._dollar = m.group(0)
                i = m.end()
            elif c == ";":
                self._buf.append(line[start:i + 1])
                sql = "".join(self._buf).strip()
                if sql != ";":
                    done.append((sql, self._start_line))
                self._buf = []
                start = i + 1
                self._start_line = lineno
                i += 1
            else:
                i += 1
        if start < n:
            self._buf.append(line[start:])
        return done


def _is_word_char(s: str, i: int) -> bool:
    return i >= 0 and (s[i].isalnum() or s[i] == "_")


def _is_e_prefix(s: str, quote_pos: int) -> bool:
    return quote_pos >= 1 and s[quote_pos - 1] in "eE" and not _is_word_char(s, quote_pos - 2)
