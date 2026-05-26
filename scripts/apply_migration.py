"""Apply one SQL migration file against the configured DATABASE_URL.

Usage:
    .venv/Scripts/python scripts/apply_migration.py supabase/migrations/0004_add_category_to_products.sql

The script reads the file, splits on `;` *only at statement boundaries* (a
naive but workable approach for our migrations since none of them contain
multi-statement procedures), and executes each statement individually via
asyncpg through SQLAlchemy. Using single-statement execution keeps PgBouncer
(transaction pooler) happy and surfaces failure on the offending statement.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.storage.db import _get_engine


def _split_statements(sql: str) -> list[str]:
    """Strip comment-only lines and split on `;` at top level.

    Walks the SQL character-by-character, tracking whether we're inside a
    ``$$ ... $$`` dollar-quoted block (used by plpgsql function bodies).
    Only top-level semicolons close a statement.
    """
    # Drop pure-comment lines so they don't generate empty statements.
    lines = [
        l for l in sql.splitlines()
        if l.strip() and not l.strip().startswith("--")
    ]
    cleaned = "\n".join(lines)

    statements: list[str] = []
    current: list[str] = []
    in_dollar = False
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        # Detect $$ delimiter (toggle in/out of plpgsql body).
        if ch == "$" and i + 1 < len(cleaned) and cleaned[i + 1] == "$":
            current.append("$$")
            in_dollar = not in_dollar
            i += 2
            continue
        if ch == ";" and not in_dollar:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


async def apply_file(path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    statements = _split_statements(sql)
    if not statements:
        print(f"(empty migration) {path.name}")
        return

    engine = _get_engine()
    async with engine.begin() as conn:
        for i, stmt in enumerate(statements, start=1):
            preview = stmt.split("\n", 1)[0][:72]
            print(f"  [{i:>2}/{len(statements)}] {preview}")
            await conn.execute(text(stmt))
    print(f"applied {path.name} ({len(statements)} statements)")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: apply_migration.py <path_to_sql_file>")
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"migration file not found: {path}")
        sys.exit(1)
    print(f"applying {path.name} against database...")
    asyncio.run(apply_file(path))


if __name__ == "__main__":
    main()
