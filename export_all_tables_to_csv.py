#!/usr/bin/env python3
"""
导出当前 Django 数据库中的全部表到 CSV 文件。

用法:
  python export_all_tables_to_csv.py
  python export_all_tables_to_csv.py --output-dir ./exports/db_csv
  python export_all_tables_to_csv.py --batch-size 5000
  python export_all_tables_to_csv.py --table simulation_api_simrealtimelinkstate --max-rows-per-table 100000
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import decimal
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all DB tables to CSV files.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="导出目录。默认: ./exports/db_csv_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="每批抓取行数（默认 2000）",
    )
    parser.add_argument(
        "--table",
        action="append",
        default=[],
        help="仅导出指定表名（可重复传多次）",
    )
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        default=0,
        help="每张表最多导出多少行。<=0 表示不限制（默认不限制）",
    )
    return parser.parse_args()


def safe_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return value


def setup_django(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "satellites_sim_backend.settings")

    import django

    django.setup()


def export_table_paginated(
    table_name: str,
    csv_path: Path,
    batch_size: int,
    max_rows_per_table: int = 0,
) -> int:
    from django.db import connection

    quoted_table = connection.ops.quote_name(table_name)
    row_count = 0

    with connection.cursor() as cursor:
        pk_column = connection.introspection.get_primary_key_column(cursor, table_name)

    with connection.cursor() as cursor, csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as fp:
        writer = csv.writer(fp)
        cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 0")
        headers = [col[0] for col in (cursor.description or [])]
        writer.writerow(headers)

        if pk_column:
            quoted_pk = connection.ops.quote_name(pk_column)
            last_pk: Any = None
            while True:
                remaining = (
                    (max_rows_per_table - row_count) if max_rows_per_table > 0 else batch_size
                )
                if max_rows_per_table > 0 and remaining <= 0:
                    break
                current_limit = min(batch_size, remaining) if max_rows_per_table > 0 else batch_size

                if last_pk is None:
                    cursor.execute(
                        f"SELECT * FROM {quoted_table} ORDER BY {quoted_pk} ASC LIMIT %s",
                        [current_limit],
                    )
                else:
                    cursor.execute(
                        f"SELECT * FROM {quoted_table} "
                        f"WHERE {quoted_pk} > %s ORDER BY {quoted_pk} ASC LIMIT %s",
                        [last_pk, current_limit],
                    )
                rows = cursor.fetchall()
                if not rows:
                    break
                pk_idx = headers.index(pk_column)
                for row in rows:
                    writer.writerow([normalize_value(v) for v in row])
                row_count += len(rows)
                last_pk = rows[-1][pk_idx]
        else:
            offset = 0
            while True:
                remaining = (
                    (max_rows_per_table - row_count) if max_rows_per_table > 0 else batch_size
                )
                if max_rows_per_table > 0 and remaining <= 0:
                    break
                current_limit = min(batch_size, remaining) if max_rows_per_table > 0 else batch_size

                cursor.execute(
                    f"SELECT * FROM {quoted_table} LIMIT %s OFFSET %s",
                    [current_limit, offset],
                )
                rows = cursor.fetchall()
                if not rows:
                    break
                for row in rows:
                    writer.writerow([normalize_value(v) for v in row])
                got = len(rows)
                row_count += got
                offset += got

    return row_count


def export_all_tables(
    output_dir: Path,
    batch_size: int,
    only_tables: list[str],
    max_rows_per_table: int,
) -> None:
    from django.db import connection

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    with connection.cursor() as cursor:
        discovered = connection.introspection.table_names(cursor)
    if only_tables:
        wanted = set(only_tables)
        table_names = [name for name in discovered if name in wanted]
        missing = sorted(wanted - set(table_names))
        for miss in missing:
            summary.append({"table": miss, "file": "", "rows": 0, "error": "table not found"})
    else:
        table_names = discovered

    print(f"[INFO] Found {len(table_names)} tables.")

    for index, table_name in enumerate(table_names, start=1):
        csv_name = f"{index:03d}_{safe_filename(table_name)}.csv"
        csv_path = output_dir / csv_name
        row_count = 0

        print(f"[INFO] Exporting [{index}/{len(table_names)}] {table_name} -> {csv_name}")

        try:
            row_count = export_table_paginated(
                table_name=table_name,
                csv_path=csv_path,
                batch_size=batch_size,
                max_rows_per_table=max_rows_per_table,
            )

            summary.append(
                {
                    "table": table_name,
                    "file": csv_name,
                    "rows": row_count,
                    "max_rows_per_table": max_rows_per_table if max_rows_per_table > 0 else None,
                }
            )
            print(f"[INFO]   done rows={row_count}")
        except Exception as exc:
            summary.append(
                {
                    "table": table_name,
                    "file": csv_name,
                    "rows": row_count,
                    "error": str(exc),
                }
            )
            print(f"[ERROR]  failed table={table_name}: {exc}")

    summary_path = output_dir / "_export_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "tables": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Summary written: {summary_path}")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        print("[ERROR] --batch-size must be > 0")
        return 2
    if args.max_rows_per_table < 0:
        print("[ERROR] --max-rows-per-table must be >= 0")
        return 2

    project_root = Path(__file__).resolve().parent
    setup_django(project_root)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (project_root / "exports" / f"db_csv_{stamp}").resolve()

    export_all_tables(
        output_dir=output_dir,
        batch_size=args.batch_size,
        only_tables=args.table,
        max_rows_per_table=args.max_rows_per_table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
