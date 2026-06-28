from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class VerificationError(Exception):
    pass


@dataclass(frozen=True)
class Change:
    table: str
    pk: str
    field: str
    after: Any


@dataclass
class IgnoreConfig:
    tables: set[str] = field(default_factory=set)
    fields: dict[str, set[str]] = field(default_factory=dict)


Record = dict[str, Any]
Tables = dict[str, dict[str, Record]]


def _format(change: Change) -> str:
    return f"{change.table}[{change.pk}].{change.field}={change.after!r}"


class RecordView:
    def __init__(self, table: str, pk: str, record: Record) -> None:
        self._table = table
        self._pk = pk
        self._record = record

    def assert_eq(self, field_name: str, value: Any) -> "RecordView":
        actual = self._record.get(field_name)
        if actual != value:
            raise VerificationError(f"{self._table}[{self._pk}].{field_name} == {actual!r}, expected {value!r}")
        return self


class TableView:
    def __init__(self, name: str, rows: dict[str, Record]) -> None:
        self._name = name
        self._rows = rows

    def row(self, pk: str) -> RecordView:
        return RecordView(self._name, pk, self._rows.get(pk, {}))


class Diff:
    def __init__(self, changes: list[Change]) -> None:
        self._changes = changes

    @property
    def changes(self) -> list[Change]:
        return list(self._changes)

    def expect_only(self, expected: Iterable[Change]) -> None:
        actual_set = set(self._changes)
        expected_set = set(expected)
        if actual_set != expected_set:
            unexpected = sorted(_format(c) for c in actual_set - expected_set)
            missing = sorted(_format(c) for c in expected_set - actual_set)
            raise VerificationError(f"unexpected changes: {unexpected}; missing: {missing}")


class StateSnapshot:
    def __init__(self, tables: Tables) -> None:
        self._tables = tables

    def table(self, name: str) -> TableView:
        return TableView(name, self._tables.get(name, {}))

    def diff(self, other: "StateSnapshot", ignore: IgnoreConfig | None = None) -> Diff:
        ignore = ignore or IgnoreConfig()
        changes: list[Change] = []
        for table_name in set(self._tables) | set(other._tables):
            if table_name in ignore.tables:
                continue
            ignored = ignore.fields.get(table_name, set())
            before = self._tables.get(table_name, {})
            after = other._tables.get(table_name, {})
            for pk in set(before) | set(after):
                before_row = before.get(pk, {})
                after_row = after.get(pk, {})
                for field_name in set(before_row) | set(after_row):
                    if field_name in ignored:
                        continue
                    if before_row.get(field_name) != after_row.get(field_name):
                        changes.append(Change(table_name, pk, field_name, after_row.get(field_name)))
        return Diff(changes)
