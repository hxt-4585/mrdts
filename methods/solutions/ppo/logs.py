"""Streaming run logs and checkpoint-boundary reconciliation."""

import csv
import time


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reconcile_logs(output, completed_epoch, *, master_started=True):
    for relative in ('metrics.csv', 'training/epochs.csv', 'training/updates.csv', 'training/validation.csv'):
        path = output / relative
        if not path.exists():
            continue
        # A crash may leave an incomplete final CSV row. Preserve all original bytes.
        content = path.read_text(encoding='utf-8')
        complete = content[:content.rfind('\n')+1]
        import io
        reader = csv.DictReader(io.StringIO(complete))
        rows = list(reader)
        kept = [row for row in rows if int(row['epoch']) <= completed_epoch
                and not (relative == 'training/validation.csv' and not master_started
                         and row['stage'] == 'master')]
        if len(kept) == len(rows) and complete == content:
            continue
        backup = path.with_name(f'{path.stem}.uncheckpointed-{time.time_ns()}.csv')
        backup.write_text(content, encoding='utf-8')
        temporary = path.with_suffix('.resume.tmp')
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        temporary.replace(path)


def metric_rows(output):
    with (output/'metrics.csv').open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            yield {key: (float(value) if value else None) for key,value in row.items()
                   if key not in ('stage',)}
