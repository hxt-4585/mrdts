"""Live Matplotlib monitor for train_rl_epochs.py; runs independently of training."""

import argparse
import csv
import io
import math
from pathlib import Path


def read_epochs(path):
    """Read only complete CSV lines; reload to reflect checkpoint resume rollback."""
    try:
        content = Path(path).read_text(encoding='utf-8')
    except FileNotFoundError:
        return []
    content = content[:content.rfind('\n') + 1]
    rows = []
    for row in csv.DictReader(io.StringIO(content)):
        parsed = dict(epoch=int(row['epoch']), stage=row['stage'], steps=int(row['steps']),
                      mean_reward=float(row['mean_reward']), epoch_return=float(row['epoch_return']),
                      validation_mean_reward=(float(row['validation_mean_reward'])
                                              if row.get('validation_mean_reward') else None))
        values = [parsed['mean_reward'], parsed['epoch_return']]
        if parsed['validation_mean_reward'] is not None:
            values.append(parsed['validation_mean_reward'])
        if (parsed['steps'] < 1 or parsed['epoch'] < 1 or parsed['stage'] not in ('member', 'master')
                or not all(math.isfinite(value) for value in values)
                or not math.isclose(parsed['epoch_return'], parsed['mean_reward'] * parsed['steps'],
                                    rel_tol=1e-7, abs_tol=1e-8)):
            raise ValueError(f"Invalid reward data at epoch {parsed['epoch']}")
        if rows and parsed['epoch'] <= rows[-1]['epoch']:
            raise ValueError('Epoch numbers must be strictly increasing')
        rows.append(parsed)
    return rows


def draw(figure, axes, rows):
    """Draw raw logged rewards; validation and training use separate series."""
    from matplotlib.ticker import MaxNLocator
    for axis, title, label in zip(axes, ('Mean step reward', 'Epoch return'),
                                  ('Mean reward', 'Sum of step rewards')):
        axis.clear()
        axis.set(title=title, xlabel='Epoch', ylabel=label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    if rows:
        epochs = [row['epoch'] for row in rows]
        for axis, field in zip(axes, ('mean_reward', 'epoch_return')):
            axis.plot(epochs, [row[field] for row in rows], color='#4477AA',
                      marker='.', linewidth=1.2, label='Training')
        validation = [row for row in rows if row['validation_mean_reward'] is not None]
        if validation:
            axes[0].plot([row['epoch'] for row in validation],
                         [row['validation_mean_reward'] for row in validation],
                         color='#AA3377', marker='o', markersize=4, linestyle='--', label='Validation')
        # Label the switch because sampling/frozen-policy behavior differs by stage.
        for previous, current in zip(rows, rows[1:]):
            if previous['stage'] != current['stage']:
                for axis in axes:
                    axis.axvline((previous['epoch'] + current['epoch']) / 2,
                                 color='#888888', linestyle=':', linewidth=1)
        axes[0].legend(frameon=False)
        last = rows[-1]
        status = (f"Epoch {last['epoch']} | {last['stage']} | {last['steps']} steps | "
                  f"mean {last['mean_reward']:.6f} | return {last['epoch_return']:.3f}")
    else:
        status = 'Waiting for the first completed epoch (initial validation runs first)'
    figure.suptitle('Live epoch reward - higher is better\n' + status, fontsize=10)
    figure.canvas.draw_idle()
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Training output directory')
    parser.add_argument('--interval', type=float, default=2., help='Refresh interval in seconds')
    parser.add_argument('--once', action='store_true', help='Export once without opening a window')
    parser.add_argument('--save', type=Path, help='PNG path; default: RUN/live-reward.png')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    if not args.once:
        print('On Windows, close this monitor before resuming training; reopen after training steps resume.',
              flush=True)
    import matplotlib
    matplotlib.use('Agg' if args.once else 'TkAgg', force=True)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    output = args.save or args.run / 'live-reward.png'
    previous_rows = None
    previous_error = None

    def refresh():
        nonlocal previous_rows, previous_error
        try:
            rows = read_epochs(args.run / 'epochs.csv')
            if rows != previous_rows or previous_error is not None:
                print(draw(figure, axes, rows), flush=True)
                # Do not create files in an empty future training directory: the
                # trainer requires it to be empty when starting a fresh run.
                if rows:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(output, dpi=150)
                previous_rows = rows
                previous_error = None
        except (OSError, ValueError, KeyError, TypeError) as error:
            if args.once:
                raise
            message = f'Log read/export error; retrying: {error}'
            if message != previous_error:
                print(message, flush=True)
                figure.suptitle(message, fontsize=9)
                figure.canvas.draw_idle()
                previous_error = message

    refresh()
    if args.once:
        plt.close(figure)
        return
    figure.canvas.manager.set_window_title(f'RL reward - {args.run.resolve()}')
    timer = figure.canvas.new_timer(interval=max(1, int(args.interval * 1000)))
    timer.add_callback(refresh)
    timer.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close(figure)
    finally:
        timer.stop()


if __name__ == '__main__':
    main()
