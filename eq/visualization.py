from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd

from eq.data import Sequence
from eq.data import Catalog

__all__ = [
    "visualize_sequence",
    "visualize_trajectories",
    "visualize_catalog",
]


def visualize_catalog(
    catalog: Catalog,
    ax=None,
    figsize: tuple = (9, 3),
    dpi: int = 100,
    event_color="C0",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    plot_style: str = "scatter",
):
    if t_start is None:
        t_start = catalog.metadata["start_ts"]
    if t_end is None:
        t_end = catalog.metadata["end_ts"]

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    def day2timestamp(N, start_time_stamp=catalog.metadata["start_ts"]):
        return start_time_stamp + N * pd.Timedelta(1, "D")

    plt_fun = {
        "hist": lambda ax, seq: ax.hist(
            day2timestamp(seq.arrival_times.numpy()), 100, alpha=0.5
        ),
        "scatter": lambda ax, seq: visualize_sequence(
            seq,
            ax,
            event_color=event_color,
            t_start=t_start,
            t_end=t_end,
            show_legend=False,
            time_transform=day2timestamp,
        ),
    }

    plt_fun[plot_style](ax, catalog.full_sequence)

    for k in ["start_ts", "end_ts", "train_start_ts", "val_start_ts", "test_start_ts"]:
        ax.axvline(
            catalog.metadata[k], ls="--", c=next(ax._get_lines.prop_cycler)["color"]
        )
    ax.text(
        1,
        1,
        f'Catalog: {catalog.metadata["name"]}\n$M_c$: {catalog.metadata["mag_completeness"]}',
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    ax.margins(x=0)
    return ax


def visualize_sequence(
    seq: Sequence,
    ax=None,
    show_nll: bool = False,
    mag_completeness: Optional[float] = None,
    figsize: tuple = (9, 3),
    dpi: int = 100,
    event_color="C0",
    nll_interval_color="C1",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    show_legend: bool = True,
    time_transform=lambda x: x,
):
    if t_start is None:
        t_start = seq.t_start
    if t_end is None:
        t_end = seq.t_end

    t = time_transform(seq.arrival_times.cpu().numpy())

    mag = seq.mag.cpu().numpy()
    if mag_completeness is None:
        mag_completeness = np.min(mag)

    if ax is None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
    ax.scatter(
        t,
        mag,
        s=np.exp(mag - mag_completeness),
        c=event_color,
        alpha=0.7,
        label="Events",
    )
    _, y_max = ax.get_ylim()
    if show_nll:
        ax.add_patch(
            Rectangle(
                [seq.t_nll_start, mag_completeness],
                t_end - seq.t_nll_start,
                y_max - mag_completeness,
                alpha=0.2,
                facecolor=nll_interval_color,
                label="Interval on which NLL is computed",
            )
        )
    ax.set_xlabel("Time (days)")
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(mag_completeness, y_max)
    ax.set_ylabel("Magnitude")
    if show_legend:
        ax.legend(loc="upper center", ncol=2, bbox_to_anchor=[0.5, 1.15])
    return ax


def plot_counting_process(seq, ax=None, color="k", alpha=1.0, T0=None, T=None):
    if ax is None:
        ax = plt.gca()

    t = np.append(seq.arrival_times.numpy(), T)
    t = np.insert(t, 0, T0, axis=0)

    N = np.arange(len(t) - 1)
    N = np.append(N, N[-1])
    ax.plot(t, N, c=color, alpha=alpha)


def visualize_trajectories(
    seq: Sequence,
    forecast: List[Sequence],
    ax=None,
    figsize: tuple = (6, 3),
    dpi: int = 100,
    event_color="C0",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    t_before: Optional[float] = None,
    num_examples: int = 10,
    add_mag_threshold: Optional[float] = None,
):
    """Show an example visualization of simulated catalog continuations"""
    if (t_start is None) and (t_end is None):
        sample_forecast = forecast[0]
        t_start = sample_forecast.t_start
        t_end = sample_forecast.t_end

    duration = sample_forecast.t_end - sample_forecast.t_start

    if t_before is None:
        t_before = duration

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=(3, 1.5),
            left=0.1,
            right=0.85,
            bottom=0.2,
            top=0.9,
            wspace=0.02,
            hspace=0.1,
        )

        axA = fig.add_subplot(gs[0])
        axAA = axA.twinx()
        axB = fig.add_subplot(gs[1], sharey=axAA)
    else:
        print("Bold choice")
        assert len(ax) == 2, "len(ax) must be two to generate both subplots"
        axA, axB = ax
        axAA = axA.twinx()

    axA.margins(x=0)

    assert num_examples <= len(
        forecast
    ), "num_examples must be <= to the number of simulations in the forecast"

    s_viz = seq.get_subsequence(
        max([0, t_start - t_before]), min(seq.t_end, t_start + duration)
    ).cpu()
    s_obs = seq.get_subsequence(t_start, min(seq.t_end, t_start + duration)).cpu()

    axA.axvline(t_start, c="k", lw=1, ls="--")
    if seq.t_end < t_start + duration:
        axA.axvline(seq.t_end, c="r", lw=1, alpha=0.5)
        axA.annotate(
            "Prospective interval",
            (0.8, 0.1),
            xycoords="axes fraction",
            c="r",
            ha="center",
        )

    axA.annotate(
        "Forecast interval", (0.75, 0.9), xycoords="axes fraction", c="k", ha="center"
    )

    visualize_sequence(seq=s_viz, ax=axA, event_color=event_color, show_legend=False)

    plot_counting_process(s_obs, axAA, "k", T0=t_start, T=t_start + duration)
    [
        plot_counting_process(
            i_samp.cpu(), axAA, "k", 0.2, T0=t_start, T=t_start + duration
        )
        for i_samp in forecast[:num_examples]
    ]
    axAA.get_yaxis().set_visible(False)
    axA.set_xticks(axA.get_xticks()[1::2])
    axA.set_xlim([axA.get_xlim()[0], t_start + duration])
    axA.margins(x=0)

    # B:

    cummulative_no_of_events = np.array(
        [len(forecast[i]) for i in range(len(forecast))]
    )

    ylim = [0, np.quantile(cummulative_no_of_events, 0.99)]

    axB.hist(
        cummulative_no_of_events,
        bins=50,
        range=(int(ylim[0]), int(ylim[1])),
        orientation="horizontal",
        facecolor="k",
        alpha=0.2,
        linewidth=1,
        edgecolor="w",
        label="Simulated",
    )

    if add_mag_threshold:
        is_bigger = np.array(
            [
                (forecast[i].mag.max() > add_mag_threshold).any().item()
                for i in range(len(forecast))
            ]
        )

        axB.hist(
            cummulative_no_of_events[is_bigger],
            bins=50,
            range=(int(ylim[0]), int(ylim[1])),
            orientation="horizontal",
            facecolor="r",
            alpha=0.2,
            linewidth=1,
            edgecolor="w",
            label=f"{sum(is_bigger)/len(forecast)*100:0.1f}% contain M{add_mag_threshold}+",
        )

    axB.axhline(len(s_obs), c="k", label="Observed")
    axB.set_ylabel("# of events\nin the interval")
    axB.yaxis.set_ticks_position("right")
    axB.yaxis.set_label_position("right")
    axB.get_xaxis().set_visible(False)
    axB.set_ylim(ylim)
    axB.legend(fontsize=7, loc="upper right")
    plt.show()

    return [axA, axB]
