import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import namedtuple


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


def _plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3




EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, smoothing_window=10, axes=None, title_suffix=""):
    """
    Plot the episode statistics using the provided axes.

    Args:
        stats: EpisodeStats object containing episode lengths and rewards.
        smoothing_window: Window size for smoothing the rewards.
        axes: List of axes to plot on. If None, new figures will be created.
        title_suffix: Suffix to add to the plot titles.

    Returns:
        List of figure handles if axes is None, otherwise None.
    """
    if axes is None:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig3, ax3 = plt.subplots(figsize=(10, 5))
    else:
        ax1, ax2, ax3 = axes

    # Plot the episode length over time
    ax1.plot(stats.episode_lengths)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Length")
    ax1.set_title(f"Episode Length over Time {title_suffix}")

    # Plot the episode reward over time
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax2.plot(rewards_smoothed)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Reward (Smoothed)")
    ax2.set_title(f"Episode Reward over Time (Smoothed){title_suffix}")

    # Plot time steps and episode number
    ax3.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Episode")
    ax3.set_title(f"Episode per Time Step {title_suffix}")

    if axes is None:
        return [fig1, fig2, fig3]
