import pandas as pd
import matplotlib.pyplot as plt
import os

from src.model_training.training import log_plots


def test_log_plots_creates_correct_plot():
    # Arrange
    df_predict = pd.DataFrame({
        'time': [0, 1, 2],
        'label1': [0, 1, 2],
        'label2': [2, 1, 0]
    })
    df_obs = pd.DataFrame({
        'time': [0, 1, 2],
        'label1': [0, 1, 2],
        'label2': [2, 1, 0]
    })
    labels = ['label1', 'label2']
    colors = ['red', 'black']
    plot_name = 'test_plot'
    epoch = 0

    # Act
    log_plots(df_predict, df_obs, labels, colors, plot_name, epoch, save_plot=True)

    # Assert
    fig = plt.gcf()
    assert len(fig.axes) == 2  # Check that there are two subplots
    for ax, label, color in zip(fig.axes, labels, colors):
        lines = ax.get_lines()
        assert len(lines) == 2  # Check that each subplot has two lines
        for line in lines:
            assert line.get_color() == color  # Check that the lines have the correct color
            assert line.get_label().startswith('model_' + label) or line.get_label().startswith('obs_' + label)
            # Check that the lines have the correct label


def test_log_plots_saves_plot_to_file():
    # Arrange
    df_predict = pd.DataFrame({
        'time': [0, 1, 2],
        'label1': [0, 1, 2],
        'label2': [2, 1, 0]
    })
    df_obs = pd.DataFrame({
        'time': [0, 1, 2],
        'label1': [0, 1, 2],
        'label2': [2, 1, 0]
    })
    labels = ['label1', 'label2']
    colors = ['red', 'black']
    plot_name = 'test_plot'
    epoch = 0

    # Act
    log_plots(df_predict, df_obs, labels, colors, plot_name, epoch, save_plot=True)

    # Assert
    assert os.path.isfile(f'fit_plot_{plot_name}.png')  # Check that the file was created

    # Clean up
    os.remove(f'fit_plot_{plot_name}.png')


test_log_plots_creates_correct_plot()
test_log_plots_saves_plot_to_file()
