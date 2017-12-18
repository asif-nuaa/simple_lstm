from datetime import datetime, timedelta
from matplotlib import pylab as plt
from matplotlib import dates as mdates

import numpy as np

if __name__ == '__main__':
    start_time = datetime(2000, 5, 13)
    num_times = 24 * 2 * 15
    delta_time = timedelta(minutes=30)
    times = [start_time + i * delta_time for i in range(num_times)]

    signal = np.sin(np.arange(num_times))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    hour_locator = mdates.DayLocator(interval=1)

    formatter = mdates.DateFormatter("%d. %b '%y")

    ax.plot(times, signal, label="Prediction")
    ax.legend()

    ax.xaxis.set_major_locator(hour_locator)
    ax.xaxis.set_major_formatter(formatter)


    # Plot a grid
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='both', axis='x')


    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()