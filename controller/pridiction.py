import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

def generate_popular_times():
    hours = list(range(9, 24)) + list(range(0, 1))  # Hours from 9am to 12am
    popularity = []

    for hour in hours:
        # Generate a random number between 0 and 100 to represent popularity percentage
        popularity.append(random.randint(0, 100))

    return popularity

def plot_popular_times(popularity):
    fig, ax = plt.subplots()

    peak_hours = [14, 17, 18]  # Define your peak hours here

    # Plot the popular times as a bar graph with different colors for peak hours
    bars = plt.bar(range(len(popularity)), popularity, align='center', color='gray', alpha=0.8)

    for i, hour in enumerate(range(9, 24) + list(range(0, 1))):
        if hour in peak_hours:
            bars[i].set_color('red')  # Set color for peak hours
            bars[i].set_edgecolor('black')  # Add black edge color
            bars[i].set_linewidth(0.8)  # Adjust edge linewidth
            bars[i].set_zorder(2)  # Set higher z-order to make bars visible above grid lines
            bars[i].set_path_effects([Rectangle(0, 0, 1, 1, 1, facecolor='none', edgecolor='black', linewidth=0.8)])  # Add corner radius

    plt.xlabel('Hour')
    plt.ylabel('Popularity')
    plt.title('Popular Times')

    ax.set_xticks(range(len(popularity)))
    ax.set_xticklabels(list(range(9, 24)) + list(range(0, 1)))  # Update the concatenation

    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7, zorder=1)  # Set lower z-order for grid lines
    plt.show()

# Generate dummy popular times for a specific day
dummy_popular_times = generate_popular_times()

# Plot the dummy popular times
plot_popular_times(dummy_popular_times)
