import matplotlib.pyplot as plt

# Dummy data
entry_datetimes = ['08:49.3', '08:49.3', '08:49.3', '09:27.4', '09:27.4', '09:27.4']
exit_datetimes = ['08:53.0', '08:55.8', '12:25.8', '09:32.7', '12:33.1', '15:34.4']
time_spent = [3.692647934, 6.467283964, 216.5019071, 5.2937572, 185.7116661, 367.0112813]
zones = ['C', 'B', 'A', 'C', 'B', 'A']

# Convert time spent to minutes
time_spent_minutes = [ts / 60 for ts in time_spent]

# Calculate average waiting time in Zone B to reach Zone A
zone_b_time = [time_spent_minutes[i] for i, z in enumerate(zones) if z == 'B']
zone_a_time = [time_spent_minutes[i] for i, z in enumerate(zones) if z == 'A']
average_waiting_time = sum(zone_b_time) / len(zone_b_time)

# Calculate total time spent in each zone
zone_totals = {}
for zone in set(zones):
    zone_times = [time_spent_minutes[i] for i, z in enumerate(zones) if z == zone]
    zone_totals[zone] = sum(zone_times)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the line graph for each zone
for zone in set(zones):
    x = [i for i, z in enumerate(zones) if z == zone]
    y = [time_spent_minutes[i] for i in x]
    plt.plot(x, y, label=f'Zone {zone}')

# Add data labels with zone totals
for zone, total_time in zone_totals.items():
    plt.text(len(zones) - 1, total_time, f'Total Time in Zone {zone}: {total_time:.2f} mins', ha='right', va='bottom')

plt.xlabel('Data Points')
plt.ylabel('Number of Users (in minutes)')
plt.title('User Activity by Zone')
plt.legend()
plt.grid(True)

# Display average waiting time in Zone B to reach Zone A
plt.annotate(f'Average Waiting Time (B to A): {average_waiting_time:.2f} mins', xy=(0.5, 0.1), xycoords='axes fraction', fontsize=12, ha='center')

plt.show()
