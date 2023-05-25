import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates from May 26, 2023, to May 31, 2023
dates = pd.date_range(start='2023-05-26', end='2023-05-31', freq='D')

# Generate footfall predictions for each date
footfall_predictions = [2500, 3000, 3200, 3000, 2300, 1800]

# Set footfall patterns for each day
day_patterns = {
    'Monday': 'Medium',
    'Tuesday': 'Medium',
    'Wednesday': 'Medium',
    'Thursday': 'Medium',
    'Friday': 'Peak',
    'Saturday': 'Peak',
    'Sunday': 'Peak'
}

# Set peak hours for each day
peak_hours = {
    'Monday': [],
    'Tuesday': [],
    'Wednesday': [],
    'Thursday': [],
    'Friday': ['4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM'],
    'Saturday': ['11 AM', '12 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM'],
    'Sunday': ['11 AM', '12 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM']
}

# Define custom date labels
date_labels = ['May 26', 'May 27', 'May 28', 'May 29', 'May 30', 'May 31']

# Define notification status for each day
notification_status = {
    'May 26': 'Notification Sent',
    'May 27': 'Pending to Send',
    'May 28': 'Pending to Send'
}

# Define prediction accuracy information
accuracy_info = "Predicted with 82% accuracy based on last 2 years' footfall and sales data"

# Plotting the prediction graph
fig, ax = plt.subplots()

# Plotting the footfall predictions
ax.plot(range(len(dates)), footfall_predictions, color='skyblue', label='Footfall Predictions')

# Highlighting peak days and sending notifications to store manager
for i, date in enumerate(dates):
    day_name = date.strftime('%A')
    if day_name in day_patterns and day_patterns[day_name] == 'Peak':
        ax.axvline(i, color='orange', linestyle='--', alpha=0.5)
        if date.strftime('%b %d') in notification_status:
            notification_text = notification_status[date.strftime('%b %d')]
            ax.annotate(notification_text, xy=(i, footfall_predictions[i]), xytext=(i, footfall_predictions[i] + 100),
                        rotation=45, horizontalalignment='center', verticalalignment='bottom')

# Annotating peak hours
for i, date in enumerate(dates):
    day_name = date.strftime('%A')
    if day_name in peak_hours and day_patterns[day_name] == 'Peak':
        for hour in peak_hours[day_name]:
            ax.annotate(hour, xy=(i, footfall_predictions[i]), xytext=(i, footfall_predictions[i] + 200), rotation=45,
                        horizontalalignment='center', verticalalignment='bottom')

# Label the axes and give a title
ax.set_xlabel('Date')
ax.set_ylabel('Footfall')
ax.set_title('Footfall Predictions - May 26, 2023, to May 31, 2023')

# Set the x-axis tick labels
ax.set_xticks(range(len(dates)))
ax.set_xticklabels(date_labels)

# Add a text box with prediction accuracy information
ax.text(0.5, 0.95, accuracy_info, transform=ax.transAxes, ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))

# Rotate x-axis tick labels for better visibility
plt.xticks(rotation=45)

# Display the graph
plt.tight_layout()
plt.show()
