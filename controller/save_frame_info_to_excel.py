import pandas as pd
import os


def save_frame_info_to_excel(camera_id, camera_name, frame, timestamp, dwell_time, object_id, checkout_area):
    # Create a DataFrame with the frame information
    columns = ['Camera ID', 'Camera Name', 'Frame', 'Timestamp', 'Dwell Time (sec)', 'UniqueId', 'Area']
    frame_data = pd.DataFrame([[camera_id, camera_name, frame, timestamp, dwell_time, object_id, checkout_area]], columns=columns)

    # Check if the Excel file already exists
    excel_path = '/Users/mukeshnaidu/MukeshGit/output/frame_info.xlsx'
    file_exists = os.path.isfile(excel_path)

    # If the file exists, append the frame information to it
    if file_exists:
        existing_data = pd.read_excel(excel_path)
        frame_data = pd.concat([existing_data, frame_data], ignore_index=True)

    # Save the DataFrame to Excel
    frame_data.to_excel(excel_path, index=False)
    print('Frame information saved to', excel_path)
