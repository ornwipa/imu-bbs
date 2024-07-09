import pandas as pd
import numpy as np
import os
import sqlite3
from scipy.spatial import ConvexHull
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
def calibrate_movement(data, ipose_means):
    for axis, col_name in zip(['X', 'Y', 'Z'], ['Acceleration X (m/s^2)', 'Acceleration Y (m/s^2)', 'Acceleration Z (m/s^2)']):
        if col_name in data.columns:
            data[f'Calibrated Acceleration {axis}'] = data[col_name].astype(float) - ipose_means[axis]
    return data

def interpolate_nan_values(data):
    nan_count_before = data.isna().sum()

    data.interpolate(method='linear', inplace=True)
    data.fillna(method='bfill', inplace=True)  
    data.fillna(method='ffill', inplace=True) 
    nan_count_after = data.isna().sum()

    filled_nans = nan_count_before - nan_count_after

def low_pass_filter(data, cutoff=3.667, fs=128, order=6):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def integrate(data, dt=1/128):

    return np.cumsum(data * dt)

def add_ellipse_and_box(subplot, x_data, y_data, label_x, label_y):

    x_std, y_std = np.std(x_data), np.std(y_data)
    ellipse = Ellipse(xy=(np.mean(x_data), np.mean(y_data)), width=2*x_std, height=2*y_std, edgecolor='r', fc='None', lw=2, label="Ellipse Fitting Area")
    subplot.add_patch(ellipse)

    min_x, max_x, min_y, max_y = np.min(x_data), np.max(x_data), np.min(y_data), np.max(y_data)
    rectangle = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, edgecolor='b', fc='None', lw=2, label="Bounding Box Area")
    subplot.add_patch(rectangle)

    subplot.set_xlabel(label_x)
    subplot.set_ylabel(label_y)
    subplot.legend()

def add_convex_hull(subplot, points, label_x, label_y):

    cleaned_points = points[~np.isnan(points).any(axis=1)]
    if len(cleaned_points) >= 3:
        hull = ConvexHull(cleaned_points)
        for simplex in hull.simplices:
            subplot.plot(cleaned_points[simplex, 0], cleaned_points[simplex, 1], 'k-', linewidth=1, label="Convex Hull Edge" if simplex[0] == 0 else "")
        print(f"Convex Hull Area ({label_x}, {label_y}): {hull.area:.2f} square units")
    else:
        print("Insufficient valid data points for Convex Hull computation.")
    subplot.set_xlabel(label_x)
    subplot.set_ylabel(label_y)
    subplot.legend()

def calculate_position(accel_data, time_data):

    velocity = cumtrapz(accel_data, time_data, axis=0, initial=0)
    return cumtrapz(velocity, time_data, axis=0, initial=0)



features_df = pd.DataFrame(columns=[
    "Participant", "Sensor", "Task", "Area", "Jerk X", "Jerk Y", "Jerk Z", 
    "RMS Acceleration", "RMS Angular Velocity", "Ellipse Area", "Box Area", "Total Pathway Length", "volume",
    "Mean Acceleration X" ,"Mean Acceleration Y", "Mean Acceleration Z"
])

base_path = r'C:\Temp\test1'  



for participant_folder in os.listdir(base_path):
    participant_path = os.path.join(base_path, participant_folder)
    if os.path.isdir(participant_path):
        for sensor_folder in os.listdir(participant_path):
            sensor_path = os.path.join(participant_path, sensor_folder)
            ipose_path = os.path.join(sensor_path, "ipose.csv")
            if os.path.exists(ipose_path):
                ipose_data = pd.read_csv(ipose_path)
                ipose_means = {
                    'X': ipose_data['Acceleration X (m/s^2)'].median(),
                    'Y': ipose_data['Acceleration Y (m/s^2)'].median(),
                    'Z': ipose_data['Acceleration Z (m/s^2)'].median()
                }

                for task_file in os.listdir(sensor_path):
                    if task_file.startswith("task"):
                        task_path = os.path.join(sensor_path, task_file)
                        data = pd.read_csv(task_path)
                        filled_nans = interpolate_nan_values(data)
                        print("Number of NaNs filled during interpolation:", filled_nans)
                        interpolate_nan_values(data)
                        data = calibrate_movement(data, ipose_means)
                        filtered_accel_x = low_pass_filter(data['Calibrated Acceleration Z'].values)
                        filtered_accel_y = low_pass_filter(data['Calibrated Acceleration Y'].values)
                        filtered_accel_z = low_pass_filter(data['Calibrated Acceleration X'].values)
                        filtered_accel = np.vstack((filtered_accel_x, filtered_accel_y, filtered_accel_z)).T

                        velocity_x = integrate(filtered_accel_x)
                        velocity_y = integrate(filtered_accel_y)
                        velocity_z = integrate(filtered_accel_z)

                        displacement_x = integrate(velocity_x)
                        displacement_y = integrate(velocity_y)
                        displacement_z = integrate(velocity_z)

                        data['Jerk X'] = np.gradient(data['Calibrated Acceleration Z'], data['time'])
                        data['Jerk Y'] = np.gradient(data['Calibrated Acceleration Y'], data['time'])
                        data['Jerk Z'] = np.gradient(data['Calibrated Acceleration X'], data['time'])
                        data['Jerk Magnitude'] = np.sqrt(data['Jerk X']**2 + data['Jerk Y']**2 + data['Jerk Z']**2)
                        data['Acceleration Magnitude'] = np.sqrt(np.sum(data[[f'Calibrated Acceleration {axis}' for axis in ['X', 'Y', 'Z']]]**2, axis=1))
                        data['Angular Velocity Magnitude'] = np.sqrt(np.sum(data[[f'Angular Velocity {axis} (rad/s)' for axis in ['X', 'Y', 'Z']]].fillna(0)**2, axis=1))

                        pathway_length = np.sum(np.abs(displacement_x)) + np.sum(np.abs(displacement_y)) + np.sum(np.abs(displacement_z))
                        print(f"Total Pathway Length: {pathway_length:.2f} meters")

                        x_std = data['Calibrated Acceleration Z'].std() 
                        y_std = data['Calibrated Acceleration Y'].std()
                        ellipse_area = np.pi * x_std * y_std

                        min_x, max_x = data['Calibrated Acceleration Z'].min(), data['Calibrated Acceleration Z'].max()
                        min_y, max_y = data['Calibrated Acceleration Y'].min(), data['Calibrated Acceleration Y'].max()
                        box_area = (max_x - min_x) * (max_y - min_y)

                        area_under_curve = np.trapz(data['Acceleration Magnitude'], data['time'])
                        rms_acceleration = np.sqrt(np.mean(data['Acceleration Magnitude']**2))
                        rms_angular_velocity = np.sqrt(np.mean(data['Angular Velocity Magnitude']**2))
                        




                        points = data[['Acceleration X (m/s^2)', 'Acceleration Y (m/s^2)', 'Acceleration Z (m/s^2)']].values
                        cleaned_points = points[~np.isnan(points).any(axis=1)]
                        if len(cleaned_points) >= 3:  
                            
                            hull = ConvexHull(cleaned_points)
                            volume = hull.volume
                        mean_accel_x = data['Acceleration Z (m/s^2)'].mean()
                        mean_accel_y = data['Acceleration Y (m/s^2)'].mean()
                        mean_accel_z = data['Acceleration X (m/s^2)'].mean()
                            


                        new_row = pd.DataFrame([{
                            "Participant": participant_folder,
                            "Sensor": sensor_folder,
                            "Task": os.path.basename(task_file),
                            "Area": area_under_curve,
                            "Jerk X": data['Jerk X'].mean(),
                            "Jerk Y": data['Jerk Y'].mean(),
                            "Jerk Z": data['Jerk Z'].mean(),
                            "RMS Acceleration": rms_acceleration,
                            "RMS Angular Velocity": rms_angular_velocity,
                            "Ellipse Area": ellipse_area,
                            "Box Area": box_area,
                            "Total Pathway Length": pathway_length,
                            "volume": volume,
                            "Mean Acceleration X": mean_accel_x,
                            "Mean Acceleration Y": mean_accel_y,
                            "Mean Acceleration Z": mean_accel_z
                            
                        
                        }])
                        features_df = pd.concat([features_df, new_row], ignore_index=True)

csv_save_path = 'C:/Users/youss/Downloads/result/result2/result2.csv'
features_df.to_csv(csv_save_path, index=False)



print("Data processing complete and saved to database.")



# the correlation matrix
numeric_features_df = features_df.select_dtypes(include=[np.number])
corr_matrix = numeric_features_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
