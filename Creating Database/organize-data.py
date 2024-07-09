import os
import pandas as pd
import csv

def extract_columns(source_folder, base_target_folder, columns):
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            components = file_name.split('-')

            participant_name = components[2]
            body_part = components[1].split('_')[-1]
            task_name = components[3].split('.')[0]  

            file_path = os.path.join(source_folder, file_name)
            df = pd.read_csv(file_path, delimiter=',', skiprows=[0, 7], on_bad_lines='skip')

            df_selected = df.iloc[:, columns]

            participant_folder = os.path.join(base_target_folder, participant_name)
            body_part_folder = os.path.join(participant_folder, body_part)

            if not os.path.exists(body_part_folder):
                os.makedirs(body_part_folder)

            output_file_path = os.path.join(body_part_folder, f"{task_name}.csv")
            df_selected.to_csv(output_file_path, index=False)

source_folder = r'C:\Program Files\TK Motion Manager\workspace\YOUSSEF' 
base_target_folder = r'C:\\Users\\youss\\Downloads\\test1'
columns = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

extract_columns(source_folder, base_target_folder, columns)

# Processing the extracted CSV files
root_directory = base_target_folder
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(dirpath, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            processed_lines = []
            for line in lines:
                line = line.strip()
                items = line.split(',')
                processed_items = [items[0]]
                for i in range(1, len(items) - 1, 2):
                    try:
                        item = items[i] + '.' + items[i + 1]
                        processed_items.append(float(item))
                    except (IndexError, ValueError):
                        continue
                processed_lines.append(processed_items)

            df = pd.DataFrame(processed_lines)

            headers = ['time', 'Acceleration X (m/s^2)', 'Acceleration Y (m/s^2)', 'Acceleration Z (m/s^2)', 
                       'Angular Velocity X (rad/s)', 'Angular Velocity Y (rad/s)', 'Angular Velocity Z (rad/s)']

            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(df.values)
