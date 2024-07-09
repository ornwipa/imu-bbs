import pandas as pd

def process_and_merge(features_path, bbs_path, output_path):
    features_data = pd.read_csv(features_path)
    
    bbs_data = pd.read_csv(bbs_path, delimiter=';')
    bbs_data.columns = ['Participant', 'Task', 'BBS', 'Extra']
    bbs_data = bbs_data.drop(columns=['Extra'])
    bbs_data['BBS'] = pd.to_numeric(bbs_data['BBS'])
    
    features_data['Task'] = features_data['Task'].str.replace('.csv', '')
    features_data['Participant'] = features_data['Participant'].str.lower().str.replace('partipant', 'participant')
    bbs_data['Participant'] = bbs_data['Participant'].str.lower()
    
    merged_data = pd.merge(features_data, bbs_data, on=["Participant", "Task"], how="left")
    
    merged_data.to_csv(output_path, index=False)

features_path = 'C:/Users/youss/Downloads/result/result2/result2.csv'
bbs_path = 'C:/Users/youss/Downloads/result/bbs.csv'
output_path = 'C:/Users/youss/Downloads/result/result2/result3.csv'

process_and_merge(features_path, bbs_path, output_path)

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['BBS'] = data['BBS'].apply(lambda x: 0 if x == 4 else 1)
    return data

data = load_and_preprocess_data(output_path)

additional_data = pd.read_csv('C:/Users/youss/Downloads/result/additional_data.csv', delimiter=';')

additional_data['Age'] = additional_data['Age'].str.replace('ans', '').astype(int)
additional_data['poids'] = additional_data['poids'].str.replace('kg', '').astype(float)
additional_data['Gender'] = additional_data['Gender'].apply(lambda x: 0 if x == 'F' else 1)

merged_data = pd.merge(data, additional_data, on='Participant', how='left')

merged_data.to_csv('C:/Users/youss/Downloads/result/result2/result4.csv', index=False)

print(merged_data.describe())
