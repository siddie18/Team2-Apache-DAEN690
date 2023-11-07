import os

import numpy as np
import pandas as pd
import torch


def load_data_from_dir(csv_directory):
    dataframes = []
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_directory, filename)
            print('reading file', file_path)
            df = pd.read_csv(file_path, low_memory=False)
            # drop the first row, it is data type of each column
            df = df.drop(0)
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    selected_df = combined_df[['Vert. Speed', 'Groundspeed', 'Altitude(AGL)', 'Date', 'System UTC Time']]
    selected_df = selected_df.dropna()
    return selected_df


def format_data(df):
    # all formatting
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['System UTC Time'], format='%Y-%m-%d %H:%M:%S.%f')
    columns_to_drop = ['Date', 'System UTC Time']
    df = df.drop(columns=columns_to_drop)
    columns_to_convert = ['Vert. Speed', 'Groundspeed', 'Altitude(AGL)']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Sort the DataFrame by the 'DateTime' column
    df.sort_values(by='DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_sliding_window_sequences(df, sequence_length, step_size):
    # Create sequences using a sliding window
    X = []
    y = []

    x_df = df[['Vert. Speed', 'Groundspeed', 'Altitude(AGL)']]
    for i in range(0, len(df) - sequence_length, step_size):
        X.append(x_df[i:i + sequence_length].values)
        y.append(df['Phase'][i + sequence_length])

    return X, y


def torch_input_output(X, y, device):
    return torch.tensor(np.array(X), dtype=torch.float32).to(device), torch.tensor(np.array(y), dtype=torch.float32).to(
        device)


def model_predict(data_loader, model, encoder):
    with torch.no_grad():
        predicted_labels = []
        actual_labels = []

        total_samples = 0
        correct_predictions = 0

        for inputs, labels in data_loader:
            labels = labels.long()
            test_outputs = model(inputs)
            _, predicted = torch.max(test_outputs, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            predicted_labels.extend(encoder.inverse_transform(predicted.cpu().numpy()))
            actual_labels.extend(encoder.inverse_transform(labels.cpu().numpy()))

        test_accuracy = correct_predictions / total_samples
        print(f'Test accuracy: {test_accuracy:.2%}')
        return predicted_labels, actual_labels
