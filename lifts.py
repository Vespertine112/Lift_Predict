from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import liftModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.sparse
import pickle

RUN_TABLE = True

SAVE_MODEL = True # Also controls the training of a model
LOAD_MODEL = True

SAVE_FIGS = True
SAVE_CHARTS = True

MODEL_NAME = "All" # Saving & Loading the Model!

try:
    from prettytable import PrettyTable
except ImportError:
    RUN_TABLE = False


# ==================================================================================================
# The Lift Model: Brayden Hill - A02287193 - Hillbgh@gmail.com
# NOTE: This will not run out of the box! You need to download the dataset! See README
# If you want to predict a given lift, remove it from the standard_scale_features array!
# Finally, getting a good model requires some (minimal) bulldozing. Target a loss of around 10 in epoch 0!
# ==================================================================================================

# ==================================================================================================
# Data [Loading / Preprocessing / Splitting]
# Division: might be useful, however there are 3780 unique "divisions" due to localization, so we will drop it
#
# region=============================================================================================
data = pd.read_csv("openpowerlifting.csv", low_memory=False)
data = data[data["Event"].str.contains("SBD")]

data["WeightClassKg"] = data["WeightClassKg"].astype(str)
data["Tested"].fillna("No", inplace=True)

drop_features = [
    "Name",
    "Event",
    "Division",
    "WeightClassKg",
    "Date",
    "Place",
    "MeetCountry",
    "MeetState",
    "MeetName",
    "Country",
    "Federation",
    "Squat1Kg",
    "Squat2Kg",
    "Squat3Kg",
    "Squat4Kg",
    "Bench1Kg",
    "Bench2Kg",
    "Bench3Kg",
    "Bench4Kg",
    "Deadlift1Kg",
    "Deadlift2Kg",
    "Deadlift3Kg",
    "Deadlift4Kg",
    "Wilks",
    "McCulloch",
    "Glossbrenner",
    "IPFPoints",
]

encode_features = [
    "Sex",
    "Equipment",
    "AgeClass",
    "Tested",
]

standard_scale_features = [
    "Age",
    "Best3SquatKg",
    "Best3BenchKg",
    "Best3DeadliftKg",
]

max_scale_features = ["BodyweightKg"]

# Drop the specified columns and any rows with NaN values
data = data.drop(columns=drop_features)
data = data.dropna()

X = data.drop(columns=["TotalKg"])
y = pd.DataFrame(data["TotalKg"])

# endregion=========================================================================================
# Model [Definition / Instantiation / Training]
#
# region ===========================================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

preprocessor = make_column_transformer(
    (OneHotEncoder(), encode_features),
    (StandardScaler(), standard_scale_features),
    (MinMaxScaler(), max_scale_features),
    sparse_threshold=0
)

x_train, x_test, y_train, y_test = train_test_split(
    preprocessor.fit_transform(X), y, test_size=0.3, random_state=32
)

# loop over inputs and yield batchSize samples for current batch.
def gen_next_batch(x_train, y_train, batch_size=32):
    for i in range(0, x_train.shape[0], batch_size):
        yield (x_train[i : i + batch_size], y_train[i : i + batch_size])

input_size = np.array(len(preprocessor.get_feature_names_out()))
output_size = 1
LR = 0.005
batch_size = 16

ANN = liftModel.getLiftRegressionModel(inFeatures=input_size, outFeatures=output_size).to(device)
opt = optim.Adam(ANN.parameters(), lr=LR)
lossFunc = nn.L1Loss()

if SAVE_MODEL:
    ANN.train()
    loss_tracker, e = 0, -1, 
    for epoch in range(0, 3):
        for i, (batchX, batchY) in enumerate(gen_next_batch(x_train, y_train, batch_size=batch_size)):
            batchX = torch.from_numpy(np.array(batchX).astype(np.float32)).to(device)
            batchY = torch.from_numpy(np.array(batchY).astype(np.float32)).to(device)

            predictions = ANN(batchX)
            loss = lossFunc(predictions, batchY)


            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = loss.item() * batchY.size(0)
            loss_tracker += loss_val
            print(f"Epoch {epoch:>10} | Loss: {loss:>10.3f} | Loss avg: {(loss_tracker/ (i*epoch+(i+1))):>10.3f}")
            if e != epoch:
                e = epoch

# endregion=========================================================================================
# Model Save & Load
# 
# region ===========================================================================================
if SAVE_MODEL:
    model_filename = f"Models/lift_model_{MODEL_NAME}.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(ANN, model_file)
    print(f"Model saved to {model_filename}")

if LOAD_MODEL:
    loaded_model_filename = f"Models/lift_model_{MODEL_NAME}.pkl"
    with open(loaded_model_filename, 'rb') as loaded_model_file:
        loaded_ANN = pickle.load(loaded_model_file)
    print(f"Model loaded from {loaded_model_filename}")
else:
    loaded_ANN = ANN

# endregion ========================================================================================
# Model Evaluation 
# There are several metrics calculated: R^2, Average RMSE, Average Loss
# 
# region ============================================================================================

testLoss, r2_scores, rmse_scores, num_samples = 0, [], [], 0
predictions_all, actual_values_all = [], []

if RUN_TABLE:
    results_table = PrettyTable()
    original_feature_names = encode_features
    original_feature_names.extend(standard_scale_features)
    original_feature_names.extend(max_scale_features)
    results_table.field_names = [
        "Sample ID",
        *original_feature_names,  # Unpack the feature names
        "Actual TotalKg",
        "Predicted TotalKg",
    ]

loaded_ANN.eval()
with torch.no_grad():
    for i, (batchX, batchY) in enumerate(gen_next_batch(x_test, y_test, 1)):
        if (i % 100) == 0: print(f"Evaluating Batch {i}")

        (batchX, batchY) = (
            torch.from_numpy(np.array(batchX).astype(np.float32)).to(device),
            torch.from_numpy(np.array(batchY).astype(np.float32)).to(device),
        )

        predictions = loaded_ANN(batchX)
        loss = lossFunc(predictions, batchY)

        predictions_all.extend(predictions.cpu().numpy())
        actual_values_all.extend(batchY.cpu().numpy())

        testLoss += loss.item()
        num_samples += len(batchX)

        rmse_scores.append(
            np.sqrt(mean_squared_error(batchY.cpu().numpy(), predictions.cpu().numpy()))
        )

        # Creates a table if you have PrettyTable
        if RUN_TABLE and i < 32:
            for idx in range(len(batchX)):
                bound_len = len(standard_scale_features) + len(max_scale_features)
                original_categorical_features = preprocessor.named_transformers_["onehotencoder"].inverse_transform(batchX[:, :-bound_len].cpu().numpy())
                original_standard_scale_features = preprocessor.named_transformers_["standardscaler"].inverse_transform(batchX[:, -bound_len:-1].cpu().numpy())
                original_max_scale_features = preprocessor.named_transformers_["minmaxscaler"].inverse_transform(batchX[:, -1:].cpu().numpy())
                original_input_features = np.concatenate((original_categorical_features, original_standard_scale_features, original_max_scale_features), axis=1)
                formatted_values = [
                    f"{value:.1f}" if isinstance(value, (float, np.float32, np.double)) else value for value in original_input_features[idx]
                ]

                results_table.add_row(
                    [
                        i + 1,  # Sample ID number
                        *formatted_values,  # Original input features
                        f"{batchY[idx].item():.3f}",  # Actual TotalKg
                        f"{predictions[idx].item():.3f}",  # Predicted TotalKg
                    ]
                )

# Calculate Average Test Loss
averageTestLoss = testLoss / num_samples
print(f"Average Test Loss: {averageTestLoss:.4f}")

# Calculate Average R-squared score
average_r2 = r2_score(actual_values_all, predictions_all)
print(f"Average R-squared Score: {average_r2:.4f}")

# Calculate Average RMSE
average_rmse = np.mean(rmse_scores)
print(f"Average RMSE: {average_rmse:.4f}")

if RUN_TABLE:
    print("Example Inputs and Predictions:")
    if SAVE_CHARTS:
        with open(f"Predictions/{MODEL_NAME}.txt", 'w') as f:
            f.write(results_table.get_string()+"\n")
            f.write(f"Average RMSE: {average_rmse:.4f}\n")
            f.write(f"Average R-squared Score: {average_r2:.4f}\n")
            f.write(f"Average Test Loss: {averageTestLoss:.4f}\n")

    else:
        print(results_table)

# Visualization 1: Scatter plot of Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(actual_values_all, predictions_all, alpha=0.5)
plt.title(f"Actual vs. Predicted Values [{MODEL_NAME}]")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
if SAVE_FIGS:
    plt.savefig(f"Plots/{MODEL_NAME}_AP.png")
else:
    plt.show()

# Visualization 2: Residuals Plot
residuals = np.array(actual_values_all) - np.array(predictions_all)
plt.figure(figsize=(10, 6))
plt.scatter(predictions_all, residuals, alpha=0.5)
plt.title(f"Residuals Plot [{MODEL_NAME}]")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y=0, color="r", linestyle="--")
if (SAVE_FIGS):
    plt.savefig(f"Plots/{MODEL_NAME}_RES.png")
else:
    plt.show()

# endregion ========================================================================================

print("Complete!")