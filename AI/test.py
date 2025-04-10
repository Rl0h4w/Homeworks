import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import multiprocessing
import time
import math
# Import sklearn correctly
import sklearn
import sklearn.model_selection # Explicitly import submodule
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
from tqdm import tqdm
import pandas as pd

# --- Markov Random Field Generation (Optimized for PyTorch CPU/GPU) ---
def generate_mrf_batched(
    batch_size,
    size=50,
    interaction_strength=1.0,
    external_field=0.0,
    num_iterations=1000,
    device="cpu", # Defaulting to CPU here for clarity, but controlled externally
):
    """Generates a batch of MRF states using PyTorch convolutions."""
    # Initial random state (-1 or 1)
    state = (
        torch.randint(
            0, 2, (batch_size, 1, size, size), dtype=torch.float32, device=device
        )
        * 2
        - 1
    )

    # Define the convolution kernel for neighbor sum
    kernel = torch.tensor(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device
    ).reshape(1, 1, 3, 3)

    # Create checkerboard masks for parallel updates
    mask_white = torch.zeros((size, size), dtype=torch.bool, device=device)
    mask_white[::2, ::2] = 1
    mask_white[1::2, 1::2] = 1
    mask_black = ~mask_white

    mask_white = mask_white.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, size, size)
    mask_black = mask_black.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, size, size)

    # Gibbs sampling iterations
    for _ in range(num_iterations):
        # --- Update "white" cells ---
        neighbor_sum = F.conv2d(state, kernel, padding=1)
        local_field = external_field + interaction_strength * neighbor_sum
        prob_stay = torch.sigmoid(2 * state * local_field) # Probability P(spin=s) = sigmoid(2*s*local_field)
        random_numbers = torch.rand_like(state)
        # Flip state if random number > probability of staying AND it's a white cell
        flip_condition_white = (random_numbers > prob_stay) & mask_white
        state[flip_condition_white] *= -1

        # --- Update "black" cells ---
        # Recompute sums/fields as white cells have changed
        neighbor_sum = F.conv2d(state, kernel, padding=1)
        local_field = external_field + interaction_strength * neighbor_sum
        prob_stay = torch.sigmoid(2 * state * local_field)
        random_numbers = torch.rand_like(state)
        # Flip state if random number > probability of staying AND it's a black cell
        flip_condition_black = (random_numbers > prob_stay) & mask_black
        state[flip_condition_black] *= -1

    # Convert state from {-1, 1} to {0, 1} and remove channel dimension
    binary_matrix_batch = ((state + 1) / 2).squeeze(1)
    return binary_matrix_batch


# --- Percolation Check (CPU-bound, suitable for multiprocessing) ---
def check_percolation(binary_map):
    """
    Checks for a percolating cluster (left to right) using BFS.
    Args:
        binary_map (np.array): 2D NumPy array with 0s and 1s.
    Returns:
        int: 1 if percolation occurs, 0 otherwise.
    """
    rows, cols = binary_map.shape
    if np.sum(binary_map[:, 0]) == 0: # Optimization: No 1s in first col -> no percolation
        return 0

    visited = np.zeros_like(binary_map, dtype=bool)
    queue = deque()

    # Initialize queue with all '1' cells in the first column
    for r in range(rows):
        if binary_map[r, 0] == 1:
            queue.append((r, 0))
            visited[r, 0] = True

    # Breadth-First Search
    while queue:
        r, c = queue.popleft()

        # Check if we reached the right edge
        if c == cols - 1:
            return 1

        # Explore neighbors (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check boundaries, if it's a '1', and if not visited
            if 0 <= nr < rows and 0 <= nc < cols and \
               binary_map[nr, nc] == 1 and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # If the queue empties and we haven't reached the right edge
    return 0

# --- Wrapper for Multiprocessing Pool ---
def check_percolation_wrapper(args):
    """Helper function to unpack arguments for multiprocessing."""
    index, binary_map = args
    return index, check_percolation(binary_map)

# --- Dataset Creation (Uses multiprocessing for percolation check) ---
def create_percolation_dataset_optimized(
    num_samples,
    size=50,
    interaction_strength=1.0,
    external_field=0.0,
    mrf_iterations=100,
    batch_size=64, # Batch size for MRF generation (can be large on CPU too)
    num_workers=None, # Number of CPU processes for percolation check
    device="cpu", # Device for MRF generation (forced CPU later)
):
    """
    Generates dataset by creating MRF maps (on specified device)
    and checking percolation in parallel on CPU cores.
    """
    if num_workers is None:
        try:
            num_workers = os.cpu_count()
            print(f"Using {num_workers} workers for percolation check.")
        except NotImplementedError:
            num_workers = 1 # Fallback if cpu_count() fails
            print("Could not determine CPU count, using 1 worker for percolation check.")

    # Pre-allocate NumPy arrays on CPU RAM
    X_data = np.zeros((num_samples, size, size), dtype=np.float32)
    y_labels = np.zeros(num_samples, dtype=np.int8) # Use int8 for {0, 1} labels

    num_batches = math.ceil(num_samples / batch_size)
    generated_count = 0

    # Create the multiprocessing pool *once*
    # Use 'spawn' context if issues arise with 'fork' (default on Linux/macOS)
    # try:
    #    multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #    pass # Already set or not applicable
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=num_samples, desc="Generating dataset") as pbar:
            for i in range(num_batches):
                current_batch_size = min(batch_size, num_samples - generated_count)
                if current_batch_size <= 0:
                    break # Should not happen with ceil, but safety check

                # 1. Generate MRF maps batch (on CPU as per device setting)
                mrf_maps_batch_tensor = generate_mrf_batched(
                    batch_size=current_batch_size,
                    size=size,
                    interaction_strength=interaction_strength,
                    external_field=external_field,
                    num_iterations=mrf_iterations,
                    device=device, # Pass the specified device (e.g., "cpu")
                )

                # 2. Convert to NumPy (already on CPU if device is "cpu")
                # No explicit .cpu() needed if generated on CPU
                mrf_maps_batch_cpu = mrf_maps_batch_tensor.numpy()

                # 3. Prepare tasks for parallel percolation check
                tasks = [
                    (generated_count + j, mrf_maps_batch_cpu[j])
                    for j in range(current_batch_size)
                ]

                # 4. Execute percolation checks in parallel and store results
                # imap_unordered gets results as they finish, good for varying task times
                for idx, percolates in pool.imap_unordered(check_percolation_wrapper, tasks):
                    original_batch_index = idx - generated_count
                    # Ensure index is within bounds (should be)
                    if 0 <= original_batch_index < current_batch_size:
                         X_data[idx] = mrf_maps_batch_cpu[original_batch_index]
                         y_labels[idx] = percolates
                    else:
                         print(f"Warning: Index mismatch detected. idx={idx}, generated_count={generated_count}") # Debugging
                    pbar.update(1) # Update progress bar for each completed sample

                generated_count += current_batch_size

    return X_data, y_labels


# --- Convolutional Neural Network Model ---
class PercolationCNN(nn.Module):
    def __init__(self, size=50): # Pass size for dynamic calculation
        super(PercolationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after pooling layers
        # Each pool layer halves the size (approximately, depends on odd/even size)
        pooled_size = size // (2**4) # 4 pooling layers
        if pooled_size == 0:
            # Handle cases where the image becomes too small
            print(f"Warning: Image size {size} might be too small for 4 pooling layers. Adjusting.")
            pooled_size = 1 # Ensure at least 1x1 feature map

        # Calculate the flattened size for the fully connected layer
        fc_input_features = 256 * pooled_size * pooled_size

        self.fc1 = nn.Linear(fc_input_features, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid() # Use Sigmoid for binary classification output

    def forward(self, x):
        # x shape: (batch_size, height, width)
        x = x.unsqueeze(1)  # Add channel dim -> (batch_size, 1, height, width)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten the output for the fully connected layer
        # Use view or flatten
        # x = x.view(x.size(0), -1) # Dynamically calculates the flattened size
        x = torch.flatten(x, 1) # Flattens dims starting from dim 1

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x) # Output raw logits
        # Sigmoid is often applied outside or included in the loss (BCEWithLogitsLoss)
        # Here we apply sigmoid as the original code did, matching BCELoss
        x = self.sigmoid(x)

        return x

# --- Training Loop ---
def fit(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """Trains the model."""
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        train_loss = 0.0
        train_preds_list = []
        train_targets_list = []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1) # Ensure targets are float and have channel dim for BCE

            optimizer.zero_grad() # Clear previous gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            train_loss += loss.item() * inputs.size(0)
            train_preds_list.extend(outputs.detach().cpu().numpy())
            train_targets_list.extend(targets.detach().cpu().numpy())
            train_pbar.set_postfix(loss=loss.item())

        train_loss = train_loss / len(train_loader.dataset)
        train_preds = np.array(train_preds_list).flatten()
        train_targets = np.array(train_targets_list).flatten()
        # Use try-except for metrics if only one class is present in a batch/epoch
        try:
            train_roc_auc = roc_auc_score(train_targets, train_preds)
        except ValueError:
            train_roc_auc = 0.5 # Or np.nan
        train_accuracy = accuracy_score(train_targets.round().astype(int), train_preds.round().astype(int))

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad(): # Disable gradient calculations for validation
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_preds_list.extend(outputs.detach().cpu().numpy())
                val_targets_list.extend(targets.detach().cpu().numpy())
                val_pbar.set_postfix(loss=loss.item())

        val_loss = val_loss / len(val_loader.dataset)
        val_preds = np.array(val_preds_list).flatten()
        val_targets = np.array(val_targets_list).flatten()
        try:
            val_roc_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_roc_auc = 0.5 # Or np.nan
        val_accuracy = accuracy_score(val_targets.round().astype(int), val_preds.round().astype(int))

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Train ROC AUC: {train_roc_auc:.4f}")
        print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_accuracy:.4f} | Valid ROC AUC: {val_roc_auc:.4f}")

# --- Prediction Function ---
def predict(model, data_loader, device):
    """Generates predictions for the given data loader."""
    model.eval() # Set model to evaluation mode
    predictions_list = []
    with torch.no_grad():
        for inputs_batch in tqdm(data_loader, desc="Predicting"):
            # DataLoader might return [inputs] or [inputs, targets]
            # We only need the inputs for prediction.
            if isinstance(inputs_batch, (list, tuple)):
                inputs = inputs_batch[0].to(device)
            else: # Assuming it's just the input tensor
                 inputs = inputs_batch.to(device)

            outputs = model(inputs)
            predictions_list.extend(outputs.cpu().numpy()) # Move predictions to CPU

    return np.array(predictions_list).flatten()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    num_data_points = 10000  # Reduced for quicker testing on CPU
    map_size = 50
    interaction = 1.0  # Ising model interaction strength
    external_h = 0.0   # Ising model external field
    mrf_iters = 500    # Fewer iterations might suffice, adjust as needed
    gen_batch_size = 512 # Batch size for MRF generation (adjust based on RAM)
    train_batch_size = 64 # Batch size for training/validation
    
    # Determine number of workers for multiprocessing
    try:
        # Use half the cores for safety, adjust as needed
        default_workers = max(1, os.cpu_count() // 2) 
    except NotImplementedError:
        default_workers = 1
    cpu_workers_perc = default_workers # Workers for percolation check in generation
    cpu_workers_loader = default_workers # Workers for DataLoader during training/prediction

    # Force CPU usage
    device = torch.device("cpu")
    print(f"Forcing execution on device: {device}")

    # --- Dataset Generation ---
    print("Starting Optimized Dataset Generation...")
    start_time = time.time()

    # Ensure create_percolation_dataset uses the CPU device for MRF generation
    X, y = create_percolation_dataset_optimized(
        num_samples=num_data_points,
        size=map_size,
        interaction_strength=interaction,
        external_field=external_h,
        mrf_iterations=mrf_iters,
        batch_size=gen_batch_size,
        num_workers=cpu_workers_perc, # For percolation check pool
        device=str(device), # Pass device string "cpu"
    )

    end_time = time.time()
    print(f"\nDataset generation finished in {end_time - start_time:.2f} seconds.")
    print(f"  Generated {len(X)} samples.")
    print(f"  Map size (X shape): {X.shape}")
    print(f"  Labels (y shape): {y.shape}")
    print(f"  Data types: X={X.dtype}, y={y.dtype}")
    print(f"  Fraction percolating: {np.mean(y):.3f}")

    # --- Data Splitting and Preparation ---
    print("\nSplitting data and creating DataLoaders...")
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # Stratify is important for imbalanced datasets
    )

    # Convert numpy arrays to PyTorch Tensors (already on CPU)
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).float()
    X_val_tensor = torch.tensor(X_val).float()
    y_val_tensor = torch.tensor(y_val).float()

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create DataLoaders with multiprocessing workers for loading
    # pin_memory=True is only useful for GPU, set to False for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers_loader,
        pin_memory=False # Set to False for CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=cpu_workers_loader,
        pin_memory=False
    )
    print(f"Using {cpu_workers_loader} workers for DataLoaders.")


    # --- Model Training ---
    print("\nInitializing and training the model...")
    # Pass map_size to the model constructor
    model = PercolationCNN(size=map_size).to(device) # Move model to CPU
    
    # Use Binary Cross Entropy Loss for binary classification with sigmoid output
    criterion = nn.BCELoss() 
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15 # Reduced epochs for faster CPU training example

    fit(model, train_loader, val_loader, optimizer, criterion, epochs, device)


    # --- Prediction Examples & Evaluation ---
    print("\nMaking predictions (example)...")
    # Use predict function on validation set
    val_loader_predict = DataLoader(
        val_dataset, 
        batch_size=train_batch_size, 
        num_workers=cpu_workers_loader,
        pin_memory=False
    )
    val_predictions = predict(model, val_loader_predict, device)
    val_roc_auc_predict = roc_auc_score(y_val, val_predictions)
    val_accuracy_predict = accuracy_score(
        np.array(y_val).round().astype(int), np.round(val_predictions).astype(int)
    )
    print(f"Validation Accuracy (post-train predict): {val_accuracy_predict:.4f}")
    print(f"Validation ROC AUC (post-train predict): {val_roc_auc_predict:.4f}")
    print(f"Example Validation Predictions (first 10 raw): {val_predictions[:10]}")


    # --- Evaluate on External Test File (Example from Kaggle path) ---
    # Adjust these paths if necessary
    test_file_path = "Xtest.pickle" # Assuming file is in the current directory or provide full path
    submission_file_path = "submission.csv"
    model_save_path = "percolation_cnn_model_cpu.pth"

    print(f"\nAttempting to load and predict on test data from: {test_file_path}")

    if os.path.exists(test_file_path):
        try:
            with open(test_file_path, "rb") as f:
                X_test_np = pickle.load(f)
            print(f"  Loaded test data. Shape: {X_test_np.shape}, Type: {X_test_np.dtype}")

            # Convert test data to Tensor
            X_test_tensor = torch.tensor(X_test_np).float()

            # Create DataLoader for test data (no labels)
            test_dataset = TensorDataset(X_test_tensor)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=train_batch_size, 
                num_workers=cpu_workers_loader,
                pin_memory=False
            )

            # Make predictions
            print("  Making predictions on test data...")
            test_predictions_raw = predict(model, test_loader, device)
            # Convert probabilities to binary predictions (0 or 1)
            test_predictions_binary = np.round(test_predictions_raw).astype(int)

            print(f"  Test Predictions (first 10 binary): {test_predictions_binary[:10]}")

            # --- Generate Submission File ---
            # Assuming the order in Xtest corresponds to sequential IDs 0, 1, 2...
            num_test_samples = len(X_test_np)
            ids = np.arange(num_test_samples)

            submission_df = pd.DataFrame({"id": ids, "prediction": test_predictions_binary})
            submission_df.to_csv(submission_file_path, index=False)
            print(f"\nSubmission file saved to: {submission_file_path}")

            # --- Optional: Evaluate if ytest.pickle exists ---
            y_test_file_path = test_file_path.replace("Xtest", "ytest")
            if os.path.exists(y_test_file_path):
                try:
                    with open(y_test_file_path, "rb") as f:
                        y_test_np = pickle.load(f)
                    print(f"  Loaded test labels. Shape: {y_test_np.shape}")

                    # Evaluate the binary predictions
                    test_accuracy = accuracy_score(y_test_np.astype(int), test_predictions_binary)
                     # Use raw predictions (probabilities) for ROC AUC
                    test_roc_auc = roc_auc_score(y_test_np, test_predictions_raw) 
                    print(f"  Test Accuracy: {test_accuracy:.4f}")
                    print(f"  Test ROC AUC: {test_roc_auc:.4f}")
                except Exception as e:
                    print(f"  Error loading or evaluating test labels: {e}")
            else:
                print("  Test labels (ytest.pickle) not found. Cannot calculate test accuracy/ROC AUC.")

        except FileNotFoundError:
            print(f"Error: The test file {test_file_path} was not found in the expected location.")
        except Exception as e:
            print(f"An error occurred during test data processing: {e}")
    else:
        print(f"Test file not found at '{test_file_path}'. Skipping test prediction.")


    # --- Save the Trained Model ---
    print(f"\nSaving trained model to: {model_save_path}")
    try:
        # Save only the model's state dictionary (recommended)
        torch.save(model.state_dict(), model_save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\nScript finished.")