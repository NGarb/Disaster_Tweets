import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from typing import Any, Dict, List
from sklearn.model_selection import train_test_split
import pandas as pd
from google.colab import drive
from pathlib import Path
import os
import mlflow



def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
      exp_id = mlflow.create_experiment(name)
      return exp_id
    return exp.experiment_id


drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/Colab Notebooks/Kaggle/Disaster Tweets/')

data_path = Path("./data")



class TweetDataset(Dataset):
    def __init__(self, tweets: List[str], labels: List[int], tokenizer: BertTokenizer) -> None:
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.tweets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tweet = str(self.tweets[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=64,  # Adjust as per your requirements
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def train(model, train_loader, val_loader, optimizer, epochs):
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_loss /= len(train_loader)

        val_accuracy, val_loss = validate(model, val_loader)

        print(f"Epoch: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    return train_loss, train_accuracy, val_loss, val_accuracy


def validate(model, val_loader):
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    val_accuracy = correct / total
    val_loss /= len(val_loader)
    val_loss /= len(val_loader)
    return val_accuracy, val_loss


if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/NGarb/Disaster_Tweets.mlflow")
    mlflow.set_experiment("Disaster_Tweets")

    # Load the data
    train_df = pd.read_csv(data_path / "train.csv")

    # Specify the features (X) and the target variable (y)
    X = train_df.drop('target', axis=1)
    y = train_df['target']

    # Split the data into train and validation sets
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load and tokenize the pre-trained BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Define the hyperparameters
    batch_size = 32
    learning_rate = 2e-5
    epochs = 3

    # Create training and validation datasets
    train_dataset = TweetDataset(train_X['text'].to_list(), train_y.to_list(), tokenizer)
    val_dataset = TweetDataset(val_X['text'].to_list(), val_y.to_list(), tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    mlflow.tensorflow.autolog()
    experiment_id = get_experiment_id("Disaster_Tweets")
    with mlflow.start_run(experiment_id=experiment_id):
        train_loss, train_accuracy, val_loss, val_accuracy = train(model, train_loader, val_loader, optimizer, epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_param("optimizer_name", optimizer.__class__.__name__)
        mlflow.log_param("loss_fn_name", loss_fn.__class__.__name__)
        mlflow.log_param("model_name", model.__class__.__name__)
        mlflow.log_param("tokenizer_name", tokenizer.__class__.__name__)
        mlflow.log_param("tokenizer_max_length", tokenizer.max_len)
        mlflow.pytorch.log_model(model, "model")

