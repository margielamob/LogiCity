import numpy as np
from prettytable import PrettyTable
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

class DecisionTreeRunner:
    def __init__(self, data_X_train, data_Y_train, data_X_test, data_Y_test, Yname, logger):
        self.data_X_train = data_X_train
        self.data_Y_train = data_Y_train
        self.Yname = Yname
        self.logger = logger

    def log_distribution(self, message):
        # TODO: Log the distribution of both the training and test dataset
        unique_rows, counts = np.unique(self.data_Y, axis=0, return_counts=True)
        # Create a PrettyTable instance
        table = PrettyTable()
        # Use Yname for the field names
        table.field_names = self.Yname + ["Count"]
        table.title = message

        # Populate the table with data
        for row, count in zip(unique_rows, counts):
            # Convert row to labels using Yname
            label_row = [self.Yname[i] if x == 1 else "UNKNOWN" for i, x in enumerate(row)]
            table.add_row(label_row + [count])

        # Sort the table by the 'Count' column
        table.sortby = "Count"
        table.reversesort = True

        # Log the table using the provided logger
        self.logger.info(f"\n{table}")

    def run(self):
        # Log the distribution of the dataset
        self.log_distribution("Dataset distribution before training")

        # Convert the 0.5 in data_Y to 0 to indicate 'UNKNOWN'/'FALSE'
        data_Y_converted = np.where(self.data_Y_train == 0.5, 0, 1)

        # Initialize the Decision Tree Classifier
        self.clf = DecisionTreeClassifier()

        # Fit the classifier to your data
        self.clf.fit(self.data_X_train, data_Y_converted)
    
    def evaluate(self):
        # TODO: Log the accuracy for each label
        # Make predictions
        predictions = self.clf.predict(self.data_X_test)
        data_Y_converted = np.where(self.data_Y_test == 0.5, 0, 1)
        # Calculate accuracy
        accuracy = accuracy_score(data_Y_converted, predictions)
        self.logger.info(f"Accuracy: {accuracy}")
        self.logger.info(classification_report(data_Y_converted, predictions, target_names=self.Yname))