import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tasks.logic.pkl_parser import parse_pkl
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

def runner(args, logger, writer):
    # Load your dataset
    _, _, data_X, data_Y, _, _ = parse_pkl(args.data_path)

    # Convert the 0.5 in data_Y to 0 to indicate 'UNKNOWN'/'FALSE'
    data_Y_converted = np.where(data_Y == 0.5, 0, 1)

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier()

    # Fit the classifier to your data
    clf.fit(data_X, data_Y_converted)

    # To make predictions
    predictions = clf.predict(data_X)

    # Since the outputs are binary (0/1), you can use the MultiLabelBinarizer to reverse the transformation
    mlb = MultiLabelBinarizer()
    mlb.fit(data_Y_converted)
    predicted_labels = mlb.inverse_transform(predictions)

    # Calculate accuracy
    accuracy = accuracy_score(data_Y_converted, predictions)
    logger.info(f"Accuracy: {accuracy}")

    # Log accuracy to TensorBoard
    writer.add_scalar('Accuracy', accuracy)

    # If you want to log a classification report
    report = classification_report(data_Y_converted, predictions, target_names=args.Yname, output_dict=True)
    logger.info(f"Classification Report: \n{report}")

    # Log the classification report to TensorBoard
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Avoid summary rows such as 'accuracy'
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f"{label}/{metric_name}", metric_value)

    # TensorBoard logging: add decision tree visualization (if needed)
    # writer.add_graph(clf, data_X)  # This is hypothetical since scikit-learn models are not supported by TensorBoard's add_graph

    # Make sure to close the writer when you're done
    writer.close()