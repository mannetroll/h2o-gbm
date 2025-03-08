import sys
import logging
import h2o
from h2o.estimators import H2OGradientBoostingEstimator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize H2O
    h2o.init()

    # Parse command-line arguments
    # Default values:
    train = "../csv/small.csv"
    ntrees = 30
    max_depth = 10
    when = "now"

    if len(sys.argv) == 2:
        train = sys.argv[1]
    elif len(sys.argv) == 5:
        train = sys.argv[1]
        ntrees = int(sys.argv[2])
        max_depth = int(sys.argv[3])
        when = sys.argv[4]

    train_model(train, ntrees, max_depth, when)
    logger.info("Done!")
    h2o.shutdown(prompt=False)

def train_model(train, ntrees, max_depth, when):
    try:
        logger.info("*** when: %s", when)
        logger.info("*** train: %s", train)
        logger.info("*** ntrees: %d", ntrees)
        logger.info("*** max_depth: %d", max_depth)

        learn_rate = 0.09
        min_rows = 9
        min_split_improvement = 1e-8

        logger.info("*** learn_rate: %f", learn_rate)
        logger.info("*** min_rows: %d", min_rows)
        logger.info("*** min_split_improvement: %e", min_split_improvement)

        # Columns to ignore (if they exist)
        ignored_columns = ["Group", "EventTime", "EventIndex", "DTA", "Target", "Fraction"]
        logger.info("*** length: %d", len(ignored_columns))
        logger.info("*** ignoredColumns: %s", ignored_columns)

        # Load the training data
        frame_train = h2o.import_file(train)
        logger.info("*** frame_train: %s", frame_train)

        # Specify the response column
        response_column = "TTATA"

        # Determine the predictor columns by excluding ignored and the response column.
        all_columns = frame_train.columns
        predictors = [col for col in all_columns if col not in ignored_columns and col != response_column]

        # Create and train the GBM model
        gbm_model = H2OGradientBoostingEstimator(
            ntrees=ntrees,
            max_depth=max_depth,
            learn_rate=learn_rate,
            min_rows=min_rows,
            min_split_improvement=min_split_improvement
        )
        gbm_model.train(x=predictors, y=response_column, training_frame=frame_train)

        # Define a model name based on parameters
        modelname = f"GBMRegression_{ntrees}_{max_depth}"

        # Export the model
        export_model(modelname, gbm_model, when)

    except Exception as e:
        logger.exception("Error during training: %s", e)

def export_model(modelname, model, when):
    try:
        # Save the model to a directory (e.g., "./models")
        path = h2o.save_model(model=model, path="./models", force=True)
        logger.info("*** Model %s exported to: %s", modelname, path)
    except Exception as e:
        logger.exception("Error exporting model: %s", e)

if __name__ == "__main__":
    main()