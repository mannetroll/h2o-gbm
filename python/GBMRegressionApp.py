import sys
import os
import json
import logging
import h2o
from h2o.estimators import H2OGradientBoostingEstimator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize H2O; this will also wait for a cluster to form.
    h2o.init()

    # Default parameters
    train_file = "../csv/small.csv"
    ntrees = 30
    max_depth = 10
    when = "now"

    # Parse command-line arguments
    # Expecting either 1 argument (train file) or 4 arguments (train file, ntrees, max_depth, when)
    if len(sys.argv) == 2:
        train_file = sys.argv[1]
    elif len(sys.argv) == 5:
        train_file = sys.argv[1]
        ntrees = int(sys.argv[2])
        max_depth = int(sys.argv[3])
        when = sys.argv[4]

    train_model(train_file, ntrees, max_depth, when)
    logger.info("Done!")
    h2o.shutdown(prompt=False)

def train_model(train_file, ntrees, max_depth, when):
    try:
        logger.info("*** when: %s", when)
        logger.info("*** train: %s", train_file)
        logger.info("*** ntrees: %d", ntrees)
        logger.info("*** max_depth: %d", max_depth)

        # Set hyperparameters
        learn_rate = 0.09
        min_rows = 9
        min_split_improvement = 1e-8

        logger.info("*** learn_rate: %f", learn_rate)
        logger.info("*** min_rows: %d", min_rows)
        logger.info("*** min_split_improvement: %e", min_split_improvement)

        # Columns to ignore (if they exist in the dataset)
        ignored_columns = ["Group", "EventTime", "EventIndex", "DTA", "Target", "Fraction"]
        logger.info("*** Number of ignored columns: %d", len(ignored_columns))
        logger.info("*** Ignored Columns: %s", ignored_columns)

        # Load the training data
        frame_train = h2o.import_file(train_file)
        logger.info("*** frame_train: %s", frame_train)

        # Define response and predictor columns
        response_column = "TTATA"
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

        # Define model name
        modelname = f"GBMRegression_{ntrees}_{max_depth}"
        export_model(modelname, gbm_model, when)

    except Exception as e:
        logger.exception("Error during training: %s", e)

def export_model(modelname, model, when):
    try:
        # Log model string representation
        logger.info("*** Model details: %s", model)

        # Create a basename for exported files
        basename = os.path.join("./models", f"{modelname}_{when}")
        os.makedirs("./models", exist_ok=True)

        # Export model parameters as JSON
        params_json = model.params
        json_path = basename + ".json"
        with open(json_path, "w") as f:
            json.dump(params_json, f, indent=2)
        logger.info("*** Model parameters saved to: %s", json_path)

        # Retrieve model category from the model output
        model_category = model._model_json["output"]["model_category"]
        logger.info("*** Model Category: %s", model_category)

        # Print metrics based on the model category
        if model_category == "Binomial":
            logger.info("*** AUC: %s", model.auc())
            logger.info("*** Logloss: %s", model.logloss())
            logger.info("*** Mean Per Class Error: %s", model.mean_per_class_error())
        elif model_category == "Regression":
            logger.info("*** MAE: %s", model.mae())

        # Print additional metrics
        # (Some metrics may not be available depending on the model type)
        try:
            logger.info("*** R2: %s", model.r2())
        except Exception:
            logger.info("*** R2 metric not available")
        try:
            logger.info("*** MSE: %s", model.mse())
        except Exception:
            logger.info("*** MSE metric not available")
        try:
            # Loss might be available as part of training metrics
            loss = model._model_json["output"].get("training_metrics", {}).get("MSE", "N/A")
            logger.info("*** Loss (MSE): %s", loss)
        except Exception:
            logger.info("*** Loss metric not available")

        logger.info("*** Model Name: %s", modelname)
        logger.info("*** Basename: %s", basename)

        # Export model as MOJO if available
        try:
            # For MOJO export, h2o provides download_mojo
            mojo_path = h2o.download_mojo(model, path="./models", get_genmodel_jar=False)
            # Rename the downloaded MOJO to match our basename
            new_mojo_path = basename + ".zip"
            os.rename(mojo_path, new_mojo_path)
            logger.info("*** MOJO exported to: %s", new_mojo_path)
        except Exception as e:
            logger.exception("*** Error exporting MOJO: %s", e)

        # Export model as a binary model
        try:
            binary_path = h2o.save_model(model=model, path="./models", force=True)
            # Optionally, rename or move the file to match the basename convention
            new_binary_path = basename + ".h2o"
            os.rename(binary_path, new_binary_path)
            logger.info("*** Binary model exported to: %s", new_binary_path)
        except Exception as e:
            logger.exception("*** Error exporting binary model: %s", e)

    except Exception as e:
        logger.exception("Error during export: %s", e)

if __name__ == "__main__":
    main()