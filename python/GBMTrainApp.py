#
# pip install h2o
#
import h2o
import logging
import json
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def train():
    # Setup training parameters
    when = "20250308_080101"
    train_file = "../csv/small.csv"
    ntrees = 80
    max_depth = 16

    logging.info("*** when: " + when)
    logging.info("*** train: " + train_file)
    logging.info("*** ntrees: " + str(ntrees))
    logging.info("*** max_depth: " + str(max_depth))

    learn_rate = 0.09
    min_rows = 9
    min_split_improvement = 1e-8

    logging.info("*** learn_rate: " + str(learn_rate))
    logging.info("*** min_rows: " + str(min_rows))
    logging.info("*** min_split_improvement: " + str(min_split_improvement))

    ignored_columns = ["Group", "EventTime", "EventIndex", "DTA", "TTATA", "Fraction"]
    logging.info("*** length: " + str(len(ignored_columns)))
    logging.info("*** ignoredColumns: " + str(ignored_columns))

    # Import the CSV file as an H2OFrame (similar to parse_csv_file in Java)
    frame_train = h2o.import_file(train_file)
    logging.info("*** frame_train: " + str(frame_train))

    # Define the response and predictors
    response_column = "Target"
    predictors = [col for col in frame_train.col_names 
                  if col not in ignored_columns + [response_column]]

    # Initialize and train the GBM model using H2OGradientBoostingEstimator
    model = H2OGradientBoostingEstimator(
        ntrees=ntrees,
        max_depth=max_depth,
        learn_rate=learn_rate,
        min_rows=min_rows,
        min_split_improvement=min_split_improvement
    )
    model.train(x=predictors, y=response_column, training_frame=frame_train)

    model_name = f"GBM_{ntrees}_{max_depth}"
    export_model(model_name, model, when)

def export_model(model_name, model, when):
    logging.info("*** model: " + str(model))
    basename = "./" + model_name + "_" + when

    # Save model parameters to a JSON file (analogous to model._parms.toJsonString())
    with open(basename + ".json", "w") as f:
        json.dump(model.params, f, indent=4)

    # Determine the model category and log corresponding metrics
    model_category = model._model_json['output']['model_category']
    logging.info("*** Category: " + model_category)
    if model_category == "Binomial":
        logging.info("*** AUC: " + str(model.auc()))
        logging.info("*** logloss: " + str(model.logloss()))
        logging.info("*** mean_per_class_error: " + str(model.mean_per_class_error()))
    elif model_category == "Regression":
        logging.info("*** mae: " + str(model.mae()))

    logging.info("*** r2: " + str(model.r2()))
    logging.info("*** mse: " + str(model.mse()))
    logging.info("*** modelname: " + model_name)
    logging.info("*** basename: " + basename)

    # Export the model as a MOJO
    try:
        mojo_path = model.download_mojo(path=".", get_genmodel_jar=False)
        logging.info("*** MOJO exported to: " + mojo_path)
    except Exception as e:
        logging.error("*** Error exporting MOJO: " + str(e))

    # Export the binary model
    try:
        # Save the model to a directory (e.g., "./models")
        saved_model_path = h2o.save_model(model=model, path="./models", force=True)
        logging.info("*** Binary model exported to: " + saved_model_path)
    except Exception as e:
        logging.error("*** Error exporting binary model: " + str(e))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    h2o.init()
    train()