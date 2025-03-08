package com.mannetroll.analysis;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.io.FileUtils;

import hex.Model;
import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;
import hex.tree.gbm.GBMModel.GBMParameters;
import water.H2O;
import water.Key;
import water.fvec.Frame;
import water.fvec.NFSFileVec;
import water.parser.ParseDataset;

public class H2OJavaGBMExample {
	private static final Logger logger = Logger.getLogger(H2OJavaGBMExample.class.getName());

	public static void main(String[] args) {
		// Start the H2O cluster
		H2O.main(args);
		H2O.waitForCloudSize(1, 10000);

		train();

		H2O.shutdown(0);
	}

	public static void train() {
		// Setup training parameters
		String when = "now";
		String trainFile = "../csv/small.csv";
		int ntrees = 80;
		int maxDepth = 16;

		logger.info("*** when: " + when);
		logger.info("*** train: " + trainFile);
		logger.info("*** ntrees: " + ntrees);
		logger.info("*** max_depth: " + maxDepth);

		double learnRate = 0.09;
		int minRows = 9;
		double minSplitImprovement = 1e-8;

		logger.info("*** learn_rate: " + learnRate);
		logger.info("*** min_rows: " + minRows);
		logger.info("*** min_split_improvement: " + minSplitImprovement);

		List<String> ignoredColumns = Arrays.asList("Group", "EventTime", "EventIndex", "DTA", "TTATA", "Fraction");
		logger.info("*** length: " + ignoredColumns.size());
		logger.info("*** ignoredColumns: " + ignoredColumns);

		// Import the CSV file as an H2O Frame using NFSFileVec
		Frame frameTrain = null;
		try {
			NFSFileVec nfs = NFSFileVec.make(trainFile);
			frameTrain = ParseDataset.parse(Key.make(), new Key[] { nfs._key });
			logger.info("*** frame_train: " + frameTrain.toString());
		} catch (Exception e) {
			logger.log(Level.SEVERE, "Error importing file: " + trainFile, e);
			return;
		}

		// Define the response column and compute the predictor columns
		String responseColumn = "Target";
		List<String> predictors = new ArrayList<>();
		for (String col : frameTrain.names()) {
			if (!ignoredColumns.contains(col) && !col.equals(responseColumn)) {
				predictors.add(col);
			}
		}

		// Configure the GBM model parameters
		GBMParameters params = new GBMParameters();
		params._train = frameTrain._key;
		params._response_column = responseColumn;
		// Specify ignored columns; predictors will be inferred as the complement of
		// these and the response
		params._ignored_columns = ignoredColumns.toArray(new String[0]);

		params._ntrees = ntrees;
		params._max_depth = maxDepth;
		params._learn_rate = learnRate;
		params._min_rows = minRows;
		params._min_split_improvement = minSplitImprovement;

		// Train the GBM model
		GBM gbm = new GBM(params);
		GBMModel model = null;
		try {
			model = gbm.trainModel().get();
		} catch (Exception e) {
			logger.log(Level.SEVERE, "Error training model", e);
			return;
		}

		String modelName = "GBM_" + ntrees + "_" + maxDepth;
		try {
			exportModel(modelName, model, when);
		} catch (IOException e) {
			logger.log(Level.SEVERE, "Error exporting model", e);
		}

		// Clean up resources: remove model and frame from the H2O cluster
		if (model != null)
			model.remove();
		if (frameTrain != null)
			frameTrain.remove();
	}

	@SuppressWarnings("rawtypes")
	protected static void exportModel(final String modelname, Model model, String when) throws IOException {
		logger.info("*** model: " + model.toString());
		final String basename = "./work/" + modelname + "_" + when;
		FileUtils.writeStringToFile(new File(basename + ".json"), model._parms.toJsonString(),
				Charset.defaultCharset());
		final String category = model.modelDescriptor().getModelCategory().name();
		logger.info("*** Category: " + category);
		if (category.equals("Binomial")) {
			logger.info("*** AUC: " + model.auc());
			logger.info("*** logloss: " + model.logloss());
			logger.info("*** mean_per_class_error: " + model.mean_per_class_error());
		}
		if (category.equals("Regression")) {
			logger.info("*** mae: " + model.mae());
		}
		logger.info("*** loss: " + model.loss());
		logger.info("*** r2: " + model.r2());
		logger.info("*** mse: " + model.mse());
		logger.info("*** modelname: " + modelname);
		logger.info("*** basename: " + basename);
		// model.exportMojo(basename + ".zip", true);
		// model.exportBinaryModel(basename + ".h2o", true);
	}
}