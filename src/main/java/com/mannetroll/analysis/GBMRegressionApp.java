package com.mannetroll.analysis;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;
import hex.tree.gbm.GBMModel.GBMParameters;
import water.fvec.Frame;

public class GBMRegressionApp extends AbstractH2O {
    private final static Logger LOGGER = LoggerFactory.getLogger(GBMRegressionApp.class);

    public static void main(String[] args) {
        stall_till_cloudsize(1, 30000);
        try {
            String train = "../csv/small.csv";
            int ntrees = 30;
            int max_depth = 10;
            String when = "now";
            if (args.length == 1) {
                train = args[0];
            } else if (args.length == 4) {
                train = args[0];
                ntrees = Integer.parseInt(args[1]);
                max_depth = Integer.parseInt(args[2]);
                when = args[3];
            }
            train(train, ntrees, max_depth, when);
        } catch (Exception e) {
            LOGGER.error(e.getMessage(), e);
        }
        LOGGER.info("Done!");
        System.exit(0);
    }

    public static void train(String train, int ntrees, int max_depth, String when) {
        try {
            LOGGER.info("*** when: " + when);
            LOGGER.info("*** train: " + train);
            LOGGER.info("*** ntrees: " + ntrees);
            LOGGER.info("*** max_depth: " + max_depth);

            double learn_rate = 0.09;
            double min_rows = 9;
            double min_split_improvement = 1e-8;

            LOGGER.info("*** learn_rate: " + learn_rate);
            LOGGER.info("*** min_rows: " + min_rows);
            LOGGER.info("*** min_split_improvement: " + min_split_improvement);

            String[] ignoredColumns = { "Group", "EventTime", "EventIndex", "DTA", "Target", "Fraction" };
            LOGGER.info("*** length: " + ignoredColumns.length);
            LOGGER.info("*** ignoredColumns: " + Arrays.asList(ignoredColumns));

            Frame frame_train = parse_csv_file(train);
            LOGGER.info("*** frame_train: " + frame_train);

            GBMParameters parms = new GBMParameters();
            parms._train = frame_train._key;
            parms._ignored_columns = ignoredColumns;
            parms._response_column = "TTATA";
            parms._ntrees = ntrees;
            parms._max_depth = max_depth;
            parms._learn_rate = learn_rate;
            parms._min_rows = min_rows;
            parms._min_split_improvement = min_split_improvement;

            final String modelname = "GBMRegression_" + ntrees + "_" + max_depth;
            GBM gbm = new GBM(parms, water.Key.make(modelname));
            GBMModel model = gbm.trainModel().get();
            exportModel(modelname, model, when);
            //
        } catch (Exception e) {
            LOGGER.error(e.getMessage(), e);
        }
    }

}
