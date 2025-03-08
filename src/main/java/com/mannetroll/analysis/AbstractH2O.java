package com.mannetroll.analysis;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import hex.Model;
import water.ExtensionManager;
import water.H2O;
import water.Key;
import water.fvec.Frame;
import water.fvec.NFSFileVec;
import water.parser.ParseDataset;

public abstract class AbstractH2O {
    private final static Logger LOGGER = LoggerFactory.getLogger(AbstractH2O.class);
    protected static boolean _stall_called_before = false;
    protected static int _initial_keycnt = 0;

    @SuppressWarnings("rawtypes")
    protected static void exportModel(final String modelname, Model model, String when) throws IOException {
        LOGGER.info("*** model: " + model.toString());
        final String basename = "./work/" + modelname + "_" + when;
        FileUtils.writeStringToFile(new File(basename + ".json"), model._parms.toJsonString());
        final String category = model.modelDescriptor().getModelCategory().name();
        LOGGER.info("*** Category: " + category);
        if (category.equals("Binomial")) {
            LOGGER.info("*** AUC: " + model.auc());
            LOGGER.info("*** logloss: " + model.logloss());
            LOGGER.info("*** mean_per_class_error: " + model.mean_per_class_error());
        }
        if (category.equals("Regression")) {
            LOGGER.info("*** mae: " + model.mae());
        }
        LOGGER.info("*** loss: " + model.loss());
        LOGGER.info("*** r2: " + model.r2());
        LOGGER.info("*** mse: " + model.mse());
        LOGGER.info("*** modelname: " + modelname);
        LOGGER.info("*** basename: " + basename);
        model.exportMojo(basename + ".zip", true);
        model.exportBinaryModel(basename + ".h2o", true);
    }

    public static void stall_till_cloudsize(int x, int timeout) {
        stall_till_cloudsize(new String[] {}, x, timeout);
    }

    public static void stall_till_cloudsize(String[] args, int x, int timeout) {
        x = Math.max(1, x);
        if (!_stall_called_before) {
            H2O.main(args);
            //H2O.registerRestApis("./");
            ExtensionManager.getInstance().registerRestApiExtensions();
            _stall_called_before = true;
        }
        H2O.waitForCloudSize(x, timeout);
        _initial_keycnt = H2O.store_size();
        // Finalize registration of REST API to enable tests which are touching Schemas.
        H2O.startServingRestApi();
    }

    public static Frame parse_csv_file(String fname) {
        String name = fname;
        int lastIndexOf = fname.lastIndexOf('/');
        if (lastIndexOf > 0) {
            name = fname.substring(lastIndexOf + 1, fname.length());
        }
        return parse_test_file(Key.make(name), fname);
    }

    public static Frame parse_test_file(Key<?> outputKey, String fname) {
        NFSFileVec nfs = makeNfsFileVec(fname);
        return ParseDataset.parse(outputKey, nfs._key);
    }

    public static NFSFileVec makeNfsFileVec(String fname) {
        try {
            return NFSFileVec.make(fname);
        } catch (IOException ioe) {
            return null;
        }
    }
}
