package com.iis.network;

import com.iis.main.CustomEventListener;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class ClassificationNetworkV1 {
    private static final Logger log = LoggerFactory.getLogger(ClassificationNetworkV1.class);

    // The width and height of the images
    private static final int width = 28;
    private static final int height = 28;

    // Number of (color) channels; in case of grey scale set to 1
    private static final int channels = 1;

    // Number of classes
    private static final int numberOfClasses = 21;

    // Hyper parameters
    private static final int batchSize = 32;
    private static final int epochs = 5;

    // Reload model and keep training or train new model
    private static final boolean load_model = true;

    // Extra parameters
    private static final boolean detailed_per_epoch_evaluation = true;
    private static final boolean save_model_every_epoch = true;

    public void run() throws Exception {
        // Random seed for predictive results
        int seed = 123;
        Random random = new Random(seed);

        // Get the data
        File trainingData = new File("data/training");
        File validationData = new File("data/validation");

        // File split
        FileSplit training = new FileSplit(trainingData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileSplit validation = new FileSplit(validationData, NativeImageLoader.ALLOWED_FORMATS, random);

        // Extract label from parent path
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        // Record image reader
        ImageRecordReader recordReaderTrain = new ImageRecordReader(height, width, channels, labelGenerator);
        recordReaderTrain.initialize(training);
        ImageRecordReader recordReaderValidate = new ImageRecordReader(height, width, channels, labelGenerator);
        recordReaderValidate.initialize(validation);

        // Dataset iterator
        DataSetIterator iteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, numberOfClasses);
        DataSetIterator iteratorValidate = new RecordReaderDataSetIterator(recordReaderValidate, batchSize, 1, numberOfClasses);

        // Scale pixel values to [0, 1]
        DataNormalization scalerTrain = new ImagePreProcessingScaler(0, 1);
        scalerTrain.fit(iteratorTrain);
        iteratorTrain.setPreProcessor(scalerTrain);
        DataNormalization scalerValidate = new ImagePreProcessingScaler(0, 1);
        scalerValidate.fit(iteratorValidate);
        iteratorValidate.setPreProcessor(scalerValidate);

        // Build our model
        MultiLayerNetwork model;
        log.info("*** Build model ***");
        if (!load_model){
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs(0.004, 0.8))
                    .l2(0.0001)
                    .list()
                    ///////////
                    //Layer 1//
                    ///////////
                    .layer(0, new Convolution2D.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0})
                            .convolutionMode(ConvolutionMode.Same)
                            .nIn(1)
                            .nOut(96)
                            .weightInit(WeightInit.XAVIER_UNIFORM)
                            .activation(Activation.RELU)
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2})
                            .build())
                    .layer(2, new DropoutLayer.Builder()
                            .dropOut(0.99)
                            .build())
                    ///////////
                    //Layer 2//
                    ///////////
                    .layer(3, new Convolution2D.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
                            .convolutionMode(ConvolutionMode.Same)
                            .nOut(128)
                            .weightInit(WeightInit.XAVIER_UNIFORM)
                            .activation(Activation.RELU)
                            .build())
                    .layer(4, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2})
                            .build())
                    .layer(5, new DropoutLayer.Builder()
                            .dropOut(0.99)
                            .build())
                    ///////////
                    //Layer 3//
                    ///////////
                    .layer(6, new Convolution2D.Builder(new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0})
                            .convolutionMode(ConvolutionMode.Same)
                            .nOut(160)
                            .weightInit(WeightInit.XAVIER_UNIFORM)
                            .activation(Activation.RELU)
                            .build())
                    .layer(7, new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2, 2})
                            .build())
                    .layer(8, new DropoutLayer.Builder()
                            .dropOut(0.99)
                            .build())
                    ///////////
                    //Layer 4//
                    ///////////
                    .layer(9, new DenseLayer.Builder()
                            .nOut(2304)
                            .activation(new ActivationReLU())
                            .weightInit(WeightInit.XAVIER)
                            .build()
                    )
                    .layer(10, new DropoutLayer.Builder()
                            .dropOut(0.95)
                            .build())
                    .layer(11, new DenseLayer.Builder()
                            .nOut(1152)
                            .activation(new ActivationReLU())
                            .weightInit(WeightInit.XAVIER)
                            .build()
                    )
                    .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(numberOfClasses)
                            .activation(new ActivationSoftmax())
                            .weightInit(WeightInit.XAVIER)
                            .build()
                    )
                    .pretrain(false)
                    .backprop(true)
                    .setInputType(InputType.convolutional(height, width, channels))
                    .build();
            model = new MultiLayerNetwork(conf);
        } else {
            model = ModelSerializer.restoreMultiLayerNetwork(new File("trained_qd_model.zip"));
        }

        model.setIterationCount(1);
        model.init();

        model.setListeners(new CustomScoreIterationListener(300));

        log.info("*** Training model ***");

        Evaluation evaluation;
        for (int i = 0; i < epochs; i++) {
            model.fit(iteratorTrain);
            log.info("*** Epoch " + i + " Done ***");

            evaluation = runEvaluation(model, recordReaderValidate, validation, scalerValidate);
            if (detailed_per_epoch_evaluation) {
                log.info(evaluation.stats(true, true));
            }
            gui.updateEpochValidationScore(i, evaluation.accuracy());

            // Saving Model
            if (save_model_every_epoch) saveModel(model, i);
        }

        if (!detailed_per_epoch_evaluation) {
            log.info("*** Evaluate model ***");
            log.info(evaluation.stats(true, true));
        }

        if (!save_model_every_epoch) saveModel(model, epochs);
    }

    public Evaluation runEvaluation(MultiLayerNetwork model, ImageRecordReader recordReader, FileSplit validation, DataNormalization scaler) throws Exception{
        log.info("*** Evaluate model ***");
        recordReader.reset();
        recordReader.initialize(validation);
        DataSetIterator testIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numberOfClasses);
        scaler.fit(testIterator);
        testIterator.setPreProcessor(scaler);
        // Create evaluation object
        Evaluation evaluation = new Evaluation(numberOfClasses);
        while (testIterator.hasNext()) {
            DataSet next = testIterator.next();
            INDArray output = model.output(next.getFeatures());
            evaluation.eval(next.getLabels(), output);
        }
        return evaluation;
    }

    public void saveModel(MultiLayerNetwork model, int i) throws Exception{
        log.info("*** Saving model to 'trained_qd_model.zip' ***");
        File saveLocation = new File("trained_qd_model" + i + ".zip");
        ModelSerializer.writeModel(model, saveLocation, false);
    }

    public CustomEventListener gui;

    public class CustomScoreIterationListener extends ScoreIterationListener{
        private int printIterations = 10;

        public CustomScoreIterationListener(int printIterations) {
            this.printIterations = printIterations;
        }

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            super.iterationDone(model, iteration, epoch);
            if (iteration % printIterations == 0) {
                gui.updateRegressionScore(iteration, model.score());
            }
        }
    }

}
