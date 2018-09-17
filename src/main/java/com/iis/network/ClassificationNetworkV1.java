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

public class ClassificationNetworkV1 {
    private static final Logger log = LoggerFactory.getLogger(ClassificationNetworkV1.class);

    // The width and height of the images
    private static final int width = 28;
    private static final int height = 28;

    // Number of (color) channels; in case of grey scale set to 1
    private static final int channels = 1;

    // Number of classes
    private static final int numberOfClasses = 31;

    // Hyper parameters
    private static final int batchSize = 256;
    private static final int epochs = 20;

    // Reload model and keep training or train new model
    private static final boolean load_model = false;

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
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelGenerator);
        recordReader.initialize(training);

        // Dataset iterator
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numberOfClasses);

        // Scale pixel values to [0, 1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

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
                            .nOut(64)
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
                            .nOut(96)
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
                            .nOut(128)
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
                            .nOut(2048)
                            .activation(new ActivationReLU())
                            .weightInit(WeightInit.XAVIER)
                            .build()
                    )
                    .layer(10, new DropoutLayer.Builder()
                            .dropOut(0.95)
                            .build())
                    .layer(11, new DenseLayer.Builder()
                            .nOut(1024)
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

        model.setListeners(new CustomScoreIterationListener(100));

        log.info("*** Training model ***");

        for (int i = 0; i < epochs; i++) {
            model.fit(iterator);
            log.info("*** Epoch " + i + " Done ***");

            Evaluation evaluation = new Evaluation(numberOfClasses);
            ImageRecordReader recordReaderVal = new ImageRecordReader(height, width, channels, labelGenerator);
            recordReaderVal.initialize(validation);
            DataSetIterator testIterator = new RecordReaderDataSetIterator(recordReaderVal, batchSize, 1, numberOfClasses);
            scaler.fit(testIterator);
            testIterator.setPreProcessor(scaler);
            while (testIterator.hasNext()) {
                DataSet next = testIterator.next();
                INDArray output = model.output(next.getFeatures());
                evaluation.eval(next.getLabels(), output);
            }
            gui.updateEpochValidationScore(i, evaluation.accuracy());
        }

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

        log.info(evaluation.stats(true, true));

        log.info("*** Saving model to 'trained_qd_model.zip' ***");

        File saveLocation = new File("trained_qd_model.zip");
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
