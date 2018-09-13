package com.iis.network;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
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

    private static Logger logger = LoggerFactory.getLogger(ClassificationNetworkV1.class);

    public void run() throws Exception {

        // The width and height of the images
        int width = 28;
        int height = 28;

        // Number of (color) channels; in case of grey scale set to 1
        int channels = 1;

        // Random seed for predictive results
        int seed = 123;
        Random random = new Random(seed);

        // Hyper parameters
        int batchSize = 128;
        int epochs = 15;

        // Number of (image) classes/types in the data set
        int numberOfClasses = 5;

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
        logger.info("*** Build model ***");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(100)
                        .activation(new ActivationReLU())
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(numberOfClasses)
                        .activation(new ActivationSoftmax())
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setIterationCount(1);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        logger.info("*** Training model ***");

        for (int i = 0; i < epochs; i++) {
            model.fit(iterator);
        }

        logger.info("*** Evaluate model ***");

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

        logger.info(evaluation.stats());

        logger.info("*** Saving model to 'trained_qd_model.zip' ***");

        File saveLocation = new File("trained_qd_model.zip");
        ModelSerializer.writeModel(model, saveLocation, false);
    }
}
