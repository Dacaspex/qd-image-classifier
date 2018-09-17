package com.iis.main;

import com.iis.network.ClassificationNetworkV1;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class NetworkGUI extends Application implements CustomEventListener {
    public NetworkGUI() {
        super();
    }

    @Override
    public void init() throws Exception {
        super.init();
    }

    @Override
    public void stop() throws Exception {
        super.stop();
    }

    private XYChart.Series regressionScore = new XYChart.Series();
    private XYChart.Series validationScore = new XYChart.Series();
    @Override
    public void start(Stage stage) throws Exception {
        Group root = new Group();
        Scene scene  = new Scene(root,800,600);

        TabPane tabPane = new TabPane();

        BorderPane borderPane = new BorderPane();

        final LineChart<Number,Number> lineChart_regScore =
                new LineChart<Number,Number>(new NumberAxis(), new NumberAxis());
        final LineChart<Number,Number> lineChart_valScore =
                new LineChart<Number,Number>(new NumberAxis(), new NumberAxis());
        // Tabs
        Tab tab1 = new Tab();
        tab1.setText("Regression Score");
        tab1.setContent(lineChart_regScore);
        tabPane.getTabs().add(tab1);

        Tab tab2 = new Tab();
        tab2.setText("Validation Score");
        tab2.setContent(lineChart_valScore);
        tabPane.getTabs().add(tab2);

        stage.setTitle("Neural Network Quick Draw");

        lineChart_regScore.getData().addAll(regressionScore);
        lineChart_valScore.getData().addAll(validationScore);

        borderPane.prefHeightProperty().bind(scene.heightProperty());
        borderPane.prefWidthProperty().bind(scene.widthProperty());

        borderPane.setCenter(tabPane);
        root.getChildren().add(borderPane);

        stage.setScene(scene);
        stage.show();

        new Thread() {
            @Override
            public void run() {
                ClassificationNetworkV1 network = new ClassificationNetworkV1();
                network.gui = getInstance();
                try {network.run();} catch (Exception e) {}
            }
        }.start();
    }

    public NetworkGUI getInstance(){
        return this;
    }

    public void updateRegressionScore(final int iteration, final double modelScore) {
        Platform.runLater(new Runnable(){
            public void run() {
                regressionScore.getData().add(new XYChart.Data(iteration, modelScore));
            }
        });
    }

    public void updateEpochValidationScore(final int epoch, final double score) {
        Platform.runLater(new Runnable(){
            public void run() {
                validationScore.getData().add(new XYChart.Data(epoch, score));
            }
        });
    }
}
