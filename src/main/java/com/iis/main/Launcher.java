package com.iis.main;

import com.iis.network.ClassificationNetworkV1;

public class Launcher {

    /**
     * The launcher is responsible for starting up networks
     *
     * @param args arguments (empty)
     */
    public static void main(String[] args) throws Exception {
        NetworkGUI gui = new NetworkGUI();
        javafx.application.Application.launch(NetworkGUI.class);
    }
}
