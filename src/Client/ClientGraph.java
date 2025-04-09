package Client;

import java.io.*;
import java.net.Socket;
import java.util.Scanner;

public class ClientGraph {
    public static void main(String[] args) {
        String sServerAddress = args[0];
        int nPort = Integer.parseInt(args[1]);

        try {
            Socket clientEndpoint = new Socket(sServerAddress, nPort);
            System.out.println("Client: Connected to server at " + clientEndpoint.getRemoteSocketAddress());

            DataOutputStream dosWriter = new DataOutputStream(clientEndpoint.getOutputStream());
            DataInputStream disReader = new DataInputStream(clientEndpoint.getInputStream());

            String sentFileName = "graph4.txt";
            File fileToSend = new File(sentFileName);
            if (!fileToSend.exists()) {
                System.out.println("Client: File '" + sentFileName + "' not found!");
                clientEndpoint.close();
                return;
            }

            FileInputStream fileInputStream = new FileInputStream(fileToSend);
            byte[] buffer = new byte[4096];
            int bytes = 0;

            dosWriter.writeLong(fileToSend.length());
            System.out.println("Client: Sending file '" + sentFileName + "' (" + fileToSend.length() + " byte/s)");

            while ((bytes = fileInputStream.read(buffer)) != -1) {
                dosWriter.write(buffer, 0, bytes);
            }
            dosWriter.flush();
            fileInputStream.close();
            System.out.println("Client: File sent!");

            Scanner scanner = new Scanner(System.in);
            String serverResponse = "";

            while (!serverResponse.equals("exit")) {
                // Wait for user input and send the command to the server
                System.out.print("Client: Enter a command: ");
                String command = scanner.nextLine();
                dosWriter.writeUTF(command);
                dosWriter.flush();

                // Wait for a response from the server
                serverResponse = disReader.readUTF();
                System.out.println("Client: Server response: " + serverResponse);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("Client: Connection is terminated.");
        }
    }
}