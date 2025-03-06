//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.

import com.mxgraph.layout.mxCircleLayout;
import com.mxgraph.layout.mxIGraphLayout;
import com.mxgraph.util.mxCellRenderer;
import org.jgrapht.DirectedGraph;
import org.jgrapht.Graph;
import org.jgrapht.ext.JGraphXAdapter;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;



public class Main {
    private static Map<String, List<String>> adjList =  new HashMap<>();
    private static DirectedGraph<String, DefaultEdge> directedGraph
            = new DefaultDirectedGraph<>(DefaultEdge.class);

    private static boolean parallelComp = false;
    private static final String file = "src/graph-sq-100";

    public static void main(String[] args) {
        parseFile();

        String input = "";


        while (!input.equals("exit")) {
            System.out.print("Graph Query: ");
            Scanner scanner = new Scanner(System.in);
            input = scanner.nextLine();

            if (input.equals("nodes")) {
                getNodes();
            }
            else if (input.equals("edges")){
                getEdges();
            }
            else if(input.startsWith("node ")){
                String[] tokenedinput = parseInput(input, 2, "wrong parse");
                String node = tokenedinput[1];

                long startTime = System.currentTimeMillis();
                if(checkValidNode(node)){
                    System.out.println( node + " is a node\n");
                } else{
                    System.out.println("Node not found.\n");
                }
                long endTime = System.currentTimeMillis();
                System.out.println("Search took " + durationTimer(startTime, endTime) + " ms.\n");
            }
            else if(input.startsWith("edge ")){
                String[] tokenedinput = parseInput(input, 3, "wrong parse");
                String nodeSource = tokenedinput[1];
                String nodeTarget = tokenedinput[2];

                long startTime = System.currentTimeMillis();
                if (checkValidEdge(nodeSource, nodeTarget)) {
                    System.out.println(nodeSource + " to " + nodeTarget + " is an edge\n");
                }
                else {
                    System.out.println("Edge not found.\n");
                }
                long endTime = System.currentTimeMillis();
                System.out.println("Search took " + durationTimer(startTime, endTime) + " ms.\n");
            }
            else if(input.equals("visualize")){
                visualizeGraph();
            }
            else if(input.startsWith("path ")){
                String[] tokenedinput = parseInput(input, 3, "wrong parse");
                String nodeSource = tokenedinput[1];
                String nodeTarget = tokenedinput[2];

                long startTime = System.currentTimeMillis();
                if(checkValidNode(nodeSource) && checkValidNode(nodeTarget)){
                    String path = getPath(nodeSource, nodeTarget);
                    System.out.println(path);
                }
                else{
                    System.out.println("Invalid node/s.\n");
                }
                long endTime = System.currentTimeMillis();
                System.out.println("Search took " + durationTimer(startTime, endTime) + " ms.\n");
            }
            else if(input.equals("exit")){
                System.out.println("Goodbye!\n");
            }
            else if(input.equals("toggle-para")){
                if(parallelComp){
                    parallelComp = false;
                    System.out.println("Parallel computation disabled.\n");
                }
                else{
                    parallelComp = true;
                    System.out.println("Parallel computation enabled.\n");
                }
            }
            else if(input.equals("load-file")){
                parseFile();
                System.out.println(file + " loaded successfully.\n");
            }
            else {
                System.out.println("Invalid command.\n");
            }
        }

    }

    private static long durationTimer(long startTime, long endTime) {
        long duration = endTime - startTime;
        return duration;
    }

    private static String[] parseInput(String input, int expectedParts, String errorMessage) {
        String[] parts = input.split("\\s+"); // Split by one or more spaces
        if (parts.length < expectedParts) { // Ensure the required number of parts
            System.out.println(errorMessage + "\n");
            return null; // Return null to indicate invalid input
        }
        return parts;
    }


    private static void parseFile() {
        try {
            File graphFile = new File(file);
            Scanner reader = new Scanner(graphFile);
            while (reader.hasNextLine()) {
                String data = reader.nextLine();
                processLine(data);
            }
            reader.close();
            System.out.println("Graph loaded successfully. \n");
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred loading the graph.");
            e.printStackTrace();
        }
    }

    private static void processLine(String data){
        if (data.startsWith("*")) {
            String[] parts = parseInput(data, 2, "wrong parse");
            String node = parts[1];
            adjList.putIfAbsent(node, new ArrayList<>());
            directedGraph.addVertex(node);
        }
        else if (data.startsWith("-")) {
            String[] parts = parseInput(data, 3, "wrong parse");
            String nodeSource = parts[1];
            String nodeTarget = parts[2];
            adjList.get(nodeSource).add(nodeTarget);
            directedGraph.addEdge(nodeSource, nodeTarget);
        }
    }

    private static void getEdges(){

        System.out.print("Edges: ");
        adjList.forEach((node, neighbors) -> {
            for (int i = 0; i < neighbors.size(); i++) {
                String neighbor = neighbors.get(i);
                System.out.print("(" + node + ", " + neighbor + ")");
                System.out.print(", ");
            }
        });

        System.out.println("\n");
    }
    private static void getNodes(){
        int index = 0;
        System.out.print("Nodes: ");
        for (String node : adjList.keySet()) {
            System.out.print(node);
            if (index < adjList.keySet().size() - 1) {
                System.out.print(", ");
            }
            index++;
        }

        System.out.println("\n");
    }
    private static String getPath(String nodeSource, String nodeTarget){
        if(!parallelComp) {
            List<String> path = new ArrayList<>();
            Set<String> visited = new HashSet<>();
            if (dfsFindPath(nodeSource, nodeTarget, visited, path)) {
                return String.join(" -> ", path); // Construct the path as a string
            } else {
                return "No path found from " + nodeSource + " to " + nodeTarget;
            }
        }
        else{
            // Multithreaded DFS
            List<String> path = new ArrayList<>();
            if (parallelDfsFindPath(nodeSource, nodeTarget, path)) {
                return String.join(" -> ", path);
            } else {
                return "No path found from " + nodeSource + " to " + nodeTarget;
            }

        }
    }
    private static boolean parallelDfsFindPath(String nodeSource, String nodeTarget, List<String> path) {
        ForkJoinPool pool = new ForkJoinPool(6);
        AtomicBoolean pathFound = new AtomicBoolean(false);
        CopyOnWriteArrayList<String> concurrentPath = new CopyOnWriteArrayList<>();

        try {
            pool.invoke(new ParallelDFS(nodeSource, nodeTarget, new ArrayList<>(), new HashSet<>(), pathFound, concurrentPath));
        } finally {
            pool.shutdown();
        }

        if (pathFound.get()) {
            path.addAll(concurrentPath);
            return true;
        }
        return false;
    }

    private static class ParallelDFS extends RecursiveTask<Boolean> {
        private final String currentNode;
        private final String targetNode;
        private final List<String> currentPath;
        private final Set<String> localVisited;
        private final AtomicBoolean pathFound;
        private final CopyOnWriteArrayList<String> concurrentPath;

        public ParallelDFS(String currentNode, String targetNode, List<String> currentPath,
                           Set<String> localVisited, AtomicBoolean pathFound, CopyOnWriteArrayList<String> concurrentPath) {
            this.currentNode = currentNode;
            this.targetNode = targetNode;
            this.currentPath = new ArrayList<>(currentPath);
            this.localVisited = new HashSet<>(localVisited);
            this.pathFound = pathFound;
            this.concurrentPath = concurrentPath;
        }

        @Override
        protected Boolean compute() {
            if (pathFound.get()) return false;

            System.out.println("Thread " + Thread.currentThread().getName() + " is processing node: " + currentNode);

            localVisited.add(currentNode);
            currentPath.add(currentNode);

            if (currentNode.equals(targetNode)) {
                pathFound.set(true);
                concurrentPath.clear();
                concurrentPath.addAll(currentPath);
                return true;
            }



            List<ParallelDFS> subTasks = new ArrayList<>();
            for (String neighbor : adjList.getOrDefault(currentNode, new ArrayList<>())) {
                if (!localVisited.contains(neighbor) && !pathFound.get()) {
                    ParallelDFS subTask = new ParallelDFS(neighbor, targetNode, currentPath, localVisited, pathFound, concurrentPath);
                    subTask.fork();
                    subTasks.add(subTask);
                    System.out.println("Thread " + Thread.currentThread().getName() + " is spawning a task for neighbor: " + neighbor);
                }
            }

            for (ParallelDFS task : subTasks) {
                if (task.join()) {
                    return true;
                }
            }

            return false;
        }
    }




    private static boolean dfsFindPath(String current, String target, Set<String> visited, List<String> path) {
        path.add(current);  // Add the current node to the path
        visited.add(current);  // Mark the current node as visited

        if (current.equals(target)) {  // Base case: if current node is the target node
            return true;
        }

        // Walk through all neighbors of the current node
        for (String neighbor : adjList.getOrDefault(current, new ArrayList<>())) {
            if (!visited.contains(neighbor)) {
                if (dfsFindPath(neighbor, target, visited, path)) {
                    return true;  // Return true if the target is found via this neighbor
                }
            }
        }

        // If no valid path found, backtrack
        path.remove(path.size() - 1);
        return false;
    }

    private static void visualizeGraph(){
        JGraphXAdapter<String, DefaultEdge> graphAdapter =
                new JGraphXAdapter<String, DefaultEdge>(directedGraph);
        mxIGraphLayout layout = new mxCircleLayout(graphAdapter);
        layout.execute(graphAdapter.getDefaultParent());

        BufferedImage image =
                mxCellRenderer.createBufferedImage(graphAdapter, null, 2, Color.WHITE, true, null);
        File imgFile = new File("src/graph2.png");

        try {
            ImageIO.write(image, "PNG", imgFile);
        }
        catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static boolean checkValidNode(String node){
        List<String> allNodes = adjList.keySet().stream().toList();
        int mid = allNodes.size()/2;
        List<String> firstHalf = allNodes.subList(0, mid);
        List<String> secondHalf = allNodes.subList(mid, allNodes.size());
        //linear search
        if(!parallelComp){
            for(String currentNode: allNodes){
                if(currentNode.equals(node))
                    return true;
            }
            return false;
        }
        //threads spawn and search half the list
        else{
            // create 2 threads to search for node, first one to get a valid node ends both threads and returns true
            AtomicBoolean nodeFound = new AtomicBoolean(false);

            Thread firstThread = new Thread(()-> checkValidNodeThread(node, nodeFound, firstHalf));
            Thread secondThread = new Thread(()-> checkValidNodeThread(node, nodeFound, secondHalf));

            firstThread.start();
            secondThread.start();

            try {
                firstThread.join();
                secondThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                e.printStackTrace();
            }

            return nodeFound.get();
        }
    }

    private static void checkValidNodeThread(String node, AtomicBoolean nodeFound, List<String> subNode){
        for(String currentNode: subNode){
            if(nodeFound.get())
                return;
            if(currentNode.equals(node))
                nodeFound.set(true);
        }
    }


    private static boolean checkValidEdge(String nodeSource, String nodeTarget){
        List<String> allNeighbors = adjList.get(nodeSource).stream().toList();
        int mid = allNeighbors.size()/2;
        List<String> firstHalf = allNeighbors.subList(0, mid);
        List<String> secondHalf = allNeighbors.subList(mid, allNeighbors.size());
        if(!parallelComp){
            if(allNeighbors.contains(nodeTarget))
                return true;
            return false;
        }
        else{
            AtomicBoolean edgeFound = new AtomicBoolean(false);

            Thread firstThread = new Thread(()-> checkValidEdgeThread(nodeTarget, edgeFound, firstHalf));
            Thread secondThread = new Thread(()-> checkValidEdgeThread(nodeTarget, edgeFound, secondHalf));

            firstThread.start();
            secondThread.start();

            try {
                firstThread.join();
                secondThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                e.printStackTrace();
            }

            return edgeFound.get();
        }
    }

    private static void checkValidEdgeThread(String nodeTarget, AtomicBoolean edgeFound, List<String> subNeighbors){
        for(String neighbor: subNeighbors){
            if(edgeFound.get())
                return;
            if(neighbor.equals(nodeTarget))
                edgeFound.set(true);
        }
    }
}