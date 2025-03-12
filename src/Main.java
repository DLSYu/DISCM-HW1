//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.

import com.mxgraph.layout.mxCircleLayout;
import com.mxgraph.layout.mxIGraphLayout;
import com.mxgraph.util.mxCellRenderer;
import org.antlr.v4.runtime.misc.Pair;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;


public class Main {
    private static Map<String, List<Edge>> adjList = new HashMap<>();

    static class Edge {
        String target;
        int weight;

        Edge(String target, int weight) {
            this.target = target;
            this.weight = weight;
        }
    }

    private static DirectedGraph<String, DefaultEdge> directedGraph
            = new DefaultDirectedGraph<>(DefaultEdge.class);

    private static boolean parallelComp = false;
    private static final String file = "src/graph.txt";

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
            else if (input.startsWith("prime-path")) {
                String[] tokens = parseInput(input, 3, "Invalid prime-path query.");
                String source = tokens[1];
                String target = tokens[2];
                long start = System.currentTimeMillis();
                if (checkValidNode(source) && checkValidNode(target)) {
                    PathResult result = parallelComp ? getPrimePathParallel(source, target) : getPrimePath(source, target);
                    if (result != null) {
                        System.out.println("prime path: " + String.join(" -> ", result.getPath()) + " with weight/length=" + result.getWeight());
                    } else {
                        System.out.println("No prime path from " + source + " to " + target);
                    }
                } else {
                    System.out.println("Invalid nodes.");
                }
                long end = System.currentTimeMillis();
                System.out.println("Search took " + (end - start) + " ms.");
            }
            else if (input.startsWith("shortest-path")) {
                String[] tokenedinput = parseInput(input, 3, "wrong parse");
                String nodeSource = tokenedinput[1];
                String nodeTarget = tokenedinput[2];

                long startTime = System.currentTimeMillis();
                if(checkValidNode(nodeSource) && checkValidNode(nodeTarget)){
                    PathResult result = parallelComp ?
                            findShortestPathParallel(nodeSource, nodeTarget) :
                            findShortestPath(nodeSource, nodeTarget);
                    if (result != null) {
                        System.out.println("shortest prime path: " + String.join(" -> ", result.getPath()) +
                                " with weight/length=" + result.getWeight());
                    } else {
                        System.out.println("No path from " + nodeSource + " to " + nodeTarget);
                    }
                }
                else{
                    System.out.println("Invalid node/s.\n");
                }
                long endTime = System.currentTimeMillis();
                System.out.println("Search took " + durationTimer(startTime, endTime) + " ms.\n");
            }
            else if (input.startsWith("shortest-prime-path")) {
                String[] tokens = parseInput(input, 3, "Invalid shortest-prime-path query.");
                String source = tokens[1];
                String target = tokens[2];
                long start = System.currentTimeMillis();
                if (checkValidNode(source) && checkValidNode(target)) {
                    PathResult result = parallelComp ?
                            findShortestPrimePathParallel(source, target) :
                            findShortestPrimePath(source, target);
                    if (result != null) {
                        System.out.println("shortest prime path: " + String.join(" -> ", result.getPath()) +
                                " with weight/length=" + result.getWeight());
                    } else {
                        System.out.println("No prime path from " + source + " to " + target);
                    }
                } else {
                    System.out.println("Invalid nodes.");
                }
                long end = System.currentTimeMillis();
                System.out.println("Search took " + (end - start) + " ms.");
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
            String[] parts = parseInput(data, 4, "wrong parse");
            String nodeSource = parts[1];
            String nodeTarget = parts[2];
            int weight = Integer.parseInt(parts[3]);
            adjList.computeIfAbsent(nodeSource, k -> new ArrayList<>()).add(new Edge(nodeTarget, weight));
            directedGraph.addEdge(nodeSource, nodeTarget);
        }
    }

    private static void getEdges(){

        System.out.print("Edges: ");
        adjList.forEach((node, edges) -> {
            edges.forEach(edge -> {
                System.out.print("(" + node + ", " + edge.target + ", Weight: " + edge.weight + ")");
                System.out.print(", ");
            });

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
            int totalWeight = dfsFindPath(nodeSource, nodeTarget, visited, path, 0);
            // -1 is a sentinel value indicating no path found
            if (totalWeight != -1) {
                return String.join(" -> ", path) + " with weight/length = " + totalWeight; // Construct the path as a string
            } else {
                return "No path found from " + nodeSource + " to " + nodeTarget;
            }
        }
        else{
            // Multithreaded DFS
            List<String> path = new ArrayList<>();
            AtomicInteger totalWeight = new AtomicInteger(0);
            if (parallelDfsFindPath(nodeSource, nodeTarget, path, totalWeight)) {
                return String.join(" -> ", path) + " with weight/length = " + totalWeight.get();
            } else {
                return "No path found from " + nodeSource + " to " + nodeTarget;
            }

        }
    }
    private static boolean parallelDfsFindPath(String nodeSource, String nodeTarget, List<String> path, AtomicInteger totalWeight) {
        ForkJoinPool pool = new ForkJoinPool(6);
        AtomicBoolean pathFound = new AtomicBoolean(false);
        CopyOnWriteArrayList<String> concurrentPath = new CopyOnWriteArrayList<>(); // Concurrent list to store the path


        try {
            pool.invoke(new ParallelDFS(nodeSource, nodeTarget, new ArrayList<>(), new HashSet<>(), pathFound, concurrentPath, 0, totalWeight));
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
        private final int currentWeight;
        private final AtomicInteger totalWeight;

        public ParallelDFS(String currentNode, String targetNode, List<String> currentPath,
                           Set<String> localVisited, AtomicBoolean pathFound, CopyOnWriteArrayList<String> concurrentPath,
                           int currentWeight, AtomicInteger totalWeight) {
            this.currentNode = currentNode;
            this.targetNode = targetNode;
            this.currentPath = new ArrayList<>(currentPath);
            this.localVisited = new HashSet<>(localVisited);
            this.pathFound = pathFound;
            this.concurrentPath = concurrentPath;
            this.currentWeight = currentWeight;
            this.totalWeight = totalWeight;
        }

        @Override
        protected Boolean compute() {
            if (pathFound.get()) return false;

            System.out.println("Thread " + Thread.currentThread().getName() + " is processing node: " + currentNode);

            localVisited.add(currentNode);
            currentPath.add(currentNode);

            if (currentNode.equals(targetNode)) {
                if (pathFound.compareAndSet(false, true)) {
                    concurrentPath.clear();
                    concurrentPath.addAll(currentPath);
                    totalWeight.set(currentWeight);
                    return true;
                }
                return false;
            }


                // Walk through all neighbors of the current node and set each as a subtask
                List<ParallelDFS> subTasks = new ArrayList<>();
                for (Edge edge : adjList.getOrDefault(currentNode, new ArrayList<>())) {
//                String neighborNode = edge.target; // Extract the neighbor node
//                if (!localVisited.contains(neighborNode) && !pathFound.get()) {
//                    ParallelDFS subTask = new ParallelDFS(neighborNode, targetNode, currentPath,
//                            localVisited, pathFound, concurrentPath);
//                    subTask.fork(); // Fork a new task
//                    subTasks.add(subTask); // Add the subtask to be processed later
//                }
                    if (!localVisited.contains(edge.target) && !pathFound.get()) {
                        subTasks.add(new ParallelDFS(edge.target, targetNode, currentPath, localVisited, pathFound, concurrentPath, currentWeight + edge.weight, totalWeight));
                        subTasks.get(subTasks.size() - 1).fork();
                    }

                }


                boolean found = false;
                for (ParallelDFS task : subTasks) {
                    found = task.join() || found;
                }

                if (!found) {
                    currentPath.remove(currentPath.size() - 1);
                    localVisited.remove(currentNode);
                }

                return found;
            }
    }

    private static int dfsFindPath(String current, String target, Set<String> visited, List<String> path, int currentWeight) {
        path.add(current);  // Add the current node to the path
        visited.add(current);  // Mark the current node as visited

        if (current.equals(target)) {  // Base case: if current node is the target node
            return currentWeight;
        }

        // Walk through all neighbors of the current node
        for (Edge edge : adjList.getOrDefault(current, new ArrayList<>())) {
            String neighborNode = edge.target; // Extract the neighbor node
            int edgeWeight = edge.weight; // Extract the edge weight
            if (!visited.contains(neighborNode)) {
                int weight = dfsFindPath(neighborNode, target, visited, path, currentWeight + edgeWeight);
                if (weight != -1) {// Valid path found
                    return weight;
                }
            }
        }


        // If no valid path found, backtrack
        path.remove(path.size() - 1);
        return -1;
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


    private static boolean checkValidEdge(String nodeSource, String nodeTarget) {
        List<Edge> edges = adjList.getOrDefault(nodeSource, new ArrayList<>());

        if (!parallelComp) {
            // Linear search through edges
            return edges.stream().anyMatch(edge -> edge.target.equals(nodeTarget));
        } else {
            // Split edges into two halves for parallel search
            int mid = edges.size() / 2;
            List<Edge> firstHalf = edges.subList(0, mid);
            List<Edge> secondHalf = edges.subList(mid, edges.size());

            AtomicBoolean edgeFound = new AtomicBoolean(false);

            // Create threads to search edge lists
            Thread firstThread = new Thread(() -> checkValidEdgeThread(nodeTarget, edgeFound, firstHalf));
            Thread secondThread = new Thread(() -> checkValidEdgeThread(nodeTarget, edgeFound, secondHalf));

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

    private static void checkValidEdgeThread(String nodeTarget, AtomicBoolean edgeFound, List<Edge> edgesToCheck) {
        for (Edge edge : edgesToCheck) {
            if (edgeFound.get()) return;
            if (edge.target.equals(nodeTarget)) {
                edgeFound.set(true);
            }
        }
    }

    // Check if a number is prime
    public class PrimeUtils {
        public static boolean isPrime(int n) {
            if (n <= 1) return false;
            if (n == 2) return true;
            if (n % 2 == 0) return false;
            for (int i = 3; i <= Math.sqrt(n); i += 2) {
                if (n % i == 0) return false;
            }
            return true;
        }
    }

    // Class to hold the path and its total weight
    public static class PathResult {
        private List<String> path;
        private int weight;

        public PathResult(List<String> path, int weight) {
            this.path = path;
            this.weight = weight;
        }

        public List<String> getPath() {
            return path;
        }

        public int getWeight() {
            return weight;
        }
    }

    // Prime-Path Sequential DFS
    private static PathResult getPrimePath(String nodeSource, String nodeTarget) {
        List<String> path = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        AtomicBoolean found = new AtomicBoolean(false);
        int weight = dfsFindPrimePath(nodeSource, nodeTarget, visited, path, 0, found);
        return found.get() ? new PathResult(path, weight) : null;
    }

    private static int dfsFindPrimePath(String current, String target, Set<String> visited, List<String> path, int currentWeight, AtomicBoolean found) {
        path.add(current);
        visited.add(current);

        if (current.equals(target)) {
            if (PrimeUtils.isPrime(currentWeight)) {
                found.set(true);
                return currentWeight; // Return the valid weight
            }
            path.remove(path.size() - 1);
            visited.remove(current);
            return -1; // Indicate no prime path
        }

        for (Edge neighbor : adjList.getOrDefault(current, new ArrayList<>())) {
            String neighborNode = neighbor.target;
            int edgeWeight = neighbor.weight;
            if (!visited.contains(neighborNode) && !found.get()) {
                int resultWeight = dfsFindPrimePath(neighborNode, target, visited, path, currentWeight + edgeWeight, found);
                if (resultWeight != -1) {
                    return resultWeight; // Propagate the valid weight up
                }
            }
        }

        path.remove(path.size() - 1);
        visited.remove(current);
        return -1;
    }

    // Prime-Path Parallel DFS
    private static PathResult getPrimePathParallel(String nodeSource, String nodeTarget) {
        ForkJoinPool pool = new ForkJoinPool(6);
        AtomicBoolean pathFound = new AtomicBoolean(false);
        CopyOnWriteArrayList<String> concurrentPath = new CopyOnWriteArrayList<>();
        AtomicInteger primeWeight = new AtomicInteger(-1);

        try {
            pool.invoke(new ParallelPrimeDFS(nodeSource, nodeTarget, new ArrayList<>(), new HashSet<>(), pathFound, concurrentPath, 0, primeWeight));
        } finally {
            pool.shutdown();
        }

        if (pathFound.get()) {
            return new PathResult(new ArrayList<>(concurrentPath), primeWeight.get());
        } else {
            return null;
        }
    }

    static class ParallelPrimeDFS extends RecursiveTask<Boolean> {
        private final String currentNode;
        private final String targetNode;
        private final List<String> currentPath;
        private final Set<String> localVisited;
        private final AtomicBoolean pathFound;
        private final CopyOnWriteArrayList<String> concurrentPath;
        private final int currentWeight;
        private final AtomicInteger primeWeight;

        public ParallelPrimeDFS(String currentNode, String targetNode, List<String> currentPath,
                                Set<String> localVisited, AtomicBoolean pathFound,
                                CopyOnWriteArrayList<String> concurrentPath, int currentWeight,
                                AtomicInteger primeWeight) {
            this.currentNode = currentNode;
            this.targetNode = targetNode;
            this.currentPath = new ArrayList<>(currentPath);
            this.localVisited = new HashSet<>(localVisited);
            this.pathFound = pathFound;
            this.concurrentPath = concurrentPath;
            this.currentWeight = currentWeight;
            this.primeWeight = primeWeight;
        }

        @Override
        protected Boolean compute() {
            if (pathFound.get()) return false;

            localVisited.add(currentNode);
            currentPath.add(currentNode);

            if (currentNode.equals(targetNode)) {
                if (PrimeUtils.isPrime(currentWeight)) {
                    if (pathFound.compareAndSet(false, true)) {
                        concurrentPath.clear();
                        concurrentPath.addAll(currentPath);
                        primeWeight.set(currentWeight);
                        return true;
                    }
                }
                currentPath.remove(currentPath.size() - 1);
                localVisited.remove(currentNode);
                return false;
            }

            List<ParallelPrimeDFS> subTasks = new ArrayList<>();
            for (Edge edge : adjList.getOrDefault(currentNode, new ArrayList<>())) {
                String neighborNode = edge.target;
                int edgeWeight = edge.weight;
                if (!localVisited.contains(neighborNode) && !pathFound.get()) {
                    ParallelPrimeDFS subTask = new ParallelPrimeDFS(
                            neighborNode, targetNode, currentPath,
                            localVisited, pathFound, concurrentPath,
                            currentWeight + edgeWeight, primeWeight
                    );
                    subTask.fork();
                    subTasks.add(subTask);
                }
            }

            boolean found = false;
            for (ParallelPrimeDFS task : subTasks) {
                found = task.join() || found;
            }

            currentPath.remove(currentPath.size() - 1);
            localVisited.remove(currentNode);
            return found;
        }
    }

    // Shortest Prime Path (Sequential) - Dijkstra's algorithm
    private static PathResult findShortestPrimePath(String nodeSource, String nodeTarget) {
        PriorityQueue<PathState> queue = new PriorityQueue<>(Comparator.comparingInt(ps -> ps.weight));
        queue.add(new PathState(nodeSource, 0, new ArrayList<>(Collections.singletonList(nodeSource))));

        int shortestPrimeWeight = Integer.MAX_VALUE;
        PathResult result = null;

        while (!queue.isEmpty()) {
            PathState current = queue.poll();

            // Skip paths heavier than the shortest prime found
            if (current.weight >= shortestPrimeWeight) {
                continue;
            }

            if (current.node.equals(nodeTarget)) {
                if (PrimeUtils.isPrime(current.weight)) {
                    if (current.weight < shortestPrimeWeight) {
                        shortestPrimeWeight = current.weight;
                        result = new PathResult(current.path, current.weight);
                    }
                }
                continue;
            }

            // Explore neighbors
            for (Edge neighbor : adjList.getOrDefault(current.node, new ArrayList<>())) {
                int newWeight = current.weight + neighbor.weight;
                List<String> newPath = new ArrayList<>(current.path);
                newPath.add(neighbor.target);
                queue.add(new PathState(neighbor.target, newWeight, newPath));
            }
        }
        return result;
    }

    static class PathState implements Comparable<PathState>{
        String node;
        int weight;
        List<String> path;

        public PathState(String node, int weight, List<String> path) {
            this.node = node;
            this.weight = weight;
            this.path = path;
        }

        @Override
        public int compareTo(PathState other) {
            return Integer.compare(this.weight, other.weight);
        }
    }

    // Shortest Prime Path (Parallel)
    private static PathResult findShortestPrimePathParallel(String source, String target) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        PriorityBlockingQueue<PathState> queue = new PriorityBlockingQueue<>();
        AtomicInteger shortestPrimeWeight = new AtomicInteger(Integer.MAX_VALUE);
        AtomicReference<PathResult> bestResult = new AtomicReference<>();

        // Initialize with the source node
        queue.add(new PathState(source, 0, new ArrayList<>(Collections.singletonList(source))));

        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
                futures.add(executor.submit(() -> {
                    while (!queue.isEmpty()) {
                        PathState current = queue.poll();
                        if (current == null) continue;

                        // Skip paths heavier than the current shortest prime
                        if (current.weight >= shortestPrimeWeight.get()) {
                            continue;
                        }

                        // Check if this path reaches the target and is prime
                        if (current.node.equals(target)) {
                            if (PrimeUtils.isPrime(current.weight)) {
                                // Update if this is the shortest prime found so far
                                shortestPrimeWeight.getAndUpdate(curr -> {
                                    if (current.weight < curr) {
                                        bestResult.set(new PathResult(current.path, current.weight));
                                        return current.weight;
                                    }
                                    return curr;
                                });
                            }
                        }

                        // Explore neighbors
                        for (Edge neighbor : adjList.getOrDefault(current.node, new ArrayList<>())) {
                            int newWeight = current.weight + neighbor.weight;
                            List<String> newPath = new ArrayList<>(current.path);
                            newPath.add(neighbor.target);

                            // Add to queue if potentially shorter
                            if (newWeight < shortestPrimeWeight.get()) {
                                queue.add(new PathState(neighbor.target, newWeight, newPath));
                            }
                        }
                    }
                }));
            }

            // Wait for all threads to finish
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
        } finally {
            executor.shutdown();
        }

        return bestResult.get();
    }

    // Shortest Path (Sequential) - Dijkstra's algorithm
    private static PathResult findShortestPath(String nodeSource, String nodeTarget) {
        PriorityQueue<PathState> queue = new PriorityQueue<>(Comparator.comparingInt(ps -> ps.weight));
        queue.add(new PathState(nodeSource, 0, new ArrayList<>(Collections.singletonList(nodeSource))));

        int shortestWeight = Integer.MAX_VALUE;
        PathResult result = null;

        while (!queue.isEmpty()) {
            PathState current = queue.poll();

            // Skip paths heavier than the shortest found
            if (current.weight >= shortestWeight) {
                continue;
            }

            if (current.node.equals(nodeTarget)) {
                if (current.weight < shortestWeight) {
                    shortestWeight = current.weight;
                    result = new PathResult(current.path, current.weight);
                }
                continue;
            }

            // Explore neighbors
            for (Edge neighbor : adjList.getOrDefault(current.node, new ArrayList<>())) {
                int newWeight = current.weight + neighbor.weight;
                List<String> newPath = new ArrayList<>(current.path);
                newPath.add(neighbor.target);
                queue.add(new PathState(neighbor.target, newWeight, newPath));
            }
        }
        return result;
    }

    // Shortest Path (Parallel)
    private static PathResult findShortestPathParallel(String source, String target) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        PriorityBlockingQueue<PathState> queue = new PriorityBlockingQueue<>();
        AtomicInteger shortestWeight = new AtomicInteger(Integer.MAX_VALUE);
        AtomicReference<PathResult> bestResult = new AtomicReference<>();

        // Initialize with the source node
        queue.add(new PathState(source, 0, new ArrayList<>(Collections.singletonList(source))));

        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
                futures.add(executor.submit(() -> {
                    while (!queue.isEmpty()) {
                        PathState current = queue.poll();
                        if (current == null) continue;

                        // Skip paths heavier than the current shortest
                        if (current.weight >= shortestWeight.get()) {
                            continue;
                        }

                        // Check if this path reaches the target
                        if (current.node.equals(target)) {
                            // Update if this is the shortest found so far
                            shortestWeight.getAndUpdate(curr -> {
                                if (current.weight < curr) {
                                    bestResult.set(new PathResult(current.path, current.weight));
                                    return current.weight;
                                }
                                return curr;
                            });
                        }

                        // Explore neighbors
                        for (Edge neighbor : adjList.getOrDefault(current.node, new ArrayList<>())) {
                            int newWeight = current.weight + neighbor.weight;
                            List<String> newPath = new ArrayList<>(current.path);
                            newPath.add(neighbor.target);

                            // Add to queue if potentially shorter
                            if (newWeight < shortestWeight.get()) {
                                queue.add(new PathState(neighbor.target, newWeight, newPath));
                            }
                        }
                    }
                }));
            }

            // Wait for all threads to finish
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
        } finally {
            executor.shutdown();
        }

        return bestResult.get();
    }
}