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


public class Main {
    private static Map<String, Map<String, Integer>> adjList =  new HashMap<>();
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
            else if (input.startsWith("prime-path ")) {
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
            else if (input.startsWith("shortest-prime-path ")) {
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
            adjList.putIfAbsent(node, new HashMap<>());
            directedGraph.addVertex(node);
        }
        else if (data.startsWith("-")) {
            String[] parts = parseInput(data, 4, "wrong parse");
            String nodeSource = parts[1];
            String nodeTarget = parts[2];
            int weight = Integer.parseInt(parts[3]);
            adjList.get(nodeSource).put(nodeTarget, weight);
            directedGraph.addEdge(nodeSource, nodeTarget);
        }
    }

    private static void getEdges(){

        System.out.print("Edges: ");
        adjList.forEach((node, neighbors) -> {
            neighbors.forEach((neighbor, weight) -> {
                System.out.print("(" + node + ", " + neighbor + ", Weight: " + weight + ")");
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
            for (Map.Entry<String, Integer> neighbor : adjList.getOrDefault(currentNode, new HashMap<>()).entrySet()) {
                String neighborNode = neighbor.getKey(); // Extract the neighbor node
                if (!localVisited.contains(neighborNode) && !pathFound.get()) {
                    ParallelDFS subTask = new ParallelDFS(neighborNode, targetNode, currentPath,
                            localVisited, pathFound, concurrentPath);
                    subTask.fork(); // Fork a new task
                    subTasks.add(subTask); // Add the subtask to be processed later
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
        for (Map.Entry<String, Integer> neighbor : adjList.getOrDefault(current, new HashMap<>()).entrySet()) {
            String neighborNode = neighbor.getKey(); // Extract the neighbor node
            if (!visited.contains(neighborNode)) {
                if (dfsFindPath(neighborNode, target, visited, path)) {
                    return true; // Path found
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
        List<String> allNeighbors = new ArrayList<>(adjList.getOrDefault(nodeSource, new HashMap<>()).keySet());
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
        int[] weight = {0};
        boolean result = dfsFindPrimePath(nodeSource, nodeTarget, visited, path, 0, found);
        if (result) {
            return new PathResult(path, weight[0]);
        } else {
            return null;
        }
    }

    private static boolean dfsFindPrimePath(String current, String target, Set<String> visited, List<String> path, int currentWeight, AtomicBoolean found) {
        path.add(current);
        visited.add(current);

        if (current.equals(target)) {
            if (PrimeUtils.isPrime(currentWeight)) {
                found.set(true);
                return true;
            }
            path.remove(path.size() - 1);
            visited.remove(current);
            return false;
        }

        for (Map.Entry<String, Integer> neighbor : adjList.getOrDefault(current, new HashMap<>()).entrySet()) {
            String neighborNode = neighbor.getKey();
            int edgeWeight = neighbor.getValue();
            if (!visited.contains(neighborNode) && !found.get()) {
                if (dfsFindPrimePath(neighborNode, target, visited, path, currentWeight + edgeWeight, found)) {
                    return true;
                }
            }
        }

        path.remove(path.size() - 1);
        visited.remove(current);
        return false;
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
            for (Map.Entry<String, Integer> neighbor : adjList.getOrDefault(currentNode, new HashMap<>()).entrySet()) {
                String neighborNode = neighbor.getKey();
                int edgeWeight = neighbor.getValue();
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

        Map<String, Integer> visited = new HashMap<>();
        visited.put(nodeSource, 0);

        while (!queue.isEmpty()) {
            PathState current = queue.poll();
            String currentNode = current.node;
            int currentWeight = current.weight;
            List<String> currentPath = current.path;

            if (currentNode.equals(nodeTarget)) {
                if (PrimeUtils.isPrime(currentWeight)) {
                    return new PathResult(currentPath, currentWeight);
                }
                continue;
            }

            if (currentWeight > visited.get(currentNode)) continue;

            for (Map.Entry<String, Integer> neighbor : adjList.getOrDefault(currentNode, new HashMap<>()).entrySet()) {
                String neighborNode = neighbor.getKey();
                int edgeWeight = neighbor.getValue();
                int newWeight = currentWeight + edgeWeight;

                if (!visited.containsKey(neighborNode) || newWeight < visited.get(neighborNode)) {
                    visited.put(neighborNode, newWeight);
                    List<String> newPath = new ArrayList<>(currentPath);
                    newPath.add(neighborNode);
                    queue.add(new PathState(neighborNode, newWeight, newPath));
                }
            }
        }
        return null;
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
        AtomicBoolean found = new AtomicBoolean(false);

        // Initialize with the source node
        queue.add(new PathState(source, 0, new ArrayList<>(Collections.singletonList(source))));

        // Track visited nodes with their minimal weights
        ConcurrentHashMap<String, Integer> visited = new ConcurrentHashMap<>();

        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
                futures.add(executor.submit(() -> {
                    while (!found.get() && !queue.isEmpty()) {
                        PathState current = queue.poll();
                        if (current == null) continue;

                        // Early termination if a shorter prime path is already found
                        if (current.weight >= shortestPrimeWeight.get()) {
                            continue;
                        }

                        // Check if current path reaches the target and has prime weight
                        if (current.node.equals(target)) {
                            if (PrimeUtils.isPrime(current.weight)) {
                                // Atomically update the shortest prime weight
                                if (current.weight < shortestPrimeWeight.getAndSet(current.weight)) {
                                    found.set(true);
                                    return new PathResult(current.path, current.weight);
                                }
                            }
                        }

                        // Explore neighbors
                        for (Map.Entry<String, Integer> neighbor : adjList.getOrDefault(current.node, new HashMap<>()).entrySet()) {
                            String neighborNode = neighbor.getKey();
                            int newWeight = current.weight + neighbor.getValue();
                            List<String> newPath = new ArrayList<>(current.path);
                            newPath.add(neighborNode);

                            // Update visited only if this path is shorter
                            visited.compute(neighborNode, (k, v) -> {
                                if (v == null || newWeight < v) {
                                    queue.add(new PathState(neighborNode, newWeight, newPath));
                                    return newWeight;
                                }
                                return v;
                            });
                        }
                    }
                    return null;
                }));
            }

            // Wait for tasks to complete and check results
            for (Future<?> future : futures) {
                PathResult result = (PathResult) future.get();
                if (result != null) {
                    executor.shutdownNow(); // Terminate other threads
                    return result;
                }
            }
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
        } finally {
            executor.shutdown();
        }
        return null;
    }
}