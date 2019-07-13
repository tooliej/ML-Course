package HW3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

enum LpDistance{One, Two, Three, Infinity}
enum DistanceCheck{Regular, Efficient}
enum WeightingScheme{Uniform, Weighted}

class DistanceCalculator {

    private LpDistance lpDistance;
    private DistanceCheck distanceCheck;


    public DistanceCalculator(LpDistance distance, DistanceCheck check){
        this.lpDistance = distance;
        this.distanceCheck = check;
    }

        /**
        * We leave it up to you whether you want the distance method to get all relevant
        * parameters(lp, efficient, etc..) or have it has a class variables.
        */
    public double distance (Instance one, Instance two, double maxDistance) {
        double distance = 0.0;
        switch (lpDistance){
            case One:
                if(distanceCheck == DistanceCheck.Efficient)
                    distance = efficientLpDistance(one, two, 1, maxDistance);
                else
                    distance = lpDistance(one, two, 1);
                break;
            case Two:
                if(distanceCheck == DistanceCheck.Efficient)
                    distance = efficientLpDistance(one, two, 2, maxDistance);
                else
                    distance = lpDistance(one, two, 2);
                break;
            case Three:
                if(distanceCheck == DistanceCheck.Efficient)
                    distance = efficientLpDistance(one, two, 3, maxDistance);
                else
                 distance = lpDistance(one, two, 3);
                break;
            case Infinity:
                if(distanceCheck == DistanceCheck.Efficient)
                    distance = efficientLInfinityDistance(one, two, maxDistance);
                else
                    distance = lInfinityDistance(one, two);
                break;
        }
        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two, double lp) {
        double sum = 0;
        for(int i = 0; i < one.numAttributes()-1; i++){
            sum += Math.pow(Math.abs(one.value(i)-two.value(i)), lp);
        }

        return Math.pow(sum, 1/lp);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double maxDistance = 0;
        double currentDistance;
        for(int i = 0; i < one.numAttributes()-1; i++){
            currentDistance = Math.abs(one.value(i)-two.value(i));
            if(currentDistance > maxDistance)
                maxDistance = currentDistance;
        }
        return maxDistance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two, double lp, double maxDistance) {
        double sum = 0;
        double max = Math.pow(maxDistance, lp);
        for(int i = 0; i < one.numAttributes()-1; i++){
            if(sum > max)
                return Double.POSITIVE_INFINITY;
            sum += Math.pow(Math.abs(one.value(i)-two.value(i)), lp);
        }

        return Math.pow(sum, 1/lp);
    }



    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double maxDistance) {
        double currentDistance = 0;
        double currentMax = 0;

        for(int i = 0; i < one.numAttributes()-1; i++){
            //checks if the current Instance's distance is larger than the maximum k's distance
            if(currentMax > maxDistance)
                return Double.POSITIVE_INFINITY;

            currentDistance = Math.abs(one.value(i)-two.value(i));

            if(currentDistance > currentMax)
                currentMax = currentDistance;
        }
        return currentMax;
    }

}

public class Knn implements Classifier {

    private int kNeighbors;
    private LpDistance lpDistance;
    private WeightingScheme weightingScheme;
    private Instances m_trainingInstances;
    private DistanceCheck distanceCheck;
    private long totalRunTime;


    public Knn(int kNeighbors, LpDistance lpDistance, Instances data, WeightingScheme weight, DistanceCheck distanceCheck) {
        this.kNeighbors = kNeighbors;
        this.lpDistance = lpDistance;
        m_trainingInstances = data;
        this.weightingScheme = weight;
        this.distanceCheck = distanceCheck;
    }

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        double prediction = 0;
        //finds the k nearest neighbors from training set for this instance
        Instances kNeighbors = findNearestNeighbors(instance);
        switch (weightingScheme) {
            case Uniform:
                prediction = getAverageValue(kNeighbors);
                break;
            case Weighted:
                prediction = getWeightedAverageValue(kNeighbors, instance);
                break;
        }

        return prediction;
    }

    /**
     * Caclculates the average error on a given set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        double sumOfErrors = 0;
        double currTargetValue = 0;
        double currPrediction = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            currTargetValue = instances.instance(i).classValue();
            currPrediction = regressionPrediction(instances.instance(i));
            sumOfErrors += Math.abs(currTargetValue - currPrediction);
        }
        return sumOfErrors / instances.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances    Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) {
        long totalTime = 0;
        double error = 0;

        //calculate the average error for fold i
        for (int i = 0; i < num_of_folds; i++) {
            Instances testInstances = instances.testCV(num_of_folds, i);
            m_trainingInstances = instances.trainCV(num_of_folds, i);
            long startTime = System.nanoTime();
            error += calcAvgError(testInstances);
            totalTime += (System.nanoTime() - startTime);
        }
        this.totalRunTime = totalTime;
        return error / num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */

    public Instances findNearestNeighbors(Instance instance) {

        Pair[] neighbors = new Pair[kNeighbors];
        Pair currentPair;

        //finds k nearest neighbors
        for (int i = 0; i < kNeighbors; i++) {
            Instance currentInstance = m_trainingInstances.instance(i);
            if (currentInstance != instance)
                currentPair = new Pair(i, new DistanceCalculator
                        (lpDistance, DistanceCheck.Regular).distance(instance, currentInstance, 0));
            else
                currentPair = new Pair(i, Double.POSITIVE_INFINITY);

            neighbors[i] = currentPair;
        }

        int maxKindex = findMaxKIndex(neighbors); //index of k'th nearest neighbor in neighbor array
        double maxK = neighbors[maxKindex].distance;

        //iterates through the rest of the instances to find the k nearest neighbors.
        for (int i = kNeighbors; i < m_trainingInstances.numInstances(); i++) {
            Instance currentInstance = m_trainingInstances.instance(i);
            if (currentInstance != instance)
                currentPair = new Pair(i, new DistanceCalculator(lpDistance, this.distanceCheck).distance(instance,
                        currentInstance, maxK));
            else
                currentPair = new Pair(i, Double.POSITIVE_INFINITY);
            //checks if the current instance is a new nearest neighbor. if YES adds it to the neighbor array and sorts.
            if (currentPair.distance < maxK) {
                neighbors[maxKindex] = currentPair;
                maxKindex = findMaxKIndex(neighbors);
                maxK = neighbors[maxKindex].distance;
            }
        }
        //returns the Instances from the neighbor array
        Instances kNeighbors = new Instances(m_trainingInstances, 0, 0);
        for (int i = 0; i < this.kNeighbors; i++) {
            int index = neighbors[i].instance;
            kNeighbors.add(m_trainingInstances.instance(index));
        }
        return kNeighbors;
    }


    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param
     * @return
     */
    public double getAverageValue(Instances instances) {
        double value = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            value += instances.instance(i).classValue();
        }
        return value / instances.numInstances();
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(Instances instances, Instance instance) {
        double sumOfValues = 0;
        double distanceFromInstance = 0;
        double sumOfDistances = 0;
        double distance = 0;
        for (int i = 0; i < instances.numInstances(); i++) {
            distance = new DistanceCalculator(lpDistance, DistanceCheck.Regular).distance(instance,
                    instances.instance(i), 0);
            if (distance != 0)
                distanceFromInstance = 1 / Math.pow(distance, 2);
            else
                return instances.instance(i).classValue();
            sumOfValues += distanceFromInstance * instances.instance(i).classValue();
            sumOfDistances += distanceFromInstance;
        }
        return sumOfValues / sumOfDistances;
    }

    public int findMaxKIndex(Pair[] pair) {
        double maxK = 0;
        int index = 0;
        for (int i = 0; i < pair.length; i++) {
            if (pair[i].distance > maxK) {
                maxK = pair[i].distance;
                index = i;
            }
        }
        return index;

    }
    public long getTotalTime() {
        return this.totalRunTime;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }

    public class Pair implements Comparable<Pair> {
        public int instance;
        public double distance;

        public Pair(int index, double value) {
            this.instance = index;
            this.distance = value;
        }

        @Override
        public int compareTo(Pair other) {
            return Double.valueOf(this.distance).compareTo(other.distance);

        }

    }
}

