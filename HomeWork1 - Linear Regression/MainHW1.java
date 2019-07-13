package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException{
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {

        //load data
        Instances trainingData = loadData("src/wind_training.txt");
        Instances testingData = loadData("src/wind_testing.txt");

        //find best alpha and build classifier with all attributes
        LinearRegression linearRegressionModel = new LinearRegression();
        linearRegressionModel.buildClassifier(trainingData);
        double bestAlpha = linearRegressionModel.getAlpha();
        System.out.println("The chosen alpha is: " + bestAlpha);
        System.out.println("Training error with all features is: " + linearRegressionModel.calculateMSE(trainingData));
        System.out.println("Test error with all features is: " + linearRegressionModel.calculateMSE(testingData));
        System.out.println("List of all combination of 3 features and the training error");

        //build classifiers with all 3 attributes combinations
        Remove remove = new Remove();
        LinearRegression bestThreeModel = new LinearRegression();
        Instances trainingThree = null;
        Instances testingThree = null;
        int[] bestThree = new int[4];
        int[] currentThree = new int[4];
        double currErr;
        double minErr = Double.POSITIVE_INFINITY;

        bestThreeModel.setAlpha(bestAlpha);
        bestThree[3] = trainingData.classIndex();
        currentThree[3] = trainingData.classIndex();

        // Iterates over every three attributes
        for(int i = 0; i < trainingData.numAttributes() - 1; i++) {
            for(int j = i + 1; j < trainingData.numAttributes() - 1; j++) {
                for(int k = j + 1; k < trainingData.numAttributes() - 1; k++ ) {
                    currentThree[0] = i;
                    currentThree[1] = j;
                    currentThree[2] = k;

                    // Set the three attributes in training data to be the current three attributes
                    remove.setInvertSelection(true);
                    remove.setAttributeIndicesArray(currentThree);
                    remove.setInputFormat(trainingData);
                    trainingThree = Filter.useFilter(trainingData, remove);
                    trainingThree.setClassIndex(trainingThree.numAttributes() - 1);

                    bestThreeModel.buildClassifier(trainingThree);
                    currErr = bestThreeModel.calculateMSE(trainingThree);
                    if(currErr < minErr) {
                        minErr = currErr;
                        bestThree[0] = i;
                        bestThree[1] = j;
                        bestThree[2] = k;
                    }

                    System.out.println(trainingThree.attribute(0).name() + " "
                            + trainingThree.attribute(1).name() + " " + trainingThree.attribute(2).name() + ": " + currErr);

                }
            }
        }

        remove.setInvertSelection(true);
        remove.setAttributeIndicesArray(bestThree);
        remove.setInputFormat(testingData);
        trainingThree = Filter.useFilter(trainingData, remove);
        testingThree = Filter.useFilter(testingData, remove);
        testingThree.setClassIndex(testingThree.numAttributes() - 1);

        // Using the best three attributes to train the data
        bestThreeModel.buildClassifier(trainingThree);

        System.out.println(
                "Training error the features " + trainingData.attribute(bestThree[0]).name() + " "
                + trainingData.attribute(bestThree[1]).name() + " " + trainingData.attribute(bestThree[2]).name() + ": " + minErr);
        System.out.println(
                "Test error the features " + testingData.attribute(bestThree[0]).name() + " "
                        + testingData.attribute(bestThree[1]).name() + " " +
                        testingData.attribute(bestThree[2]).name() + ": " + bestThreeModel.calculateMSE(testingThree));
    }
}
