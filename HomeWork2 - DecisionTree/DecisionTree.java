package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.*;
import java.util.*;

enum ImpurityType {
    Entropy, Gini
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex = -1;
	double returnValue;
}

public class DecisionTree implements Classifier {
	private Node rootNode;
    private ImpurityType impurityMeasure;
    private double pValue;
    private int totalHeight; // The sum of the heights of classification nodes
    private int maxHeight;
    private int numberOfClassifications;

    public DecisionTree(ImpurityType impurityMeasure){
        this.impurityMeasure = impurityMeasure;
        this.pValue = 1;
    }

    public DecisionTree(ImpurityType impurityMeasure, double pValue){
        this.impurityMeasure = impurityMeasure;
        this.pValue = pValue;
    }

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
        rootNode = new Node();

        buildTree(arg0, rootNode, impurityMeasure);
	}
    
    @Override
	public double classifyInstance(Instance instance) {
	    Node currentNode = rootNode;
	    int currentHeight = 0;

	    while(currentNode.children != null &&
                currentNode.children[(int) instance.value(currentNode.attributeIndex)] != null) {
            currentNode = currentNode.children[(int) instance.value(currentNode.attributeIndex)];
            currentHeight++;
        }

        // Updating tree height data
        numberOfClassifications++;
        totalHeight += currentHeight;
	    if(currentHeight > maxHeight)
	        maxHeight = currentHeight;

        return currentNode.returnValue;
    }
    
    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

    /**
     *
     * @return - The height of the deepest classified node in the tree
     */
    public int getMaxHeight(){
        return maxHeight;
    }

    /**
     *
     * @return - The average height of the tree
     */
    public double getAverageHeight(){
        return totalHeight / (double) numberOfClassifications;
    }

    /**
     * Builds the decision tree on given data
     * @param data
     */
	private void buildTree(Instances data, Node parentNode, ImpurityType type) throws Exception{
	    double[] classifications = getClassifications(data);

	    // Sets a return value to the node
	    parentNode.returnValue = getReturnValue(classifications);

	    // If there is perfect classification, finish building the tree
	    if(classifications[0] == 0 || classifications [1] == 0)
            return;

	    // Checks to see if all the attributes are the same apart from the classification
        if(sameAttributeValues(data))
            return;

        // Chooses this attribute with which to perform the switch
	    parentNode.attributeIndex = findAttributeIndex(data, getPercentages(classifications), type);

        // Splits the data according to the chosen attribute
        Instances[] splitData = split(data, parentNode.attributeIndex);

        // If the tree does not need to be pruned, continues building the tree
        if(pValue == 1 || !prune(classifications, splitData)) {
            parentNode.children = new Node[splitData.length];
            for(int i = 0; i < splitData.length; i++) {
                if (splitData[i] != null) {
                    Node childNode = createChildNode(parentNode, i);

                    // Recursively continues to build the tree with split data
                    buildTree(splitData[i], childNode, type);
                }
            }
        }
    }

    /**
     * Returns an array of size 2 with the number of instances classified 0.0 and the number of instances
     * classified 1 in the dataset.
     * @param data
     * @return
     */
    private double[] getClassifications(Instances data) {
        double[] classifications = new double[2];

        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).classValue() == 0.0)
                classifications[0]++;
            else
                classifications[1]++;
        }

        return classifications;
    }

    /**
     * Recieves an array of size 2 with the number of instances classified 0.0 and the number of instances
     * classified 1. Returns an array of size 2 with the percent of classifications of 0.0 and the percent of
     * classifications 1.0
     * @param classifications
     * @return
     */
    private double[] getPercentages(double [] classifications){
        double[] percentages = new double[2];
        double size = classifications[0] + classifications[1];

        for(int i = 0; i < percentages.length; i++) {
            percentages[i] = classifications[i] / size;
        }

        return percentages;
    }

    /**
     * Returns the return value of a given node based on the number of classifications
     * @param i_classifications
     * @return
     */
    private double getReturnValue(double[] i_classifications) {
	    double classify = 0.0;

	    if(i_classifications[0] < i_classifications[1])
	        classify = 1.0;

	    return classify;
    }

    /**
     * Finds the attribute best suited for the split. Returns the attributes index
     * @param data
     * @return
     */
    private int findAttributeIndex(Instances data, double[] o_classifications, ImpurityType o_type) {
        int attributeIndex = 0;

        double presentGain;
        double maxGain = Double.NEGATIVE_INFINITY;
        double impurity;

        // Checks how to calculate the impurity, either Gini or Entropy
        if(o_type.equals(ImpurityType.Entropy))
            impurity = calcEntropy(o_classifications);
        else
            impurity = calcGini(o_classifications);

        for(int i = 0; i < data.numAttributes() - 1; i++) {

            double[][] probabilityMatrix = new double[data.attribute(i).numValues()][2];
                for (int j = 0; j < data.numInstances(); j++) {
                    String attribute = data.instance(j).toString(i);
                    int indexOfValue = data.instance(j).attribute(i).indexOfValue(attribute);

                    if(data.instance(j).classValue() == 0)
                        probabilityMatrix[indexOfValue][0]++;
                    else
                        probabilityMatrix[indexOfValue][1]++;
                }

                presentGain = calcGain(data.numInstances(), probabilityMatrix, impurity, o_type);
                if(presentGain > maxGain) {
                    maxGain = presentGain;
                    attributeIndex = i;
                }
            }

	    return attributeIndex;
    }

    /**
     * Calculates the average error on a given instances set.
     * The average error is the total number of classification mistakes on the input instances set divided by
     * the number of instances in the input set.
     * @param data - Could be the training, test or validation set
     * @return
     */
    public double calcAvgError(Instances data) {
        maxHeight = 0;
        numberOfClassifications = 0;
        totalHeight = 0;
        int incorrectClassification = 0;

        for(int i = 0; i < data.numInstances(); i++) {
             if(classifyInstance(data.instance(i)) != data.instance(i).classValue())
                 incorrectClassification++;
        }

	    return incorrectClassification / (double) data.numInstances();
    }

    /**
     * Calculates the gain (giniGain or informationGain depending on the impurity measure)
     * of splitting the input data according to the attribute.
     * @param i_probabilityMatrix
     * @return - The gain
     */
    private double calcGain(int i_numInstances, double[][] i_probabilityMatrix, double o_impurity, ImpurityType i_type) {
        double gain = 0;

        for(int i = 0; i < i_probabilityMatrix.length; i++) {
            double size = i_probabilityMatrix[i][0] + i_probabilityMatrix[i][1];
            if(size != 0) {
                i_probabilityMatrix[i][0] /= size;
                i_probabilityMatrix[i][1] /= size;

                if (i_type.equals(ImpurityType.Entropy))
                    gain += (size / i_numInstances) * calcEntropy(i_probabilityMatrix[i]);
                else
                    gain += (size / i_numInstances) * calcGini(i_probabilityMatrix[i]);
            }
        }

        return o_impurity - gain;
    }

    /**
     * Calculates the Entropy of a random variable.
     * @param classifications - array of size two that holds the percent instances in the dataset with classification
     * 0.0 and the percent with classification 1.0
     * @return entropy
     */
    private double calcEntropy(double[] classifications) {
        double entropy = 0;
        for(int i = 0; i < classifications.length; i++) {
            if (classifications[i] > 0)
                entropy -= classifications[i] * (Math.log(classifications[i]) / Math.log(2));
        }

        return entropy;
    }

    /**
     * Calculates the Gini of a random variable
     * @param  classifications - array of size two that holds the percent instances in the dataset with classification
     * 0.0 and the percent with classification 1.0
     * @return
     */
    private double calcGini(double[] classifications) {
        double gini = 1 - ((classifications[0] * classifications[0]) +
                classifications[1] * classifications[1]);

        return gini;
    }

    /**
     * Checks if all the attribute values in the data are the same. In this case if classifications
     * are different a split cannot be made
     * @param data
     * @return
     */
    private boolean sameAttributeValues(Instances data) {
        boolean differentValues = true;
        for(int i = 0; i < data.numAttributes() - 1; i++) {

            // Finds the value of the first instances attribute to be compared with the rest of the instances
            int firstInstanceAttribute = (int) data.instance(0).value(i);
            for(int j = 1; j < data.numInstances(); j++) {
                if(firstInstanceAttribute != (int) data.instance(j).value(i)) {
                        differentValues = false;
                        break;
                }
            }
        }

        return differentValues;
    }

    /**
     * Split the instances according to the attribute value of a certain attribute
     * @param data
     * @param i_attributeIndex - The index of the attribute that determines the split
     * @return
     */
    private Instances[] split(Instances data, int i_attributeIndex) {
        int numOfValues = data.attribute(i_attributeIndex).numValues();
        Instances[] splitInstances = new Instances[numOfValues];

        for(int i = 0; i < data.numInstances(); i++) {

            // The index of the value of the current attribute
            int index = (int) data.instance(i).value(i_attributeIndex);
            if(splitInstances[index] == null) {
                splitInstances[index] = new Instances(data, 0, 0);
            }

            splitInstances[index].add(data.instance(i));
        }

        return splitInstances;
    }


    /**
     * Creates child node and add the node to the tree
     * @param parentNode
     * @param index
     */
    private Node createChildNode(Node parentNode, int index){
        Node childNode = new Node();
        parentNode.children[index] = childNode;
        childNode.parent = parentNode;

        return childNode;
    }

    /**
     * Calculates the chi Square value and checks in the chi square chart if with current p_value
     * the tree should be pruned
     * @param parentClassification
     * @param splitData
     * @return True if the tree is to be pruned
     */
    private boolean prune(double[] parentClassification, Instances[] splitData) {

        // Initialization of Chi Chart
        double[][] chiChart = {
            { 0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438 }, // p_value 0.75
            { 0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340 }, // p_value 0.5
            { 1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845 }, // p_value 0.25
            { 3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026 }, // p_value 0.05
            { 7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300 } // p_value 0.005
        };

        //Checks the Chi Square value for pruning
        double chiSquare = calcChiSquare(parentClassification, splitData);
        int row = getChiChartRow();
        int df = 0;

        // Calculating the degree of freedom.
        // The degree of freedom is relevant number of values of in the attribute minus 1
        for (int i = 0; i < splitData.length; i ++) {
            if(splitData[i] != null)
                df++;
        }

        // I subtract 1 from the degree of freedom to get the correct value of df.
        // Subtract another 1 from df to get the appropriate index in chiChart
        df-= 2;

        return chiSquare < chiChart[row][df];
    }

    /**
     *
     * @return - The index of the row in the Chi chart representing the p_value
     */
    private int getChiChartRow() {
        int index = 0;

        if(pValue == 0.75)
            index = 0;
        else if(pValue == 0.5)
            index = 1;
        else if(pValue == 0.25)
            index = 2;
        else if(pValue == 0.05)
            index = 3;
        else if(pValue == 0.005)
            index = 4;

        return index;
    }

    /**
     * Calculates the chi square statistic of splitting the data according to the splitting attribute as learned in class
     * @param parentClassification
     * @param splitData
     * @return - The chi square score
     */
    private double calcChiSquare(double[] parentClassification, Instances[] splitData) {
        double chiSquare = 0;

        for (int i = 0; i < splitData.length; i++) {
            if (splitData[i] != null) {
                double[] childClassifications = getClassifications(splitData[i]);
                double numberOfInstancesChild = childClassifications[0] + childClassifications[1];
                double numberOfInstancesParent = parentClassification[0] + parentClassification[1];
                double[] exp = new double[2];

                for(int j = 0; j < exp.length; j++) {
                    exp[j] = numberOfInstancesChild * (parentClassification[j] / numberOfInstancesParent);
                }

                for(int j = 0; j < exp.length; j++) {
                    chiSquare += Math.pow(childClassifications[j]  - exp[j], 2.0) / exp[j];
                }
            }
        }

        return chiSquare;
    }

    /**
     * Prints the tree to the console
     */
    public void printTree() {
        StringBuilder tree = new StringBuilder("Root\n");
        printTree(tree, rootNode, 1);
        System.out.println(tree);
    }

    private void printTree(StringBuilder tree, Node currentNode, int tabs){
        if(currentNode.children == null)
            tree.append(getTabs(tabs) + "Leaf. Returning value: " + currentNode.returnValue + "\n");
        else
            tree.append(getTabs(tabs - 1) + "Returning value: " + currentNode.returnValue + "\n");

        for(int i = 0; currentNode.children != null && i < currentNode.children.length; i++) {
            if(currentNode.children[i] != null) {
                tree.append(getTabs(tabs) + "If attribute " + currentNode.attributeIndex + " = " + i + "\n");
                printTree(tree, currentNode.children[i], tabs + 1);
            }
        }
    }

    // Adds the number of needed tabs to the string
    private static String getTabs(int numOfTabs) {
        StringBuilder tabs = new StringBuilder("");

        for(int i = 0; i < numOfTabs; i++){
            tabs.append("\t");
        }

        return tabs.toString();
    }
}





