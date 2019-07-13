package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;

public class MainHW2 {

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
		Instances trainingCancer = loadData("src/cancer_train.txt");
		Instances testingCancer = loadData("src/cancer_test.txt");
		Instances validationCancer = loadData("src/cancer_validation.txt");

        // Build two trees with training data one with Entropy as the value of impurity and the other with Gini
        DecisionTree entropyTree = new DecisionTree(ImpurityType.Entropy);
        entropyTree.buildClassifier(trainingCancer);

        DecisionTree giniTree = new DecisionTree(ImpurityType.Gini);
        giniTree.buildClassifier(trainingCancer);

        double entropyValErr = entropyTree.calcAvgError(validationCancer);
        double giniValErr = giniTree.calcAvgError(validationCancer);

        System.out.println("Validation error using Entropy: " + entropyValErr);
        System.out.println("Validation error using Gini: " + giniValErr);
        System.out.println("---------------------------------------------");

        // Chooses the impurity type for the rest of the checks,
        // The chosen type is the one with the lowest validation error
        ImpurityType chosenImpurity;
        if(entropyValErr < giniValErr)
            chosenImpurity = ImpurityType.Entropy;
        else
            chosenImpurity = ImpurityType.Gini;

        double[] pValues = {1, 0.75, 0.5, 0.25, 0.05, 0.005};
        DecisionTree[] trainingTrees = new DecisionTree[pValues.length];
        double minValidationError = Double.POSITIVE_INFINITY;
        int pValueIndex = 0;

        // Creates different trees with the chosen impurity type with p_Values varying between the trees
        for(int i = 0; i < trainingTrees.length; i++){
            trainingTrees[i] = new DecisionTree(chosenImpurity, pValues[i]);
            trainingTrees[i].buildClassifier(trainingCancer);
            double trainingError = trainingTrees[i].calcAvgError(trainingCancer);
            double validationError = trainingTrees[i].calcAvgError(validationCancer);
            if(validationError < minValidationError) {
                minValidationError = validationError;
                pValueIndex = i;
            }

            System.out.println("Decision Tree with p_value of: " + pValues[i]);
            System.out.println("The train error of the decision tree is " + trainingError);
            System.out.println("Max height on validation data: " + trainingTrees[i].getMaxHeight());
            System.out.println("Average height on validation data: " + trainingTrees[i].getAverageHeight());
            System.out.println("The validation error of the decision tree is " + validationError);
            System.out.println("---------------------------------------------");
        }

        System.out.println("Best validation error at p_value = " + pValues[pValueIndex]);
        System.out.println("Test error with best tree: " + trainingTrees[pValueIndex].calcAvgError(testingCancer));

        trainingTrees[pValueIndex].printTree();
    }
}
