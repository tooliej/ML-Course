package HW3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances data = loadData("./src/auto_price.txt");

		//Shuffles the data set
		Random rand = new Random();
		data.randomize(rand);

		double currentError;
		int bestK = 0;//initialisation for compilation
		LpDistance bestLpDistance = LpDistance.Infinity; //initialisation for compilation
		WeightingScheme bestWeightingScheme = WeightingScheme.Uniform; //initialisation for compilation

		for (int i = 0; i < 2; i++) { //i=0 is original, i=1 is scaled
			double minError = Double.POSITIVE_INFINITY; //initialise min error to infinity for each data set
			if (i == 1) //scale the data
				data = new FeatureScaler().scaleData(data);

			//checks for all combinations of hyper parameters
			for (int j = 1; j <= 20; j++) {
				for (LpDistance lp : LpDistance.values()) {
					for (WeightingScheme weight : WeightingScheme.values()) {

						//Creates KNN model and runs cross-validation
						Knn model = new Knn(j, lp, data, weight, DistanceCheck.Regular);
						currentError = model.crossValidationError(data, 10);

						//Updates best hyper-parameters
						if (currentError < minError) {
							minError = currentError;
							bestK = j;
							bestLpDistance = lp;
							bestWeightingScheme = weight;
						}
					}
				}
			}
			//Prints
			if (i == 0)
				printQ2("original", bestK, bestLpDistance, bestWeightingScheme, minError);
			else if (i == 1)
				printQ2("scaled", bestK, bestLpDistance, bestWeightingScheme, minError);
		}

		//Question 3:
		int[] num_of_folds = {data.numInstances(), 50, 10, 5, 3};
		for (int i : num_of_folds) {

			//Creates the models - regular and efficient
			Knn regularModel = new Knn(bestK, bestLpDistance, data, bestWeightingScheme, DistanceCheck.Regular);
			Knn efficientModel = new Knn(bestK, bestLpDistance, data, bestWeightingScheme, DistanceCheck.Efficient);

			//Runs cross-validation
			double regularError = regularModel.crossValidationError(data, i);
			double efficientError = efficientModel.crossValidationError(data, i);

			//Total Time
			long totalRegularTime = regularModel.getTotalTime();
			long totalEfficientTime = efficientModel.getTotalTime();

			//Average Time
			long regularAverageTime = totalRegularTime / i;
			long efficientAverageTime = totalEfficientTime / i;


			//Print
			printQ3Original(i, regularError, regularAverageTime, totalRegularTime);
			printQ3SEfficient(efficientError, efficientAverageTime, totalEfficientTime);
		}


	}

	private static void printQ2(String dataType, int k, LpDistance lp, WeightingScheme weight, double price) {
		System.out.println("----------------------------");
		System.out.println("Results for " + dataType + " dataset");
		System.out.println("----------------------------");
		System.out.println("Cross validation error with K = " + k + ", lp = " + lp + ", majority function =");
		System.out.println(weight + " for auto_price data is: " + price);
		System.out.println();
		System.out.println();
	}

	private static void printQ3Original(int folds, double price, long averageTime, long totalTime) {
		System.out.println("----------------------------");
		System.out.println("Results for " + folds + " dataset");
		System.out.println("----------------------------");
		System.out.println("Cross validation error of regular knn on auto_price dataset is "+ price + " and");
		System.out.println("the average elapsed time is " + averageTime);
		System.out.println("The total elapsed time is: " + totalTime);
		System.out.println();
	}

	private static void printQ3SEfficient(double price, long averageTime, long totalTime) {
		System.out.println("Cross validation error of efficient knn on auto_price dataset is " + price + " and");
		System.out.println("the average elapsed time is " + averageTime);
		System.out.println("The total elapsed time is: " + totalTime);
		System.out.println();
	}


}
