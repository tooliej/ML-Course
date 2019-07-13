package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

    //the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1;
        m_coefficients = new double[m_truNumAttributes + 1];

        if(m_alpha == 0)
            findAlpha(trainingData);

        for(int i = 0; i < m_coefficients.length; i++) {
            m_coefficients[i] = 1;
        }

        double currErr = 0;
        double prevErr = Double.POSITIVE_INFINITY;
        while(Math.abs(prevErr - currErr) > 0.003) {
            prevErr = currErr;
            for(int i = 0; i < 100; i++) {
                m_coefficients = gradientDescent(trainingData);
            }
            currErr = calculateMSE(trainingData);
        }
	}
	
	private void findAlpha(Instances data) throws Exception {
        double currErr;
        double prevErr;
        double minErr = Double.POSITIVE_INFINITY;
        double minAlpha = Double.POSITIVE_INFINITY;

        for(int i = -17; i <= 0; i++) {
            for(int j = 0; j < m_coefficients.length; j++) {
                m_coefficients[j] = 1;
            }

            currErr = 0;
            prevErr = Double.POSITIVE_INFINITY;
            m_alpha = Math.pow(3.0, i);
            currErr = calculateError(data, prevErr, currErr);
            if(minErr > currErr) {
                minErr = currErr;
                minAlpha = m_alpha;
            }

            m_alpha = minAlpha;
        }
	}

	private double calculateError(Instances data, double o_prevErr, double io_currErr) throws Exception {
        for(int i = 1; i <= 20000; i++){
            m_coefficients = gradientDescent(data);
            if(i % 100 == 0) {
                io_currErr = calculateMSE(data);
                if(io_currErr < o_prevErr)
                    o_prevErr = io_currErr;
                else
                    return o_prevErr;
            }
        }

	    return io_currErr;
    }

	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
	    double[] tempData = new double[m_truNumAttributes + 1];

	    for(int i = 0; i < tempData.length; i++){
	        tempData[i] = m_coefficients[i] - m_alpha *
                    (partialDerivative(trainingData, i));
        }

        for(int i = 0; i < tempData.length; i++) {
	        m_coefficients[i] = tempData[i];
        }

	    return m_coefficients;
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {

	    // Calculation of h function
        double innerProduct = m_coefficients[0];
        for(int i = 1; i <= m_truNumAttributes; i++) {
            innerProduct += instance.value(i - 1) * m_coefficients[i];
        }

        return innerProduct;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
        // J function
        double mse = 0;

        for(int i = 0; i < data.numInstances(); i++) {
            mse += Math.pow(regressionPrediction(data.instance(i)) - data.instance(i).value(m_ClassIndex), 2.0);
        }

		return mse / (2 * data.numInstances());
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
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
     * Returns the partial derivative with the respect to the variable i_coefficient
     * @param trainingData
     * @param i_coefficient
     * @return
     */
	private double partialDerivative(Instances trainingData, int i_coefficient) throws Exception{
        double partialDer = 0;

        for(int i = 0; i < trainingData.numInstances(); i++) {

           if(i_coefficient == 0)
               partialDer += regressionPrediction(trainingData.instance(i)) - trainingData.instance(i).value(m_ClassIndex);
           else
            partialDer += (regressionPrediction(trainingData.instance(i)) - trainingData.instance(i).value(m_ClassIndex))
                    * trainingData.instance(i).value(i_coefficient - 1);
        }

        return partialDer / trainingData.numInstances();
    }

    public double getAlpha() {
        return m_alpha;
    }

    public void setAlpha(double m_alpha) {
        this.m_alpha = m_alpha;
    }
}
