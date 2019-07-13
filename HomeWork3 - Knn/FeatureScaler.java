package HW3;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 *
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public Instances scaleData(Instances instances) throws Exception {
		Standardize stand = new Standardize();
		stand.setInputFormat(instances);

		return Filter.useFilter(instances, stand);
	}
}