package supersetas;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.ESignificanceTestResult;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;
import ai.libs.jaicore.basic.kvstore.KVStoreSequentialComparator;
import ai.libs.jaicore.basic.kvstore.KVStoreStatisticsUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;

public class TableGenerator {

	// list defining the order of methods in which they are presented in the generated table (from left to right)
	private static final List<String> ORDER_OF_METHODS = Arrays.asList("ignore_censored", "clip_censored", "par10", "schmee_hahn", "superset");

	// DB connection config & init
	private static final IDatabaseConfig isysDBConfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("isys-db.properties"));
	private static final IDatabaseAdapter isysDBAdapter = DatabaseAdapterFactory.get(isysDBConfig);

	// output file name
	public static final String OUTPUT_FILENAME_LINEAR_MODELS = "linear_models.tex";

	// names of common field keys
	public static final String IMP_TYPE = "imputation_type";
	public static final String ORDER = "order";

	private static void loadLinearModelData(final String table, final KVStoreCollection col) throws SQLException {
		// create a hashmap which is inserted in each of the loaded KVStores containing the imputation type.
		Map<String, String> commonFields = new HashMap<>();
		commonFields.put(IMP_TYPE, table.substring(new String("linear_model_tf").length() + 1).replace("_", "-"));
		for (int i = 0; i < ORDER_OF_METHODS.size(); i++) {
			if (ORDER_OF_METHODS.get(i).equals(commonFields.get(IMP_TYPE))) {
				commonFields.put(ORDER, i + "");
				break;
			}
		}
		System.out.println(commonFields);

		// load KVStoreCollection from the given table adding the imputation type obtained from the table's name to each row
		col.addAll(KVStoreUtil.readFromMySQLQuery(isysDBAdapter, String.format("SELECT * FROM %s WHERE metric=\"par10\"", table), commonFields));
	}

	public static void main(final String[] args) throws SQLException, IOException {
		// load data from all the tables
		KVStoreCollection linearModelCol = new KVStoreCollection();
		linearModelCol.setCollectionID("linear-models");
		loadLinearModelData("linear_model_tf_superset", linearModelCol);
		loadLinearModelData("linear_model_tf_par10", linearModelCol);
		loadLinearModelData("linear_model_tf_clip_censored", linearModelCol);
		loadLinearModelData("linear_model_tf_ignore_censored", linearModelCol);
		loadLinearModelData("linear_model_tf_schmee_hahn", linearModelCol);

		// remove GRAPHS scenario from all the results
		linearModelCol.removeAny("GRAPHS-2015");

		// group the kvstores, so that we have all the folds of a scenario and imputation type merged into one kvstore
		Map<String, EGroupMethod> grouping = new HashMap<>();
		grouping.put("result", EGroupMethod.AVG);
		KVStoreCollection grouped = linearModelCol.group(new String[] { "scenario_name", "metric", "imputation_type", "approach", ORDER }, grouping);

		// rank the imputation types per scenario according to their result and store it in rank
		KVStoreStatisticsUtil.rank(grouped, "scenario_name", "imputation_type", "result", "rank");
		// compute descriptive statistics of the ranks => basis for avg rank statistics
		Map<String, DescriptiveStatistics> avgranks = KVStoreStatisticsUtil.averageRank(grouped, "imputation_type", "rank");
		// compute descriptive statistics of the par10 score => basis for (geometric) mean of the par10 scores
		Map<String, DescriptiveStatistics> avgpar10 = KVStoreStatisticsUtil.averageRank(grouped, "imputation_type", "result");

		// sort the data according to the scenario name as first priority and then by the order as defined in the ORDER list.
		grouped.sort(new KVStoreSequentialComparator("scenario_name", ORDER));

		// Conduct wilcoxon signed rank sum test for significance testing
		// As the setting, we consider >scenario_name< for each of which we compare the >result_list< of the imputation_type >superset< to all other >imputation_type<s pairing via the >fold< number.
		// The result is stored in the field >sig<.
		System.out.println(grouped.get(0));
		KVStoreStatisticsUtil.wilcoxonSignedRankTest(grouped, "scenario_name", "imputation_type", "fold", "result_list", "superset", "sig");

		// add the average ranks of each imputation method as another row
		for (Entry<String, DescriptiveStatistics> entry : avgranks.entrySet()) {
			IKVStore store = new KVStore();
			store.put("scenario_name", "avg. rank");
			store.put("result", entry.getValue().getMean());
			store.put("imputation_type", entry.getKey());
			grouped.add(store);
		}

		// add the average par10 score of each imputation method as another row
		for (Entry<String, DescriptiveStatistics> entry : avgpar10.entrySet()) {
			IKVStore store = new KVStore();
			store.put("scenario_name", "avg. PAR10");
			store.put("result", entry.getValue().getMean());
			store.put("imputation_type", entry.getKey());
			grouped.add(store);
		}

		// add the geometric mean of par10 scores of each imputation method as another row
		for (Entry<String, DescriptiveStatistics> entry : avgpar10.entrySet()) {
			IKVStore store = new KVStore();
			store.put("scenario_name", "geo. PAR10");
			store.put("result", entry.getValue().getGeometricMean());
			store.put("imputation_type", entry.getKey());
			grouped.add(store);
		}

		KVStoreStatisticsUtil.best(grouped, "scenario_name", "imputation_type", "result", "best");
		for (IKVStore store : grouped) {
			// round values to 2 decimals + make sure each entry has exactly 2 decimals (e.g. 5 is transformed into 5.00 or 5.1 into 5.10)
			store.put("result", ValueUtil.valueToString(store.getAsDouble("result"), 2));

			// highlight best performing approach in bold face
			if (store.getAsBoolean("best")) {
				store.put("entry", "\\textbf{" + store.getAsString("result") + "}");
			} else {
				store.put("entry", store.getAsString("result"));
			}

			// if the entry has a rank make it visible in the table
			if (store.containsKey("rank")) {
				StringBuilder sb = new StringBuilder();
				sb.append(store.getAsString("entry"));

				// append extra information to the entry of each cell
				sb.append(" (");
				// rank for this scenario
				sb.append(store.getAsString("rank"));
				sb.append("/");
				// number of folds evaluated so far (shoudl definitely be commented out/removed for submission)
				sb.append(store.getAsStringList("fold").size());
				sb.append(")");

				// update the cell entry value
				store.put("entry", sb.toString());
			}

			// if a store contains a field >sig<, compile it into a bullet/circ representation
			if (store.containsKey("sig")) {
				String sigAppendix = null;
				switch (ESignificanceTestResult.valueOf(store.getAsString("sig"))) {
				case INFERIOR:
					sigAppendix = "$\\bullet$";
					break;
				case SUPERIOR:
					sigAppendix = "$\\circ";
					break;
				case TIE:
					sigAppendix = "$\\phantom{\\bullet}$";
					break;
				}
				store.put("entry", store.getAsString("entry") + " " + sigAppendix);
			}
		}

		// generate latex code for the table taking scenarios as rows, imputation types as columns and showing the value of entry in each cell
		String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(grouped, "scenario_name", "imputation_type", "entry").replace("_", "\\_");
		System.out.println(latexTable);

		// write the code of the table to file.
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File(OUTPUT_FILENAME_LINEAR_MODELS)))) {
			bw.write(latexTable);
		}

	}
}
