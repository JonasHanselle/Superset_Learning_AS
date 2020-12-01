package supersetas;

import java.io.File;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;
import ai.libs.jaicore.basic.kvstore.KVStoreStatisticsUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;

public class TableGenerator {
	// private static final IDatabaseConfig otfmlConfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("otfml.properties"));
	// private static final IDatabaseAdapter otfmlAdapter = DatabaseAdapterFactory.get(otfmlConfig);
	private static final IDatabaseConfig isysDBConfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("isys-db.properties"));
	private static final IDatabaseAdapter isysDBAdapter = DatabaseAdapterFactory.get(isysDBConfig);

	public static final String IMP_TYPE = "imputation_type";

	private static void loadLinearModelData(final String table, final KVStoreCollection col) throws SQLException {
		// create a hashmap which is inserted in each of the loaded KVStores containing the imputation type.
		Map<String, String> commonFields = new HashMap<>();
		commonFields.put(IMP_TYPE, table.substring(new String("linear_model_tf").length() + 1).replace("_", "-"));
		// load KVStoreCollection from the given table adding the imputation type obtained from the table's name to each row
		col.addAll(KVStoreUtil.readFromMySQLQuery(isysDBAdapter, String.format("SELECT * FROM %s WHERE metric=\"par10\"", table), commonFields));
	}

	public static void main(final String[] args) throws SQLException {
		// load data from all the tables
		KVStoreCollection linearModelCol = new KVStoreCollection();
		loadLinearModelData("linear_model_tf_superset", linearModelCol);
		loadLinearModelData("linear_model_tf_par10", linearModelCol);
		loadLinearModelData("linear_model_tf_clip_censored", linearModelCol);
		loadLinearModelData("linear_model_tf_ignore_censored", linearModelCol);
		loadLinearModelData("linear_model_tf_schmee_hahn", linearModelCol);

		Map<String, EGroupMethod> grouping = new HashMap<>();
		grouping.put("result", EGroupMethod.AVG);
		KVStoreCollection grouped = linearModelCol.group(new String[] { "scenario_name", "metric", "imputation_type", "approach" }, grouping);

		KVStoreStatisticsUtil.rank(grouped, "scenario_name", "imputation_type", "result", "rank");
		Map<String, DescriptiveStatistics> avgranks = KVStoreStatisticsUtil.averageRank(grouped, "imputation_type", "rank");
		Map<String, DescriptiveStatistics> avgpar10 = KVStoreStatisticsUtil.averageRank(grouped, "imputation_type", "result");

		for (Entry<String, DescriptiveStatistics> entry : avgranks.entrySet()) {
			IKVStore store = new KVStore();
			store.put("scenario_name", "avg. rank");
			store.put("result", entry.getValue().getMean());
			store.put("imputation_type", entry.getKey());
			grouped.add(store);
		}
		for (Entry<String, DescriptiveStatistics> entry : avgpar10.entrySet()) {
			IKVStore store = new KVStore();
			store.put("scenario_name", "avg. PAR10");
			store.put("result", entry.getValue().getMean());
			store.put("imputation_type", entry.getKey());
			grouped.add(store);
		}

		for (Entry<String, DescriptiveStatistics> entry : avgpar10.entrySet()) {
			IKVStore store = new KVStore();
			store.put("scenario_name", "geo. PAR10");
			store.put("result", entry.getValue().getGeometricMean());
			store.put("imputation_type", entry.getKey());
			grouped.add(store);
		}

		KVStoreStatisticsUtil.best(grouped, "scenario_name", "imputation_type", "result", "best");
		for (IKVStore store : grouped) {
			store.put("scenario_name", store.getAsString("scenario_name").replace("_", "\\_"));
			store.put("result", ValueUtil.valueToString(store.getAsDouble("result"), 2));

			if (store.getAsBoolean("best")) {
				store.put("entry", "\\textbf{" + store.getAsString("result") + "}");
			} else {
				store.put("entry", store.getAsString("result"));
			}

			if (store.containsKey("rank")) {
				store.put("entry", store.getAsString("entry") + " (#" + store.getAsString("rank") + "/" + store.getAsStringList("fold").size() + ")");
			}
		}

		String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(grouped, "scenario_name", "imputation_type", "entry");
		System.out.println("\n#########\n");
		System.out.println(latexTable);

	}
}
