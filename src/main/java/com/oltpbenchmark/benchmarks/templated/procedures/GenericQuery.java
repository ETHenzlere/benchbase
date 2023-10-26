/*
 * Copyright 2020 by OLTPBenchmark Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package com.oltpbenchmark.benchmarks.templated.procedures;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Types;
import java.util.List;
import java.util.Random;

import org.immutables.value.Value;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.oltpbenchmark.api.Procedure;
import com.oltpbenchmark.api.SQLStmt;
import com.oltpbenchmark.distributions.ScrambledZipfianGenerator;
import com.oltpbenchmark.distributions.ZipfianGenerator;
import com.oltpbenchmark.util.TextGenerator;

public abstract class GenericQuery extends Procedure {

    protected static final Logger LOG = LoggerFactory.getLogger(GenericQuery.class);

    /** Execution method with parameters. */
    public void run(Connection conn, List<Object> params) throws SQLException {
        try (PreparedStatement stmt = getStatement(conn, params); ResultSet rs = stmt.executeQuery()) {
            while (rs.next()) {
                // do nothing
            }
        }
        conn.commit();
    }

    /** Execution method without parameters. */
    public void run(Connection conn) throws SQLException {
        QueryTemplateInfo queryTemplateInfo = this.getQueryTemplateInfo();

        try (PreparedStatement stmt = this.getPreparedStatement(conn, queryTemplateInfo.getQuery());
                ResultSet rs = stmt.executeQuery()) {
            while (rs.next()) {
                // do nothing
            }
        }
        conn.commit();
    }

    public PreparedStatement getStatement(Connection conn, List<Object> params) throws SQLException {
        QueryTemplateInfo queryTemplateInfo = this.getQueryTemplateInfo();
        Random rng = new Random();

        PreparedStatement stmt = this.getPreparedStatement(conn, queryTemplateInfo.getQuery());
        String[] paramsTypes = queryTemplateInfo.getParamsTypes();
        for (int i = 0; i < paramsTypes.length; i++) {
            if (paramsTypes[i].equalsIgnoreCase("NULL")) {
                stmt.setNull(i + 1, Types.NULL);
                // ENTER RIGHT HERE WITH THE DISTRIBUTION
            } else if (paramsTypes[i].equalsIgnoreCase("DISTRIBUTION")) {
                String distType = params.get(i).toString();
                String min, max;
                int val;
                switch (distType) {
                    case "zipf":
                        min = params.get(i + 1).toString();
                        max = params.get(i + 2).toString();
                        ZipfianGenerator zipf = new ZipfianGenerator(rng, Integer.parseInt(min),
                                Integer.parseInt(max));
                        stmt.setInt(i + 1, zipf.nextInt());
                        break;
                    case "uniform":
                        max = params.get(i + 1).toString();
                        val = rng.nextInt(Integer.parseInt(max));
                        stmt.setInt(i + 1, val);
                        break;
                    case "binomial":
                        int minI = Integer.parseInt(params.get(i + 1).toString());
                        int maxI = Integer.parseInt(params.get(i + 2).toString());
                        do {
                            val = (int) (minI + Math.abs(rng.nextGaussian()) * maxI);
                        } while (val > maxI || val < minI);

                        stmt.setInt(i + 1, val);
                        break;
                    case "scrambled":
                        min = params.get(i + 1).toString();
                        max = params.get(i + 2).toString();
                        ScrambledZipfianGenerator scramZipf = new ScrambledZipfianGenerator(Integer.parseInt(min),
                                Integer.parseInt(max));
                        stmt.setInt(i + 1, scramZipf.nextInt());
                        break;
                    case "string":
                        max = params.get(i + 1).toString();
                        String randText = TextGenerator.randomStr(rng, Integer.parseInt(max));
                        stmt.setString(i + 1, randText);
                        break;
                    default:
                        throw new RuntimeException(
                                "No suitable distribution found. Currently supported are 'zipf' | 'normal' | 'uniform' | 'string' ");
                }

                System.out.println(stmt.toString());

            } else {
                try {
                    // TODO: add support for nullable other types
                    // For instance, can we provide a <value /> tag in the XML file to represent a
                    // NULL value?
                    // Or does it need a special marker like "$null" to signify a NULL value?
                    Object param = params.get(i);
                    stmt.setObject(i + 1, param,
                            Integer.parseInt(Types.class.getDeclaredField(paramsTypes[i]).get(null).toString()));
                } catch (Exception e) {
                    e.printStackTrace();
                    throw new RuntimeException(
                            "Error when setting parameters. Parameter type: " + paramsTypes[i] + ", parameter value: "
                                    + params.get(i));
                }
            }
        }
        return stmt;
    }

    public abstract QueryTemplateInfo getQueryTemplateInfo();

    @Value.Immutable
    public interface QueryTemplateInfo {

        /** Query string for this template. */
        SQLStmt getQuery();

        /** Query parameter types. */
        String[] getParamsTypes();

        /** Potential query parameter values. */
        String[] getParamsValues();
    }

}
