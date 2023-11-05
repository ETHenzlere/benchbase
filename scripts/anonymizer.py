import jaydebeapi
import pandas as pd
from snsynth import Synthesizer
import sys
import numpy as np
import jpype
from jpype.types import *
import jpype.imports


def getTimestampColumns(dbTypeList):
    timestampIndexes = []
    for i, entry in enumerate(dbTypeList):
        for dType in entry.values:
            if dType == "TIMESTAMP":
                timestampIndexes.append(i)
    return timestampIndexes


def splitString(longString):
    return [] if longString == "-1" else [i for i in longString.split(",")]


def dfFromTable(curs, table):
    curs.execute("SELECT * FROM {0}".format(table))

    res = curs.fetchall()
    meta = curs.description

    cols = []
    colTypes = []
    for entry in meta:
        cols.append(str(entry[0]))
        colTypes.append(entry[1])
    timestampIndexes = getTimestampColumns(colTypes)

    frame = pd.DataFrame(res, columns=cols)

    return frame, timestampIndexes


def populateAnonFromDF(curs, df, table, timestampIndexes):
    df = df.replace(np.nan, None)

    for ind in timestampIndexes:
        name = df.columns[ind]
        df[name] = pd.to_datetime(df[name])

    tuples = [tuple(x) for x in df.values]

    if len(timestampIndexes):
        import java

        for i in range(0, len(tuples)):
            li = list(tuples[i])
            for j in timestampIndexes:
                li[j] = java.sql.Timestamp @ li[j]
            tuples[i] = tuple(li)

    colSlots = "({0})".format(",".join("?" for col in df.columns))
    insertSQL = "insert into {0} values {1}".format(table, colSlots)
    curs.executemany(insertSQL, tuples)


def getDroppableInfo(dropCols, dataset):
    savedColumns = dataset[dropCols]
    savedColumnsIndexes = []

    for col in dropCols:
        savedColumnsIndexes.append(dataset.columns.get_loc(col))
    return savedColumns, savedColumnsIndexes


def synthesize(
    driver,
    url,
    username,
    password,
    eps,
    preEps,
    table,
    dropCols,
    categorical,
    continuous,
    ordinal,
):
    jpype.startJVM(classpath=["benchbase.jar"])

    conn = jaydebeapi.connect(driver, url, [username, password])

    curs = conn.cursor()

    dataset, timestampIndexes = dfFromTable(curs, table)

    savedColumns = []
    savedColumnsIndexes = []
    if dropCols:
        savedColumns, savedColumnsIndexes = getDroppableInfo(dropCols, dataset)
        dataset = dataset.drop(dropCols, axis=1)

    nullableFlag = dataset.isnull().values.any()

    synthFrame = pd.DataFrame()

    synth = Synthesizer.create("mst", epsilon=eps, verbose=True)

    if len(categorical) == 0 and len(continuous) == 0 and len(ordinal) == 0:
        sample = synth.fit_sample(
            dataset,
            preprocessor_eps=preEps,
            nullable=nullableFlag,
        )
        synthFrame = pd.DataFrame(sample)
    else:
        sample = synth.fit_sample(
            dataset,
            preprocessor_eps=preEps,
            categorical_columns=categorical,
            continuous_columns=continuous,
            ordinal_columns=ordinal,
            nullable=nullableFlag,
        )
        synthFrame = pd.DataFrame(sample)

    synthFrame = pd.DataFrame(sample)

    # Stitching the Frame back to its original form
    if dropCols:
        for ind in range(len(dropCols)):
            columnName = dropCols[ind]
            synthFrame.insert(
                savedColumnsIndexes[ind], columnName, savedColumns[columnName]
            )

    print(synthFrame.describe(include="all"))

    # Create empty table called ANON
    anonTableName = table + "_anonymized"
    try:
        curs.execute("drop table {0}".format(anonTableName))
        print("Dropped old Anonymized table. Creating new table")
    except:
        print("No Anonymized table found. Creating new table")

    createTableQuery = "CREATE TABLE {0} AS TABLE {1} WITH NO DATA".format(
        anonTableName, table
    )
    curs.execute(createTableQuery)

    populateAnonFromDF(curs, synthFrame, anonTableName, timestampIndexes)

    curs.close()
    conn.close()


def main():
    if len(sys.argv) < 8:
        print(
            "Not enough arguments provided: <driver> <url> <username> <password> <eps> <preprocessorEps> <table> <dropColumns> <categoricalColumns> <continuousColumns> <ordinalColumns>"
        )
        return

    driver = sys.argv[1]

    url = sys.argv[2]

    username = sys.argv[3]

    pw = sys.argv[4]

    eps = float(sys.argv[5])

    preprocEps = float(sys.argv[6])

    table = sys.argv[7]

    saveList = sys.argv[8]

    columnsToSave = splitString(saveList)

    catList = sys.argv[9]
    categorical = splitString(catList)

    contList = sys.argv[10]
    continuous = splitString(contList)

    ordList = sys.argv[11]
    ordinal = splitString(ordList)

    print("The following columns will be saved: ")
    print(columnsToSave)

    synthesize(
        driver,
        url,
        username,
        pw,
        eps,
        preprocEps,
        table,
        columnsToSave,
        categorical,
        continuous,
        ordinal,
    )
    return


if __name__ == "__main__":
    main()
