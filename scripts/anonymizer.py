import jaydebeapi
import pandas as pd
from snsynth import Synthesizer
import sys
import re
import glob


def dfFromTable(curs, table):
    getTableQuery = "SELECT * FROM {0}".format(table)
    curs.execute(getTableQuery)

    res = curs.fetchall()
    meta = curs.description

    cols = []
    for entry in meta:
        cols.append(entry[0])

    print("Extracted Columns:")
    print(cols)

    frame = pd.DataFrame(res)
    frame.columns = cols

    return frame


def populateAnonFromDF(curs, df, table):
    tuples = [tuple(x) for x in df.values]
    colSlots = "({0})".format(",".join("?" for col in df.columns))
    insertSQL = "insert into {0} values {1}".format(table, colSlots)
    curs.executemany(insertSQL, tuples)


def getJar(url):
    jdbcInfo = re.search(r":(\w+):", url).group(1)
    jdbcJarPath = glob.glob("lib/{0}*.jar".format(jdbcInfo))[0]
    print("Found the following Jar: " + jdbcJarPath)
    return jdbcJarPath


def splitString(longString):
    return [] if longString == "" else [i for i in longString.split(",")]


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
    pw,
    eps,
    preEps,
    table,
    dropCols,
    categorical,
    continuous,
    ordinal,
):
    jar = getJar(url)
    conn = jaydebeapi.connect(
        driver,
        url,
        [username, pw],
        jar,
    )

    curs = conn.cursor()

    dataset = dfFromTable(curs, table)

    savedColumns = []
    savedColumnsIndexes = []
    if dropCols:
        savedColumns, savedColumnsIndexes = getDroppableInfo(dropCols, dataset)
        dataset = dataset.drop(dropCols, axis=1)

    synth = Synthesizer.create("mst", epsilon=eps, verbose=True)

    sample = synth.fit_sample(
        dataset,
        preprocessor_eps=preEps,
        categorical_columns=categorical,
        continuous_columns=continuous,
        ordinal_columns=ordinal,
        nullable=False,
    )

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

    populateAnonFromDF(curs, synthFrame, anonTableName)

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
