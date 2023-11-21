"""Module that handles the full Anonymization pipeline
"""
import sys
import jpype
import jaydebeapi
import pandas as pd
from snsynth import Synthesizer
from faker import Faker
import numpy as np
import xml.etree.ElementTree as ET


def rewriteFakeQueries(fakeValues, path):
    tree = ET.parse(path)
    templates = tree.getroot()
    for fakerDict in fakeValues:
        for key in fakerDict:
            for tag in templates.iter("value"):
                if tag.text == key:
                    print("FOUND. REPLACING: " + tag.text)
                    tag.text = fakerDict[key]
    print("Query templates rewritten for sensitive values")
    # ET.indent(templates)
    tree.write(path)


def fakeColumn(dataset, col, locales, method, seed=0):
    """Method that generates fake values for columns that are considered sensitive

    Args:
        dataset (DataFrame): A Pandas dataframe holding anonymized data
        col (str): The name of the column
        locales (str[]): A list of locales
        method (str): A string matching the desired faker method
        seed (int, optional): Seed for the fake values. Defaults to 0.

    Returns:
        dict: Mapping from original to fake values
    """
    fake = Faker(locales)

    fake.seed_instance(seed)

    sensDict = {}

    dataset[col] = dataset[col].astype(str)

    try:
        fakerFunc = getattr(fake.unique, method)
        exists = True
    except AttributeError:
        exists = False
        print("Faker method '" + method + "' not found. Resorting to random String")
        fakerFunc = fake.unique.pystr

    for val in dataset[col].unique():
        if exists:
            sensDict[val] = fakerFunc()
        else:
            maxLen = len(val)
            sensDict[val] = fakerFunc(min_chars=1, max_chars=maxLen)

    dataset[col] = dataset[col].map(sensDict)

    fake.unique.clear()

    return dataset, sensDict


def getTimestampColumns(dbTypeList):
    """A helper function that returns a list of indexes of timestamp-type columns

    Args:
        dbTypeList (any): A list of Database column metadata

    Returns:
        int[]: A list of indexes of timestamp-type columns
    """
    timestampIndexes = []
    for i, entry in enumerate(dbTypeList):
        for dType in entry.values:
            if dType == "TIMESTAMP":
                timestampIndexes.append(i)
    return timestampIndexes


def dfFromTable(curs, table):
    """Helper function that creates a pandas DataFrame from a jaydebe connection

    Args:
        curs (cursor): The connection cursor
        table (str): The name of the table

    Returns:
        Dataframe,string[]: The table as a DataFrame and the indexes of timestamp columns
    """
    curs.execute(f"SELECT * FROM {table}")

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
    """Helper function to fill a DB table from a DataFrame

    Args:
        curs (cursor): The connection cursor
        df (DataFrame): Pandas DataFrame
        table (string): The name of the table
        timestampIndexes (int[]): A list of indexes of timestamp-type columns
    """
    df = df.replace(np.nan, None)

    for ind in timestampIndexes:
        name = df.columns[ind]
        df[name] = pd.to_datetime(df[name])

    tuples = [tuple(x) for x in df.values]

    if len(timestampIndexes):
        import java

        for i, tup in enumerate(tuples):
            li = list(tup)
            for j in timestampIndexes:
                li[j] = java.sql.Timestamp @ li[j]
            tuples[i] = tuple(li)

    colSlots = f"({','.join('?' for _ in df.columns)})"
    insertSQL = f"insert into {table} values {colSlots}"
    curs.executemany(insertSQL, tuples)


def getDroppableInfo(dropCols, dataset):
    """Helper function that saves droppable columns from anonymization

    Args:
        dropCols (str[]): A list of column names
        dataset (DataFrame): The dataset

    Returns:
        DataFrame,int[]: The saved columns as a DataFrame and a list of the original indexes of the columns
    """
    savedColumns = dataset[dropCols]
    savedColumnsIndexes = []

    for col in dropCols:
        savedColumnsIndexes.append(dataset.columns.get_loc(col))
    return savedColumns, savedColumnsIndexes


def anonymize(dataset: str, anonConfig: dict, sensConfig: dict, templatesPath: str):
    dropCols = anonConfig["hide"]
    alg = anonConfig["alg"]
    eps = float(anonConfig["eps"])
    preEps = float(anonConfig["preEps"])
    cat = anonConfig["cat"]
    cont = anonConfig["cont"]
    ordi = anonConfig["ord"]
    savedColumns = []
    savedColumnsIndexes = []

    if dropCols:
        savedColumns, savedColumnsIndexes = getDroppableInfo(dropCols, dataset)
        dataset = dataset.drop(dropCols, axis=1)

    nullableFlag = dataset.isnull().values.any()

    synth = Synthesizer.create(alg, epsilon=eps, verbose=True)
    synthFrame = pd.DataFrame()

    if len(cat) == 0 and len(cont) == 0 and len(ordi) == 0:
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
            categorical_columns=cat,
            continuous_columns=cont,
            ordinal_columns=ordi,
            nullable=nullableFlag,
        )
        synthFrame = pd.DataFrame(sample)

    # Stitching the Frame back to its original form
    if dropCols:
        for ind, col in enumerate(dropCols):
            synthFrame.insert(savedColumnsIndexes[ind], col, savedColumns[col])

    if sensConfig:
        fakerList = []
        for sensCol in sensConfig["cols"]:
            synthFrame, values = fakeColumn(
                synthFrame,
                sensCol["name"],
                sensCol["locales"],
                sensCol["method"],
                int(sensConfig["seed"]),
            )
            fakerList.append(values)
        rewriteFakeQueries(fakerList, templatesPath)

    return synthFrame


def anonymizeDB(
    jdbcConfig: dict, anonConfig: dict, sensConfig: dict, templatesPath: str
):
    driver = jdbcConfig["driver"]
    url = jdbcConfig["url"]
    username = jdbcConfig["username"]
    password = jdbcConfig["password"]
    jar = "benchbase.jar"

    jpype.startJVM(classpath=[jar])

    conn = jaydebeapi.connect(driver, url, [username, password])

    curs = conn.cursor()
    table = anonConfig["table"]
    dataset, timestampIndexes = dfFromTable(curs, table)

    datasetAnon = anonymize(dataset, anonConfig, sensConfig, templatesPath)

    # Create empty table called ANON
    anonTableName = table + "_anonymized"

    curs.execute(f"DROP TABLE IF EXISTS {anonTableName}")

    createTableQuery = f"CREATE TABLE {anonTableName} AS TABLE {table} WITH NO DATA"
    curs.execute(createTableQuery)
    populateAnonFromDF(curs, datasetAnon, anonTableName, timestampIndexes)
    curs.close()
    conn.close()


def listFromString(string):
    if string:
        return list(string.split(","))
    else:
        return []


def configFromXML(path):
    tree = ET.parse(path)
    parameters = tree.getroot()
    jdbcConfig = {
        "driver": parameters.find("driver").text,
        "url": parameters.find("url").text,
        "username": parameters.find("username").text,
        "password": parameters.find("password").text,
    }
    anonConfig = {}
    sensConfig = {}

    anon = parameters.find("anon")

    for table in anon.findall("anonTable"):
        anonConfig["table"] = table.find("tablename").text
        anonConfig["hide"] = listFromString(table.find("droppable").text)
        anonConfig["cat"] = listFromString(table.find("categorical").text)
        anonConfig["cont"] = listFromString(table.find("continuous").text)
        anonConfig["ord"] = listFromString(table.find("ordinal").text)
        anonConfig["eps"] = table.find("eps").text
        anonConfig["preEps"] = table.find("preEps").text
        anonConfig["alg"] = table.find("algorithm").text

    sens = parameters.find("sensitive")

    if sens:
        sensConfig["seed"] = sens.get("seed", 0)
        sensList = []
        for sensCol in sens.findall("colName"):
            sensList.append(
                {
                    "name": sensCol.text,
                    "method": sensCol.get("method"),
                    "locales": sensCol.get("locales"),
                }
            )
        sensConfig["cols"] = sensList

    return jdbcConfig, anonConfig, sensConfig


def main():
    """Entry method"""
    if len(sys.argv) == 2:
        confPath = sys.argv[1]
        templatesPath = ""

    elif len(sys.argv) == 3:
        confPath = sys.argv[1]
        templatesPath = sys.argv[2]

    else:
        print("Not enough arguments provided: <configPath> <templatesPath (optional)>")
        return

    jdbcConfig, anonConfig, sensConfig = configFromXML(confPath)
    anonymizeDB(jdbcConfig, anonConfig, sensConfig, templatesPath)


if __name__ == "__main__":
    main()
