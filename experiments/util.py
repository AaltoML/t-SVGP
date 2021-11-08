import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def data_load(name, split=0.8, seed=0, path="./data/", normalize=False):
    """
    Util function to load the UCI datasets
    """
    np.random.seed(seed)
    df = pd.read_csv(r"" + path + name + ".csv")
    if name == "boston":
        Xo = df[
            [
                "crim",
                "zn",
                "indus",
                "chas",
                "nox",
                "rm",
                "age",
                "dis",
                "tax",
                "ptratio",
                "black",
                "lstat",
            ]
        ].to_numpy()
        Yo = df["medv"].to_numpy().reshape((-1, 1))
    elif name == "concrete":
        Xo = df[["cement", "water", "coarse_agg", "fine_agg", "age"]].to_numpy()
        Yo = df["compressive_strength"].to_numpy().reshape((-1, 1))
    elif name == "airfoil":
        Xo = df[
            [
                "Frequency",
                "AngleAttack",
                "ChordLength",
                "FreeStreamVelocity",
                "SuctionSide",
            ]
        ].to_numpy()
        Yo = df["Sound"].to_numpy().reshape((-1, 1))
    elif name == "ionosphere":
        n = np.genfromtxt(r"" + path + name + ".csv", delimiter=",")
        Xo = n[:, :-1]
        Yo = n[:, -1]
        Yo = np.reshape(Yo, (-1, 1))
    elif name == "sonar":
        n = np.genfromtxt(r"" + path + name + ".csv", delimiter=",")
        Xo = n[:, :-1]
        Yo = n[:, -1]
        Yo = np.reshape(Yo, (-1, 1))
    elif name == "diabetes":
        Xo = df[
            [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
        ].to_numpy()
        Yo = df["Outcome"].to_numpy().reshape((-1, 1))
    elif name == "german_t":
        numvars = [
            "creditamount",
            "duration",
            "installmentrate",
            "residencesince",
            "age",
            "existingcredits",
            "peopleliable",
            "classification",
        ]
        Xo = df[numvars[:-2]].to_numpy()
        Yo = df["classification"].to_numpy().reshape((-1, 1))
    elif name == "german_n":
        numvars = [
            "existingchecking",
            "duration",
            "credithistory",
            "purpose",
            "creditamount",
            "savings",
            "employmentsince",
            "installmentrate",
            "statussex",
            "otherdebtors",
            "residencesince",
            "property",
            "age",
            "otherinstallmentplans",
            "housing",
            "existingcredits",
            "job",
            "peopleliable",
            "telephone",
            "foreignworker",
            "extra1",
            "extra2",
            "extra3",
            "extra4",
        ]
        Xo = df[numvars].to_numpy()
        Yo = df["classification"].to_numpy().reshape((-1, 1))
    elif name == "wine":
        Xo = df[
            [
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
            ]
        ].to_numpy()
        Yo = df["y_value"].to_numpy().reshape((-1, 1))
    elif name == "naval":
        Xo = df[
            [
                "Lever",
                "Ship speed",
                "Gas Turbine",
                "GT rate",
                "Gas Generator",
                "Starboard Propeller",
                "Port Propeller Torque",
                "Hight Pressure",
                "GT Compressor",
                "GT Compressor",
                "HP Turbine",
                "GT Compressor",
                "GT Compressor",
                "GT exhaust",
                "Turbine",
                "Fuel",
            ]
        ].to_numpy()
        Yo = df["y2"].to_numpy().reshape((-1, 1))
    elif name == "snelson":
        Xo = df["x"].to_numpy().reshape((-1, 1))
        Yo = df["y"].to_numpy().reshape((-1, 1))

    if normalize == True:
        X_scaler = StandardScaler().fit(Xo)
        Y_scaler = StandardScaler().fit(Yo)
        X = X_scaler.transform(Xo)
        Y = Y_scaler.transform(Yo)
    else:
        X = Xo
        Y = Yo

    perm = np.random.permutation(X.shape[0])
    np.take(X, perm, axis=0, out=X)
    np.take(Y, perm, axis=0, out=Y)

    n = Y.shape[0]
    t = int(np.floor(n * split))
    x = X[:t, :]
    y = Y[:t, :]
    xt = X[t:, :]
    yt = Y[t:, :]
    data_train = (x, y)
    data_test = (xt, yt)

    return data_train, data_test
