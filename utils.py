import numpy as np

def importData(filepath):
    '''
    Expects a file with this structure:
    A_id,Size,Weight,Sweetness,Crunchiness,Juiciness,Ripeness,Acidity,Quality
    
    INPUTS:
    filepath: string - The filepath for the dataset

    OUTPUTS:
    X: nparray - (n,7) array of apple features
    Y: nparray - (n,1) array of apple quality, 0 = bad, 1 = goods
    '''
    #Initialization
    file = open(filepath)
    X = np.array([])
    Y = np.array([])    

    #Parse every line
    for line in file:
        splitStr = line.split(",")
        X = np.append(X, splitStr[1:8])
        if(splitStr[-1][:len(splitStr[-1])-1] == "good"):
            Y = np.append(Y, 1)
        else:
            Y = np.append(Y, 0)
    
    #Fix datatype
    X = np.array([float(val) for val in X])
    #Reshape the arrays
    X = X.reshape((-1,7))
    Y = Y.reshape((-1,1))
    return X,Y


def zScoreNormilization(X):
    mean = np.mean(X, axis = 0)
    deviation = np.std(X, axis = 0)

    normalized_x = (X - mean)/deviation

    return normalized_x, mean, deviation




