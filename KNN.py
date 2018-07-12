
import numpy as np
import pandas as pd

""" Preprocessing Function"""
""" ----------------------"""
def data_type_preprocessing(df,y, var_type = 'continuous',
                            continuous_list = None, categorical_list = None, categorical_only= False):
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split
    ''' The following function will preprocess the data and create a train test split.
    It will be able to handle dataframes that are all categorical, all continuous, or mixed.
    it will create dummies for categorical variables. It will normalize continous variables.
    it will also create a train test split. For mixed dataframes it will return the indexes of 
    the continuous and categorical variables.
    
    var_type
    --------
    var_type = 'continuous' (by default. Indicates only continous variables in the dataframe)
    var_type = 'categorical' ('Indicates only a categorical dataframe)
    var_type = 'Mixed ('Indicates both categorical and continous variables present. 
                        Requires both categorical list, and continuous lists.)
    '''
    
    #get variable names from the function
    #------------------------------------
    
    #get the y col
    y_col = y
    
    #create the y array 
    y = np.array(df[y].astype(str))
    
    """ Determine the type variables we are working with and 
    process accrodingly"""
    
    if var_type == 'mixed':
        #get categorical & continuous variables
        cont = continuous_list
        cat = categorical_list
        
        #subset continuous
        cont_df = df[cont]
        
        #get the length of continuous and categorical to slice arrays
        split_position = len(cont)
        
        #get dummies for categorical variables
        cat_df = pd.get_dummies(df[cat].astype(str))
    
        #recreate the dataframe with dummies
        X = pd.merge(cont_df,cat_df, how = 'left', left_index=True, right_index = True)
        
    elif var_type == 'categorical':
        #create a dataframe of dummy variables that do not include y
        X = pd.get_dummies(df[[col for col in df.columns if col not in y_col]])
        
    elif var_type == 'continuous':
        X = df[[col for col in df.columns if col not in y_col]]
    
    
    # Get column names and convert X to an array
    col_names = X.columns
    X = np.array(X)
    
    # Create the train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
    #preprocess continuous variables
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    if var_type == 'mixed':
        #idenify the array index of categorical and continuous columns
        cont_idx = [i for i in range(split_position)]
        cat_idx = [i for i in range(split_position, col_names.shape[0])] 
        return X_train, X_test, y_train, y_test, col_names, cont_idx, cat_idx
    else:
        return X_train, X_test, y_train, y_test, col_names
    
"""Distance Measure Functions"""
"""--------------------------"""
# True minkowski Distance between two arrays
def minkowski_distance(A, B, p =2):
    p = 2
    diff_abs = np.abs(A - B)
    to_the_p = diff_abs**p
    sum_dist = sum(to_the_p)
    sum_div_p = sum_dist**(1/p)
    return sum_div_p

# Minkowski Distance Between Two Points for the Gower Distance Calculation
def minkowski_point_array(A, B, p = 2):
    '''This function will be used in the gower distance calculation.
    it differs from the true minkowsi in that calculates the minkowski distance
    between two points rather than between two arrays'''
    #absolute value of the difference
    diff_abs = np.abs(A - B)
    #to the p
    to_the_p = diff_abs**p
    #each point in the array raised to the value of 1/p
    dist_raised_1div_p = to_the_p**(1/p)
    return dist_raised_1div_p

# Jaccard Distance for Categorical Variables
def jaccard_distance(A, B):
    a = []
    b = []
    c = []
    for index in range(len(A)):
        if A[index] == 1 and B[index] == 1:
            a.append(1)
        elif A[index] == 1 and B[index] == 0:
            b.append(1)
        elif A[index] == 0 and B[index] == 1:
            c.append(1)
    return (sum(b) + sum(c))/(sum(a) + sum(b) + sum(c))

# Gower Distance for Mixed Data Types
def gower_distance(A, B, cont_idx, cat_idx, p = 2):
    """This function will return the jaccard distance calculation, an array of the minkowski point calculations.
    note: the minkowski 'point' distances are not summed. Just an array of the gower distances from point to point
    is returned. The function was built this way because we need to nomalize the gower array before computing the
    gower distances. The function also returns the demominator for the gower distance calculations
    (+ 1 for each continous and a 1 if categorical variables exist)"""
    denominator = len(cont_idx) + 1
    return jaccard_distance(A[cat_idx], B[cat_idx]), minkowski_point_array(A[cont_idx], B[cont_idx],p = 2), denominator


""" The KNN Algorithm"""
"""------------------"""
def knn_predictions(X_train, X_test, y_train, y_test, col_names, 
                    cont_idx = None, cat_idx = None, k = 5, distance = 'gower'):
    
    """
    Distance_measures
    -----------------
    1. 'gower'
    2. 'jaccard'
    3. 'euclidean'
    4. 'manhattan'
    """
    
    from sklearn.preprocessing import MinMaxScaler
    predictions = []
    for test_index in range(X_test.shape[0]):
        test = X_test[test_index,:]
        distances = []
        
        """The following section will populate the distance for each observation in the training
        set to the current test observation. Due to the normalization of the distances prior to 
        concatenation with the jaccard caculation, the gower distance has a much lengthy procedure."""
        
        if distance == 'gower':
        #calculate the distance of the test line to all of the other training lines
            for train_index in range(X_train.shape[0]):
                #indentify the training line
                train = X_train[train_index,:]

                #Calcuates the Gower Distance

                '''get the an the jaccard distances calculations, 
                an array of the minkowski distance calculations for each point, and 
                the denominator'''

                jaccard, gower_point_dist, denominator = gower_distance(test, train, cont_idx, cat_idx)
                #store each calculation to an array
                if train_index == 0:
                    jaccard_array = np.array(jaccard).reshape(1,1)
                    gower_array = gower_point_dist.reshape(1,len(cont_idx))
                else:
                    jaccard_array = np.append(jaccard_array, np.array(jaccard).reshape(1,1),axis=0)
                    gower_array = np.append(gower_array,gower_point_dist.reshape(1,len(cont_idx)),axis=0)

            #normalize the gower distance to be on a zero to one scale
            scaler = MinMaxScaler()
            gower_array = scaler.fit_transform(gower_array)

            #combine the jackard distance calcuation wtih the normalized gower array
            normalized_gower_array = np.hstack((jaccard_array, gower_array))
            '''CALCULATE ALL THE GOWER DISTANCES LINE BY LINE AND APPEND TO A LIST''' 
            for i in range(normalized_gower_array.shape[0]):
                distances.append((sum(normalized_gower_array[i,:]))/denominator)
                
        #create an array of manhattan distances        
        elif distance == 'manhattan':
            for train_index in range(X_train.shape[0]):
                train = X_train[train_index,:]
                distances.append(minkowski_distance(test, train, p =1))
                    
        #create an array of euclidean distances 
        elif distance == 'euclidean':
            for train_index in range(X_train.shape[0]):
                train = X_train[train_index,:]
                distances.append(minkowski_distance(test, train, p =2))
                    
        #create an array of manhattan distances 
        elif distance == 'jaccard':
            for train_index in range(X_train.shape[0]):
                train = X_train[train_index,:]
                distances.append(jaccard_distance(test, train))
                    
        # 3. get the indexes for the nearest neighbors
        prediction_index = np.array(distances).argsort()[:k]

        # 4. get all the predictions values
        prediction_values = y_train[prediction_index]

        # 5. Use the numpy unique function to determine which prediction appeared the most
        # returns the position in the unique array with the maximum count
        max_index = np.argmax(np.unique(prediction_values , return_counts=True)[1])

        # 6. returns the value of the maximum count
        prediction = np.unique(prediction_values , return_counts=True)[0][max_index]

        # 7. store the prediction to the list
        predictions.append(prediction)
        
    return np.array(predictions)

"""Accuarcy Report"""
"""---------------"""
def accuaracy_report(y_test, predictions):
    from sklearn.metrics import classification_report, confusion_matrix
    print('The Accuracy Score is: ',np.array(np.sum(np.equal(predictions, y_test))) / y_test.shape[0],'\n')
    print('The Classification Report')
    print('-------------------------')
    print(classification_report(y_test, predictions),'\n')
    print('The Confusion Matrix')
    print('-------------------------')
    print(pd.DataFrame(confusion_matrix(y_test, predictions)).apply(lambda x: x / sum(x), axis=1))

"""********************************"""

"""The Complete KNN Wrapper Fuction"""
"""--------------------------------"""
def Complete_KNN(df, y, var_type = 'continuous',
                 continuous_list = None, categorical_list = None, categorical_only= False,
                 k = 5, distance_measure = 'euclidean'):
    
    #import packages
    import numpy as np
    import pandas as pd
    
    # the index variables will be populated if we are using a mixed type
    cont_idx = None
    cat_idx = None
    
    # Step 1 - Preprocessing
    if var_type == 'mixed':
        X_train, X_test, y_train, y_test, col_names, cont_idx, cat_idx = data_type_preprocessing(df,y, var_type = var_type,
                                                                                        continuous_list = continuous_list,
                                                                                        categorical_list = categorical_list,
                                                                                        categorical_only= categorical_only)
    else:
        X_train, X_test, y_train, y_test, col_names = data_type_preprocessing(df,y, var_type = var_type,
                                                                                        continuous_list = continuous_list,
                                                                                        categorical_list = categorical_list,
                                                                                        categorical_only= categorical_only)
        
    # Step 2 - Generate Predictions
    predictions = knn_predictions(X_train, X_test, y_train, y_test, col_names, 
                                 cont_idx = cont_idx, cat_idx = cat_idx,
                                 k = k, distance = distance_measure)
       
    #Step 3 - Print Accuarcy Report
    accuaracy_report(y_test, predictions)
    
    # Step 4 - Return Predictions
    return predictions