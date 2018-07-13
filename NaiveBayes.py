import numpy as np
import pandas as pd

def independant_prob(arr, show = 'n'):
    '''This function will return a dictionary of the unique values in an array and counts
    for that value in the form of a python dictionary. It will also determine the independant probability
    and return as dictionary'''
    #get the count dictionary
    cat_dict = dict(zip(np.unique(arr, return_counts= True)[0],np.unique(arr, return_counts= True)[1]))
    prob_dict = {}
    total = sum(cat_dict.values())
    for key in cat_dict.keys():
        prob_dict[key] = cat_dict[key] / total
    
    #Return a Summary of the calcations
    if show == 'y':
        print('Count Dictionary:')
        print(cat_dict,'\n')
        print('Probability Dictionary')
        print(prob_dict)
    
    return prob_dict


def conditional_prob(data_list, cat_cols, uniqueY):
    given_prob_dict = {}
    for data_set in range(len(uniqueY)):
        df = data_list[data_set]
        y_given = {}
        for cat in cat_cols:
        #need to make sure that the input stays strings at conversion
            y_given[str(cat)] = independant_prob(np.array(df[cat]).astype(str))
        #need to keep the float string format
        given_prob_dict[str(float(data_set))] = y_given 
    return given_prob_dict


def nb_preprocessing(df, y):
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    #subset X and Y
    x_names = [x for x in df.columns.tolist() if x not in [y]]
    X = df[x_names]
    
    #store the y column name and identify the y variable
    y_col = y
    y = df[y_col]

    """In order for this fuction to be able to used with multiple classes
    the y values should be label encoded. The label encoding is necessary
    because when the naive bayes algorithm is iteratively making its predictions,
    it  will need to associate the prediction with the position in the list its X subset
    is in. To easily calculate given probabilites, I will subset the X_train by its Y_train category."""
    
    #label encode the y 
    """note the encoded y will still needed to be converted back to a string for the train test split.
    I will need to account for this when i split X_train into subsets."""
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    y = y.astype('float').astype('str')
    
    y_prob = independant_prob(y.astype(str))
    
    # Test Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Turn the datasets back into dataframes
    X_train = pd.DataFrame(data = X_train, columns = x_names).reset_index(drop=True)
    X_test = pd.DataFrame(data = X_test, columns = x_names).reset_index(drop=True)
    y_train = pd.DataFrame(data = y_train, columns = ['survived']).reset_index(drop=True)
    y_test = pd.DataFrame(data = y_test, columns = ['survived']).reset_index(drop=True)

    #--------------------------------------
    #sort unique label encoded Ys. 
    """since the Ys have been label encoded and sorted, the X_train
    dataframes will be stored to the list in the order assosciated with the y
    variable. This will allow us to correspond the y responce with the position in the list
    when generating the predictions."""
    
    uniqueY = sorted(pd.Series(y).astype(float).unique().tolist())
    X_train_list = []
    
    for subset in range(len(uniqueY)):
        subset_df = X_train[y_train.survived == str(float(subset))]
        # Make sure every column is still a string 
        for col in subset_df.columns.tolist():
            subset_df[col] == subset_df[col].astype(str)
        # add subset to a list
        X_train_list.append(subset_df)
                        
    return X_train_list, X_test, y_train, y_test, uniqueY, x_names, le, y_prob


def Naive_Bayes_Predictions(X_test,probabilities, survived_prob):
    testObs = X_test
    classification_list = []
    classification_probs = []
    #iterate over each row
    for row in range(testObs.shape[0]):
        predictions = {}
        #calcuate the probablity for each outcome seperately
        for outcome in survived_prob.keys():
            #iterate over each column in the row
            prob_list = []
            prob_list.append(survived_prob[outcome])
            for col in testObs.columns.tolist():
                #pull probabilites out of 3x nested dictionary. keys are as follows: y response, column, 
                #category in the column
                prob_list.append(probabilities[outcome][col][testObs.loc[row, col]])   
            predictions[outcome] = np.prod(np.array(prob_list)) 
        
        #get the prediction of the greatest number
        pred_key = max(predictions, key=predictions.get)
        classification_list.append(pred_key)
        classification_probs.append(predictions[pred_key] / sum(predictions.values()))
        
    return classification_list, classification_probs

def accuaracy_report(y_test, predictions):
    from sklearn.metrics import classification_report, confusion_matrix
    print('The Accuracy Score is: ',np.array(np.sum(np.equal(predictions, y_test)) / y_test.shape[0]),'\n')
    print('The Classification Report')
    print('-------------------------')
    print(classification_report(y_test, predictions),'\n')
    print('The Confusion Matrix')
    print('-------------------------')
    print(pd.DataFrame(confusion_matrix(y_test, predictions)).apply(lambda x: x / sum(x), axis=1))
    
def Complete_NaiveBayes(df, y):
    #check for categorical variables: must be an object or a category
    for col in df.columns.tolist():
        if df[col].dtype != 'object':
            return 'All columns must have a dytpe of object'
        
    # get test train list
    X_train_list, X_test, y_train, y_test, uniqueY, x_names, label_encoder, y_prob  = nb_preprocessing(df, 'survived')
        
    # generate conditional probabilites     
    probabilities = conditional_prob(X_train_list, x_names, uniqueY)
    
    # generate predictions and the list of probabilites
    classification_list, classification_probs = Naive_Bayes_Predictions(X_test,probabilities, y_prob)
    
    # print the accuracy report
    accuaracy_report(np.array(y_test),np.array(classification_list).reshape(len(classification_list),1))
    
    #return the label encoding categories
    print('\nThe label encoding mappings')
    print('-----------------------------')
    for val in uniqueY:
        print(label_encoder.inverse_transform(int(val)),' : ',val)
        
    #return the classification list and the probabilities    
    print('\nThe Classification Report & Classification Probabilites')
    print('---------------------------------------------------------')       
    return classification_list, classification_probs