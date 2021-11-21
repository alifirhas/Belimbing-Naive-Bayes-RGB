from random import randrange

def cross_validation_split(dataset: list, folds: int):
    """split dat menjadi beberapa folds

    Args:
        dataset (list): data yang akan diproses
        folds (int): jumlah folds yang akan dibuat

    Returns:
        list: list data yang telah dibagi
    """
    dataset_split = []
    df_copy = dataset
    fold_size = int(df_copy.shape[0] / folds)
    
    # for loop to save each fold
    for i in range(folds):
        fold = []
        # while loop to add elements to the folds
        while len(fold) < fold_size:
            # select a random element
            r = randrange(df_copy.shape[0])
            # determine the index of this element 
            index = df_copy.index[r]
            # save the randomly selected line 
            fold.append(df_copy.loc[index].values.tolist())
            # delete the randomly selected line from
            # dataframe not to select again
            df_copy = df_copy.drop(index)
        # save the fold     
        dataset_split.append(np.asarray(fold))
        
    return dataset_split

def kfoldCV(dataset, f=5, k=5, model="logistic"):
    data=cross_validation_split(dataset,f)
    result=[]
    # determine training and test sets 
    for i in range(f):
        r = list(range(f))
        r.pop(i)
        for j in r :
            if j == r[0]:
                cv = data[j]
            else:    
                cv=np.concatenate((cv,data[j]), axis=0)
        
        # apply the selected model
        # default is logistic regression
        if model == "logistic":
            # default: alpha=0.1, num_iter=30000
            # if you change alpha or num_iter, adjust the below line         
            c = logistic(cv[:,0:4],cv[:,4],data[i][:,0:4])
            test = c['Y_prediction_test']
        elif model == "knn":
            test = kNN(cv[:,0:4],cv[:,4],data[i][:,0:4],k)
            
        # calculate accuracy    
        acc=(test == data[i][:,4]).sum()
        result.append(acc/len(test))
        
    return result