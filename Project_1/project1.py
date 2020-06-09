import numpy as np
import pandas as pd

data = pd.read_csv('iris.data', names=['Sepal Length','Sepal Width','Petal Length','Petal Width','Species']) #Importing data from Iris dataset with specific column names.
species_values = data['Species'].values #extracting species/class names  
species_list = []

for i in species_values:
    if(i =='Iris-setosa'):
        species_list.append(1)
    elif(i =='Iris-versicolor'):
        species_list.append(2)
    elif(i == 'Iris-virginica'):
        species_list.append(3)
    else:
        break

#creating X and Y matrices. 
X_actual = np.matrix([np.ones(150), data['Sepal Length'].values, data['Sepal Width'].values, data['Petal Length'].values, data['Petal Width'].values]).T #Creating X matrix with 1's at the first position and transposing the matrix
Y_actual = np.matrix(species_list).T

#Calculating beta value for the entire dataset using the formula B = ((X'X)^-1)X'Y
X_trans_X = (np.dot(X_actual.T, X_actual)).I
X_trans_Y = np.dot(X_actual.T, Y_actual)
Beta = np.dot(X_trans_X, X_trans_Y) #Beta value for the entire dataset

print("The beta values for the entire dataset is: \n")
print(Beta,"\n")

unseen_input = np.matrix([1,2.1,3.2,1.2,1.1]) #These values can be changes, except for the first value, which remains constant.
Y_cap=int(np.dot(unseen_input,Beta).T.round())

if(Y_cap <= 1):
    print("The species is: Iris-setosa for the values:",unseen_input)
elif(Y_cap == 2):
    print("The species is: Iris-versicolor for the values:",unseen_input)
elif(Y_cap >= 3):
    print("The species is: Iris-virginica for the values:",unseen_input)
print("\n")

K_Fold_X_Train, K_Fold_Y_Train, K_Fold_X_Test, K_Fold_Y_Test, Non_Zero_Error_List  = [], [], [], [], []

#The value of K can be changed so as to have equal folds of the dataset i.e., 2,3,5,6,10
K_Fold_X = np.split(X_actual, 5) #Splitting the X matrix into 'k' folds
K_Fold_Y = np.split(Y_actual, 5) #Splitting the Y matrix into 'k' folds

#print(K_Fold_X)
#print(K_Fold_Y)

print("The actual values of Y are: ")
print(Y_actual.T, "\n")
print("___________________________________________________________________________________\n")

#Implementation of K-Fold Cross Validation
for i in range(0, len(K_Fold_X)):
    for j in range(0,len(K_Fold_X)):
        if(j == i):
            K_Fold_X_Test.append(K_Fold_X[j])
            K_Fold_Y_Test.append(K_Fold_Y[j])
        else:
            K_Fold_X_Train.append(K_Fold_X[j])
            K_Fold_Y_Train.append(K_Fold_Y[j])
            
    X_Train = np.matrix(np.concatenate(np.array(K_Fold_X_Train))) #X training matrix for K-Fold Cross Validation
    #print(X_Train)
    Y_Train = np.matrix(np.concatenate(np.array(K_Fold_Y_Train))) #Y training matrix for K-Fold Cross Validation
    #print(Y_Train)
    X_Test = np.matrix(np.array(K_Fold_X_Test)) 
    #print(X_Test)
    Y_Test = np.matrix(np.array(K_Fold_Y_Test))
    
    Beta_T = np.dot((np.dot(X_Train.T,X_Train)).I,(np.dot(X_Train.T,Y_Train))) #Calculating beta value for the trained dataset
    print("Fold ",i+1,":\n")
    print("Beta values for",i+1, "fold is: ")
    print(Beta_T, "\n")
    
    Y_T_pred = np.dot(X_Test, Beta_T).T.round() #Predicting value of Y using testing dataset and Beta value calculated using train
    
    print("The test values of Y are: ")
    print(Y_Test, "\n")
    print("The predicted values of Y are: ")
    print(Y_T_pred, "\n")
    
    Error = (Y_Test - Y_T_pred).astype(int) #Calculating error using the formula: E = Y (actual) - Y^(predicted) and converting float value to integer
    
    print("The difference in values of Y are: ")
    print(Error, "\n")
    
    Non_Zero_Error_List.append(1-(np.count_nonzero(Error)/len(K_Fold_X[0]))) #The non-zero elements in the list indicate the ones which have error whereas the one with 0 difference shows there is no error in predicting.
    
    print("Average error for fold", i+1 , "is:", 1-(np.count_nonzero(Error)/len(K_Fold_X[0])), "\n")
    print("____________________________________________________________________________________\n")
   
    K_Fold_X_Train.clear()
    K_Fold_X_Test.clear()
    K_Fold_Y_Train.clear()
    K_Fold_Y_Test.clear()
    Error = np.array([])
    
accuracy = sum(Non_Zero_Error_List)/len(Non_Zero_Error_List) * 100    
print("\nAccuracy: ", accuracy,"%")