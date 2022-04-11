
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
from sklearn import metrics
import skimage.measure
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import pandas as pd
from math import exp
from tensorflow.keras.layers import Input, Lambda, Dense, concatenate
from tensorflow.keras.models import Model
import math
import datetime
from tensorflow import keras

from tensorflow.python.keras.backend import learning_phase 
from sklearn.metrics import confusion_matrix

start_time = time.time()
#Import Dataset Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names =  ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Reshape
x_train = x_train.reshape(60000, 784).astype("float32") / 255.0
x_test = x_test.reshape(10000, 784).astype("float32") / 255.0

print("Shape of x_train: {}".format(x_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print()
print("Shape of x_test: {}".format(x_test.shape))
print("Shape of y_test: {}".format(y_test.shape))

#Print classes and an example of train images
class_names_fordataframe = {"Labels" : [0,1,2,3,4,5,6,7,8,9],"Classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]}
result=pd.DataFrame(class_names_fordataframe)
print(result)

print(x_train[0][0])
print(y_train[0])

#Definition of parameters
n=3
m=3
k=10
#q=2*n*m +k
#d=2*n+1


############################################################################
#LAMBDA 
print("--------------------------MCLNN WITH FUNCTIONAL API----------------")

def create_Mask_Matrix(rows ,columns , bandwith, overlap):
	my_array=[]
	for i in range(rows):
		for j in range(columns):
			my_array.append(0)
			
	space=rows - bandwith + bandwith - overlap  #734 #6

	for i in range(0,len(my_array),bandwith+space):
		if i+bandwith> rows*columns-1:
			for j in range(i,rows*columns):
				my_array[j]=1
		else:
			for j in range(i,i+bandwith):
				my_array[j]=1
			
			

	A=np.array(my_array)
	B=A.reshape(columns,rows) 	 #!!!! vazw anapoda ta rows , columns apo oti ta thelw
	Mask_Matrix=B.T
	return Mask_Matrix

def create_Mask_Matrix_as_square(rows ,columns):
    
    my_array=[]
    for i in range(rows):
        for j in range(columns):
            my_array.append(0)

    start=int((0.2*rows)/2 *columns +(0.2*columns)/2)
    stop= int((rows - (0.2*rows)/2 ) *columns +columns - (0.2*columns)/2)
    step= int(0.2*columns)

    A=np.array(my_array)
    B=A.reshape(columns,rows) 	 
    Mask_Matrix=B.T
    

    for i in range(int((0.2*rows/2)), int(rows-(0.2*rows/2))):   
        for j in range(int((0.2*columns)/2),int(columns - (0.2*columns)/2 )):
            Mask_Matrix[i][j]=1
   
    return Mask_Matrix




def find_neuros_for_lambda_function(matrix, rows, columns):
	temp_list=[]
	for i in range(columns):
		temp_list.append(i)

	dict_for_neurons={}
	for i in temp_list:
		dict_for_neurons[i] = None
		
	for j in range(columns):
		list_for_neurons=[]
		for i in range(rows):    	
			if matrix[i][j]==1:
				list_for_neurons.append(i)
		
		dict_for_neurons[j]=list_for_neurons

	return dict_for_neurons




#INPUT LAYER --> FIRST HIDDEN LAYER 
print("-------------------INPUT LAYER --> FIRST HIDDEN LAYER ------------------------------------ ")

print("MASK MATRIX IS: ")
print()
#Mask_matrix=create_Mask_Matrix(784,22,400,-10) #(784,22,20,-5)   :  Bandwtih=20 , Overlap=-5 as best values from plots in papers
Mask_matrix=create_Mask_Matrix_as_square(784,22)
print(Mask_matrix)

print("EACH 22 WILL HAVE TO BE CONNECTED DENSILY WITH THE CORRESPONDING LIST : ")
print()
input_to_first_hidden_layer=find_neuros_for_lambda_function(Mask_matrix,784,22)  #(Mask_matrix,784, 22)
print(input_to_first_hidden_layer)


inp = Input(shape=(784,))
inp1 = Lambda(lambda x: x , input_to_first_hidden_layer[0])(inp)
inp2 = Lambda(lambda x: x , input_to_first_hidden_layer[1])(inp)
inp3 = Lambda(lambda x: x , input_to_first_hidden_layer[2])(inp)
inp4 = Lambda(lambda x: x , input_to_first_hidden_layer[3])(inp)
inp5 = Lambda(lambda x: x , input_to_first_hidden_layer[4])(inp)
inp6 = Lambda(lambda x: x , input_to_first_hidden_layer[5])(inp)
inp7 = Lambda(lambda x: x , input_to_first_hidden_layer[6])(inp)
inp8 = Lambda(lambda x: x , input_to_first_hidden_layer[7])(inp)
inp9 = Lambda(lambda x: x , input_to_first_hidden_layer[8])(inp)
inp10 = Lambda(lambda x: x , input_to_first_hidden_layer[9])(inp)
inp11 = Lambda(lambda x: x , input_to_first_hidden_layer[10])(inp)
inp12 = Lambda(lambda x: x , input_to_first_hidden_layer[11])(inp)
inp13 = Lambda(lambda x: x , input_to_first_hidden_layer[12])(inp)
inp14 = Lambda(lambda x: x , input_to_first_hidden_layer[13])(inp)
inp15 = Lambda(lambda x: x , input_to_first_hidden_layer[14])(inp)
inp16 = Lambda(lambda x: x , input_to_first_hidden_layer[15])(inp)
inp17 = Lambda(lambda x: x , input_to_first_hidden_layer[16])(inp)
inp18 = Lambda(lambda x: x , input_to_first_hidden_layer[17])(inp)
inp19 = Lambda(lambda x: x , input_to_first_hidden_layer[18])(inp)
inp20 = Lambda(lambda x: x , input_to_first_hidden_layer[19])(inp)
inp21 = Lambda(lambda x: x , input_to_first_hidden_layer[20])(inp)
inp22 = Lambda(lambda x: x , input_to_first_hidden_layer[21])(inp)

first_layer1 = Dense(22, activation='relu')(inp1)
first_layer2 = Dense(22, activation='relu')(inp2)  
first_layer3 = Dense(22, activation='relu')(inp3) 
first_layer4 = Dense(22, activation='relu')(inp4) 
first_layer5 = Dense(22, activation='relu')(inp5) 
first_layer6 = Dense(22, activation='relu')(inp6) 
first_layer7 = Dense(22, activation='relu')(inp7) 
first_layer8 = Dense(22, activation='relu')(inp8) 
first_layer9 = Dense(22, activation='relu')(inp9) 
first_layer10 = Dense(22, activation='relu')(inp10)
first_layer11 = Dense(22, activation='relu')(inp11) 
first_layer12 = Dense(22, activation='relu')(inp12) 
first_layer13 = Dense(22, activation='relu')(inp13) 
first_layer14 = Dense(22, activation='relu')(inp14) 
first_layer15 = Dense(22, activation='relu')(inp15)
first_layer16 = Dense(22, activation='relu')(inp16) 
first_layer17 = Dense(22, activation='relu')(inp17)
first_layer18 = Dense(22, activation='relu')(inp18) 
first_layer19 = Dense(22, activation='relu')(inp19) 
first_layer20 = Dense(22, activation='relu')(inp20)
first_layer21 = Dense(22, activation='relu')(inp21)
first_layer22 = Dense(22, activation='relu')(inp22)

first_layer=concatenate([first_layer1 , first_layer2 , first_layer3 , first_layer4 , first_layer5 , first_layer6 , first_layer7 , first_layer8 , first_layer9 , first_layer10 ,
                        first_layer11 , first_layer12 , first_layer13 , first_layer14 , first_layer15 , first_layer16 , first_layer17 , first_layer18 , first_layer19 , first_layer20 , 
                        first_layer21 , first_layer22])


#FIRST HIDDEN LAYER  --> SECOND HIDDEN LAYER 
print("-------------------FIRST HIDDEN LAYER  --> SECOND HIDDEN LAYER ------------------------------------ ")

print("MASK MATRIX IS: ")
print()
#Mask_matrix=create_Mask_Matrix(484,16,400,-10) #Bandwtih=20 , Overlap=-5 as best values from plots in papers
Mask_matrix=create_Mask_Matrix_as_square(484,16)
print(Mask_matrix)

print("EACH 16 WILL HAVE TO BE CONNECTED DENSILY WITH THE CORRESPONDING LIST : ")
print()
input_to_first_hidden_layer=find_neuros_for_lambda_function(Mask_matrix,484,16)  
print(input_to_first_hidden_layer)

inp1_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[0])(first_layer)   
inp2_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[1])(first_layer)
inp3_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[2])(first_layer)
inp4_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[3])(first_layer)
inp5_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[4])(first_layer)
inp6_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[5])(first_layer)
inp7_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[6])(first_layer)
inp8_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[7])(first_layer)
inp9_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[8])(first_layer)
inp10_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[9])(first_layer)
inp11_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[10])(first_layer)
inp12_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[11])(first_layer)
inp13_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[12])(first_layer)
inp14_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[13])(first_layer)
inp15_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[14])(first_layer)
inp16_first_layer = Lambda(lambda x: x , input_to_first_hidden_layer[15])(first_layer)

second_layer1 = Dense(16, activation='relu')(inp1_first_layer)
second_layer2 = Dense(16, activation='relu')(inp2_first_layer)
second_layer3 = Dense(16, activation='relu')(inp3_first_layer)
second_layer4 = Dense(16, activation='relu')(inp4_first_layer)
second_layer5 = Dense(16, activation='relu')(inp5_first_layer)
second_layer6 = Dense(16, activation='relu')(inp6_first_layer)
second_layer7 = Dense(16, activation='relu')(inp7_first_layer)
second_layer8 = Dense(16, activation='relu')(inp8_first_layer)
second_layer9 = Dense(16, activation='relu')(inp9_first_layer)
second_layer10 = Dense(16, activation='relu')(inp10_first_layer)
second_layer11 = Dense(16, activation='relu')(inp11_first_layer)
second_layer12 = Dense(16, activation='relu')(inp12_first_layer)
second_layer13 = Dense(16, activation='relu')(inp13_first_layer)
second_layer14 = Dense(16, activation='relu')(inp14_first_layer)
second_layer15= Dense(16, activation='relu')(inp15_first_layer)
second_layer16 = Dense(16, activation='relu')(inp16_first_layer)

second_layer =concatenate([second_layer1 , second_layer2 , second_layer3 , second_layer4 , second_layer5 , second_layer6 , second_layer7 , second_layer8 , second_layer9 , second_layer10 , 
                            second_layer11 , second_layer12 , second_layer13 , second_layer14 , second_layer15 , second_layer16])

#SECOND HIDDEN LAYER  --> OUTPUT LAYER 
print("-------------------SECOND HIDDEN LAYER  --> OUTPUT LAYER  ------------------------------------ ")

print("MASK MATRIX IS: ")
print()
#Mask_matrix=create_Mask_Matrix(256,10,400,-10) #Bandwtih=20 , Overlap=-5 as best values from plots in papers
Mask_matrix=create_Mask_Matrix_as_square(256,10)
print(Mask_matrix)

print("EACH 10 WILL HAVE TO BE CONNECTED DENSILY WITH THE CORRESPONDING LIST : ")
print()
input_to_first_hidden_layer=find_neuros_for_lambda_function(Mask_matrix,256,10)  
print(input_to_first_hidden_layer)

inp1_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[0])(second_layer)   
inp2_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[1])(second_layer) 
inp3_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[2])(second_layer) 
inp4_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[3])(second_layer) 
inp5_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[4])(second_layer) 
inp6_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[5])(second_layer) 
inp7_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[6])(second_layer) 
inp8_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[7])(second_layer) 
inp9_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[8])(second_layer) 
inp10_second_layer = Lambda(lambda x: x , input_to_first_hidden_layer[9])(second_layer) 

out_layer1 = Dense(10, activation='relu')(inp1_second_layer)
out_layer2 = Dense(10, activation='relu')(inp2_second_layer)
out_layer3 = Dense(10, activation='relu')(inp3_second_layer)
out_layer4 = Dense(10, activation='relu')(inp4_second_layer)
out_layer5 = Dense(10, activation='relu')(inp5_second_layer)
out_layer6 = Dense(10, activation='relu')(inp6_second_layer)
out_layer7 = Dense(10, activation='relu')(inp7_second_layer)
out_layer8 = Dense(10, activation='relu')(inp8_second_layer)
out_layer9 = Dense(10, activation='relu')(inp9_second_layer)
out_layer10 = Dense(10, activation='relu')(inp10_second_layer)

out_layer= concatenate([out_layer1 , out_layer2 , out_layer3 , out_layer4 , out_layer5 , out_layer6 , out_layer7 , out_layer8 , out_layer9 , out_layer10])

final_output = Dense(10, activation='softmax')(out_layer)

model = Model(inp, final_output)
print(model.summary())

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer = opt,
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=100 , callbacks=[tensorboard_callback])

#print(model.weights)
#print(len(model.weights))

#Print accuracy
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)
print()

#Make Predictions

print("-----------------THIS IS AN EXAMPLE OF FIRST 10 IMAGES OF TEST SET---------------------")
predictions = model.predict(x_test)   

y_pred=[]
temp=0
num_of_predicted_images=10000
for i in range(num_of_predicted_images):
    print(predictions[i])
    predicted_label = np.argmax(predictions[i])
    t1=math.modf(predicted_label)
    predicted_label=int(t1[1])
    if predicted_label==y_test[i]:
        print(" Test Image " + str(i) + ": CORRECT PREDICTION . CLASS :" + str(predicted_label ))
        print()
        temp+=1
    else:
        print(" Test Image " + str(i) + ": WRONG PREDICTION. PREDICTED CLASS :" + str(predicted_label) + " AND CORRECT CLASS : " + str(y_test[i]))
        print()
    y_pred.append(predicted_label)

percentage_of_predictions=temp/num_of_predicted_images *100
print("Percentage of predictions : " +str(percentage_of_predictions)+ " %")
print('\nTest loss:', test_loss)
print('\nTest accuracy:'+ str(test_acc*100) + " %")
print()

print("CONFUSION MATRIX : ")
print()
conf_matrix=confusion_matrix(y_test, y_pred, labels=[0 ,1 ,2 ,3 ,4 ,5 , 6, 7, 8, 9])
print(conf_matrix)
print()

time1=(time.time() - start_time)/60
print("--- %s seconds ---" % (time1))

