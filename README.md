# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 8 neurons in the first layer and 10 neurons in the second layer, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

<img width="884" height="490" alt="image" src="https://github.com/user-attachments/assets/fecba2c7-07c1-41e9-b420-2ed3513b0912" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Kamesh R R
### Register Number: 212223230095
```python
kamesh_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(kamesh_brain.parameters(),lr=0.001)

def train_model(kamesh_brain,X_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(kamesh_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    kamesh_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

train_model(kamesh_brain,X_train_tensor,Y_train_tensor,criterion,optimizer)


```
## Dataset Information
<img width="306" height="332" alt="image" src="https://github.com/user-attachments/assets/0ffeb0c6-eb97-4b5b-b2d7-48ca9b7d1576" />

## OUTPUT
<img width="1379" height="695" alt="image" src="https://github.com/user-attachments/assets/41ddea84-632d-4e91-bba6-4da86aa2d28a" />

### Training Loss Vs Iteration Plot
<img width="345" height="212" alt="image" src="https://github.com/user-attachments/assets/349aff5b-c700-4f13-93c8-07fb84715bf9" />
<img width="669" height="437" alt="image" src="https://github.com/user-attachments/assets/b3b72a69-259f-4c7f-ad76-0a71c3036143" />



## RESULT
To develop a neural network regression model for the given dataset is excuted sucessfully.
