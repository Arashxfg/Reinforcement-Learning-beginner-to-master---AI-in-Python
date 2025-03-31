# Function approximators

&nbsp;&nbsp;&nbsp;In the previous section, we saw the first of two strategies that we are going to learn to work with continuous state spaces. In this section we are going to learn the second one that involves using a tool known as function approximators. The strategy that we used in the previous section consisted of taking a continuous state space like the one you see on the screen and transforming it into a discrete state space with a finite number of states. The tabular methods can work with. To do that, we use two different techniques : 
 
![](../Assets/photos/neural%20networks_1.PNG)

&nbsp;&nbsp;&nbsp;The first one, called state aggregation, consisted in cutting the range of valid values for the state into a finite number of intervals. And then we aggregated all the values inside that range into a single state.

![](../Assets/photos/neural%20networks_2.PNG)

&nbsp;&nbsp;&nbsp;Then we learned a second technique called tile coding, which was a generalization of the state aggregation technique in tile coding, we create several state aggregations with different sizes and positions. Thanks to this technique, we obtain better estimates of the values.

![](../Assets/photos/neural%20networks_3.PNG)

&nbsp;&nbsp;&nbsp;However, these two techniques, and in general this strategy, have some weaknesses that make them incapable of handling tasks with moderate complexity. The first of these limitations is that the precision of the estimates will be limited due to the fact that we are aggregating several states into one. Moreover, the computational complexity of this strategy grows very fast as the number of state dimension grows. Imagine that we want to divide each dimension of the state into 20 segments and that the state only has one dimension. In that case, there will be 20 possible states. Now imagine that the state has two dimensions. In that case, there will be 20 by 20. That is 400 possible states. And if there were five dimensions, there would be 20 to the fifth power states. That is 3,200,000 possible states. In some real world control tasks, the state might contain hundreds or thousands of different dimensions. So with this strategy, those problems would be intractable.

![](../Assets/photos/neural%20networks_4.PNG)

&nbsp;&nbsp;&nbsp;So what we need is a precise alternative with limited complexity. Enter function Approximators. Take a look at the graph in blue. You can see the optimal value function of a given control task that we just invented. The specific shape of this function doesn't matter right now. In orange you can see another function that tries to fit as well as possible the blue one. That is, it tries to approximate the blue function. This orange function is the tool that will replace the table of values that we used in the previous sections. Before each point on the x axis was a state, and for each one of them we stored an estimate on our value table. Now, now, instead we are going to have a function with a series of parameters that will decide the shape of that function. During the learning process, we are going to modify the values of those parameters so that the function fits as well as possible. The real value function or Q value function.

![](../Assets/photos/neural%20networks_5.PNG)

&nbsp;&nbsp;&nbsp;Recall that the learning process follows the generalized policy iteration template in which evaluation and improvement of the policy take turns until the optimal value, function and policy are found. The optimal value function is not known in advance, but we arrive at it by continuously evaluating the policy.

![](../Assets/photos/neural%20networks_6.PNG)

&nbsp;&nbsp;&nbsp;As the policy approaches, the optimal policy at the beginning of the learning process, the function that we are going to optimize can have any given shape and it can poorly estimate the optimal values for each state. As the learning process advances, this function will approach the optimal value function and by the end of the process, if the function we have chosen is the right one, it will be as close as possible to the optimal value function as you see in the graph.

![](../Assets/photos/neural%20networks_7.PNG)  
![](../Assets/photos/neural%20networks_8.PNG)  
![](../Assets/photos/neural%20networks_9.PNG)  
![](../Assets/photos/neural%20networks_10.PNG)

&nbsp;&nbsp;&nbsp;As you can see, we can start with any given initial function here represented by F one. It is a function that takes a state as input and contains a number of parameters that specify the shape of the function. Each time we perform. A policy evaluation cycle will modify the parameters on that function and obtain the function. F two. We'll repeat this cycle as we always have until we get to the function f n that will be sufficiently close to the optimal value function.

![](../Assets/photos/neural%20networks_11.PNG)

&nbsp;&nbsp;&nbsp;Now we are going to see two examples of function approximators :  
The first one is called `linear Approximator` and consists of the weighted sum of each dimension of the state by a certain parameter w that will measure the importance of that element here. Calling the function on a specific state produces an estimate of the value of that state, and that value is computed as each of the dimensions of the state times its corresponding parameter.

![](../Assets/photos/neural%20networks_12.PNG)

&nbsp;&nbsp;&nbsp;Let's see it with an example.  
In the graph in blue we have the optimal value function of a certain task and in orange we have our linear approximator. As the state has only one value, it has only one dimension. The linear approximator will have a single W parameter. The best possible fit occurs with this value for the parameter. The best possible fit occurs with this value for the parameter.

![](../Assets/photos/neural%20networks_13.PNG)

&nbsp;&nbsp;&nbsp;Let's look at an approximator that will achieve a better fit. This new approximator is called the polynomial Approximator and it will also be a weighted sum of each element of the state and the exponents of those elements up to a certain value. K In this case. For example, we can include the first element, the first element squared, the first element cubed, etcetera until we reach a given exponent. And the same with the rest of the elements of the state. And each one of these elements will be multiplied by its corresponding parameter.

![](../Assets/photos/neural%20networks_14.PNG)

&nbsp;&nbsp;&nbsp;And this is the fit that we achieve with the two element polynomial estimator. The first element is the value of the state and the second one is the value of the state squared. Each of these elements is weighted by a W parameter, and when the fit is optimal, the values of each parameter are these ones.

![](../Assets/photos/neural%20networks_15.PNG)

&nbsp;&nbsp;&nbsp;These are just two examples of adaptive functions that we can use to represent the value function of a control task. And doing it has some important advantages.  
For example, they require little memory because all we have to store is the vector with the W parameters.

![](../Assets/photos/neural%20networks_16.PNG)

&nbsp;&nbsp;&nbsp;Another advantage is that the same function approximator can be used to approximate different functions. Simply by modifying the W parameters.

![](../Assets/photos/neural%20networks_17.PNG)

&nbsp;&nbsp;&nbsp;Yet another advantage is that if we choose the right function approximator, we can get a very accurate feed.

![](../Assets/photos/neural%20networks_18.PNG)


# Artificial Neural Networks

&nbsp;&nbsp;&nbsp;We're going to learn about a tool capable of approximating functions in a very flexible and accurate way. In fact, nowadays, neural networks achieve the best results in a huge number of function approximation tasks. Neural networks are a very large and complex field, but we are going to focus only on the basics that we'll use in deep reinforcement learning. But what exactly are artificial neural networks?  
Well, they are a computing system inspired by the biological neural networks that constitute our brain. And we study them because this computing system can help us in the task of approximating value functions. Neural networks can approximate functions by adapting a set of parameters that they contain.

![](../Assets/photos/neural%20networks_19.PNG)

&nbsp;&nbsp;&nbsp;Neural networks can be represented using a graph like this one in which each of the nodes is an artificial neuron, which we'll discuss later. Each of these neurons is connected to other ones, and that connection is represented by the edges that you see in this diagram.

![](../Assets/photos/neural%20networks_20.PNG)

&nbsp;&nbsp;&nbsp;Neurons are organized in layers. That is a group of neurons arranged in parallel, and a group of connected layers is called a neural network. The layers have different names depending on their location in the neural network. The first layer, which receives external inputs and propagates them to the next layer, is known as the input layer. The next two layers which are located inside the neural network and neither emit nor receive information from the outside, are called hidden layers. Finally, the final layer that produces the result of applying the neural network to the outside is known as the output layer.

![](../Assets/photos/neural%20networks_21.PNG)

&nbsp;&nbsp;&nbsp;There's a huge number of types of neural network depending on the organization type and connections of its neurons. But we are going to use a basic and very well known type known as feedforward neural networks. This type of neural network is characterized by the fact that the information of each layer propagates only forward to the next layer.

![](../Assets/photos/neural%20networks_22.PNG)

&nbsp;&nbsp;&nbsp;Neurons receive signals from other neurons. In this case, this neuron receives inputs from these two neurons. Then this neuron will process and aggregate those inputs and pass the result to all the other neurons to which it is connected. This signals that the neuron propagates are numerical values. Which the neuron either inhibits or amplifies before passing it to the next layer. If each neuron is connected to all the neurons in the previous and next layer, like for example, this one here, which is connected to all the neurons in the input layer and all the neurons in the output layer. If that is the case for every neuron in that layer, then this layer is known as a fully connected layer.

![](../Assets/photos/neural%20networks_23.PNG)



# Artificial Neurons

&nbsp;&nbsp;&nbsp;In this section, we'll discover what artificial neurons are and how they work. They are a mechanism inspired by the design and functioning of biological neurons. We can say that a neuron is a cell of the nervous system of an animal capable of receiving, processing and sending electrical or chemical signals. To other neurons with which it is connected. These connections are known as synapses. The neuron is connected to other neurons through its dendrites and axons which are the ends of the cell. The dendrites are the connections through which the neuron receives these chemical and electrical signals. The received signals are aggregated and processed in the cell body, and when they exceed a certain intensity, the cell emits electrical pulses through the axons to other cells. We could say that the dendrites are the part of the cell that receives stimuli and the axons are the part of the cell that propagates the signals. Through this mechanism, the neuron can participate in motor and cognitive tasks.

![](../Assets/photos/neural%20networks_24.PNG)

&nbsp;&nbsp;&nbsp;Well, the researchers who developed neural networks observed this mechanism and took it as an inspiration to develop a computational model based on these neurons. And this computational model is called the artificial neuron. The artificial neuron is a mathematical function that takes as input numerical values from other neurons, each of them with a different intensity represented by a parameter w that measures the intensity of the connection between the source and the target neuron. And then the neuron aggregates all those signals obtained from the source neurons and weighted by their intensity, and it applies an activation function to this aggregated value. The result of this activation function is the signal that the neuron will propagate to the neurons of the next layer to which it is connected.

![](../Assets/photos/neural%20networks_25.PNG)

&nbsp;&nbsp;&nbsp;So in summary, an artificial neuron is simply a mathematical function that aggregates and transforms signals received from other neurons and then propagates the transformed value. We can use different activation functions depending on the value that we want to propagate to subsequent layers.

![](../Assets/photos/neural%20networks_26.PNG)

&nbsp;&nbsp;&nbsp;The simplest activation layer is called the identity function. This function doesn't do any changes to the value aggregated from the inputs received from other neurons. It simply propagates that value to the next layer without modifying it. A neural network with this activation function in all layers, learns very efficiently, but is not capable of approximating very complex functions accurately. We are going to use this identity function only in the output layer.

![](../Assets/photos/neural%20networks_27.PNG)

&nbsp;&nbsp;&nbsp;In the inner layers, we'll use nonlinear functions specifically in the hidden layers. That is the layers that have no contact with the outside of the neural network. We'll use an activation function called the rectifier function. This activation function converts the value aggregated from the inputs of other neurons to zero if the value aggregated is negative. And otherwise it doesn't do any modifications to the aggregated value. That is, if the value aggregated from other neurons is five. The rectifier function will leave it as it is. And the neuron will propagate the value five to the next layer. However, if the aggregated value is negative five, then the rectifier function will transform it to zero before propagating it to the next layer. This function is used as an activation function in the hidden layers because it speeds up and facilitates the learning process of the neural network, especially when the network has many hidden layers. In addition, it allows the neural network to approximate more complex functions. A neuron that uses this activation function is called a Relu rectified linear unit.

![](../Assets/photos/neural%20networks_28.PNG)

&nbsp;&nbsp;&nbsp;Another example of a linear activation function is the sigmoid function. This activation function compresses the values aggregated from the other neurons into the range of values between 0 and 1. Before propagating that value. This function is normally used in the last layer of a neural network when we want the output computed by the neural network to be a probability value because of course probabilities are always between 0 and 1. However, this activation function is not used in the inner layers of the neural network because it slows down and hinders the learning process.

![](../Assets/photos/neural%20networks_29.PNG)


# How to represent a Neural Network

&nbsp;&nbsp;&nbsp;Now we know what the neural network is and the structure that it has, but we still need to know how we are going to represent the neural network in our code and how we are going to use it to approximate the value function.  
This is the structure of a three layer neural network. As you can see, the input layer has three neurons. The only hidden layer that this network has has six neurons and the output layer has two. If we want to solve a control task where the state has three dimensions and there's two actions available, this neural network is perfect because we can train it to estimate the Q values of the two actions for each state that we give it as input.

![](../Assets/photos/neural%20networks_30.PNG)

&nbsp;&nbsp;&nbsp;Now let's look at the neural network in parts. Here we have the first part, the input layer and the hidden layer. With their connections, we'll represent this part of the neural network using a vector for the inputs. This x vector here, and this is the vector with the values that enter the neural network through the input layer. In our case, it will be a vector with the values for each dimension of the state. On the other hand, we'll store a matrix representing the connections between the two layers, and each element inside that matrix will represent the intensity of the connection between a neuron from the input layer and a neuron from the hidden layer. The first index of each element represents the neuron from the input layer from which the connection comes out, and the second index for each element represents the destination neuron in the hidden layer. Thus, the first column of the connection matrix represents the intensity of the connections between all the neurons in the input layer with the first neuron from the hidden layer. The second column represents the connections to the second neuron from the hidden layer and so on and so forth with all the columns. The matrix of connections will have as many rows as there are neurons in the input layer and as many columns as there are neurons in the hidden layer. Finally, we'll have a vector called H that will hold the result of processing the inputs for each neuron. This vector is obtained by calculating the product between the vector x and the matrix W1. The first element of this vector contains the aggregated input multiplied by the intensity of its connections for the first neuron. An activation function is applied to this aggregated input. The second value in this vector corresponds to the result of the second neuron processing the inputs from the previous layer. Of course, multiply it by the intensity of each input and then applying the activation function. And so on and so forth. The vector will have as many elements as there are neurons in the hidden layer, and it contains the values that this layer of neurons will propagate to the next layer.

![](../Assets/photos/neural%20networks_31.PNG)

&nbsp;&nbsp;&nbsp;Now let's look at the second part of the neural network, where we will connect the hidden layer to the output layer.  
Since there are six neurons in the hidden layer and two in the output layer, the matrix of connections will have six rows and two columns. The first column represents the intensity of the connections between the neurons in the hidden layer and the first neuron on the output layer. The second column will represent the intensity of the connections between the neurons in the hidden layer and the second output neuron. Therefore, the vector of inputs of the output layer will now be the h vector which will obtain processing the inputs which are the values that the hidden layer will propagate to the output layer. The output vector will be the result of aggregating and processing the inputs by the output layer, and in this case it will consist of a vector with two values because we have two neurons. This vector is obtained by multiplying the vector h by the matrix of connections w two.

![](../Assets/photos/neural%20networks_32.PNG)

&nbsp;&nbsp;&nbsp;Now let's look at the whole neural network and follow the path of the values that enter as inputs until they leave as outputs. The vector x enters through the input layer and is multiplied by the matrix of connections W1. To the result, which is a vector of six elements. We apply the activation function of the hidden layer and thus we obtain the vector h as a result. Then the values of this vector are propagated to the output layer by multiplying them by the matrix of connections w. And then the activation function of the output layer will be applied to the result. And those are the outputs that come out of the neural network.

![](../Assets/photos/neural%20networks_33.PNG)

&nbsp;&nbsp;&nbsp;If you notice, the neural network is nothing more than a function, a function with parameters w that can be adjusted to modify the outputs that they produce. Therefore, neural networks can be used to approximate functions and as function approximators, they are very flexible and very powerful. The more hidden layers that the neural network has, the more complex functions it can approximate accurately, although it will require more computation and memory in the learning process. In our coding exercises, we'll use the PyTorch library to build our neural networks. And we'll do it using a class called Sequential that allows us to build neural networks that apply operations to the inputs in a sequential manner that our inputs come through the input layer. Then they are multiplied by the connection matrix. Then the activation function from the hidden layer will be applied.

![](../Assets/photos/neural%20networks_34.PNG)



# Stochastic Gradient Descent

&nbsp;&nbsp;&nbsp;We are going to see how to minimize the cost function. We are going to do it with an algorithm called stochastic gradient descent. This algorithm works as follows.  
First, we are going to estimate the value of the cost function using the rewards obtained from the environment and the neural network estimates. To perform that algorithm will do the following steps. First, we are going to estimate the value of the cost function using the rewards obtained from the environment and the neural network estimates. Once we have the estimates of this function, we are going to find its gradient vector.

![](../Assets/photos/neural%20networks_35.PNG)

&nbsp;&nbsp;&nbsp;The gradient vector is obviously a vector where each element is the partial derivative of the estimated cost function with respect to each of the parameters of the neural network. What does this mean?  
Well, the gradient is a vector that points in the direction in which each parameter of the neural network must be modified so that the cost function grows as much as possible. Don't worry, because we are not going to compute this vector by hand. We'll use the PyTorch library. That will do it for us.  
The gradient vector is computed by an algorithm called back propagation. We are not going to cover it in detail because it's not closely related to reinforcement learning.

![](../Assets/photos/neural%20networks_36.PNG)

&nbsp;&nbsp;&nbsp;Once we have this gradient vector, we are going to carry out the update rule. As you can see, it's very similar to the update rule that we used with tabular methods. But instead of updating values stored in a table, we are updating the parameters of the neural network. And these parameters represent the intensity of the connections between the neurons in this update rule to the previous values. For the parameters, we subtract a percentage alpha times the gradient of the cost function. What this means is that we'll move the parameters in the opposite direction of the direction of maximum growth of the cost function. That is, we'll find the parameters that will produce the lowest values for the cost function by going in the opposite direction of the estimated direction of maximum growth.

![](../Assets/photos/neural%20networks_37.PNG)

&nbsp;&nbsp;&nbsp;Now let's see visually how stochastic gradient descent works. Let's see it with the two cost function examples that we saw earlier. Remember that this is a simplified example of the cost function in which there are only two parameters W1 and W2, and the vertical axis shows the value of the cost function for each possible combination of W1 and W2. In the graph on the right, we see the same cost function from above to see it better. The arrows that you see here are the negative of the gradient of the cost function. That is, they are arrows that point in the direction of maximum decrease of the cost function. Note that they are the gradients of the real cost function, not of the estimates that will produce based on the experience that the agent collects. Now let's see how the gradient descent algorithm works. When we initialize the neural network, the parameters will have a random value. In this case, this here is the initial value of the parameters based on the experience that the agent collects. Interacting with the environment will compute an estimate of the cost function and the gradient vector. It's negative, which is what we're interested in, points in this direction. It is a relatively good but not perfect approximation to the true gradient of the cost function. Based on the alpha parameter. We'll take a step in this direction and the parameters of the neural network will change to these values. We'll repeat this process until the estimate of the cost function stops decreasing, and at that moment we'll assume that we've reached the optimal or close to optimal values for the parameters.

![](../Assets/photos/neural%20networks_38.PNG)

&nbsp;&nbsp;&nbsp;Let's see now with a more complex cost function, the cost functions that our neural network will generate will be much more complex than the ones you see here, since they have a huge number of parameters and represent complex phenomena. Let's repeat the gradient descent process. Let's imagine that the initial values for the neural network are these ones, W2 and W1. Then starting at this point, we'll use samples of experience to compute the estimate of the cost function. The estimated gradient and then we'll follow the negative of the gradient by a percentage alpha and we'll move the parameters of the neural network here. We'll continue to do the same until we reach a point that is a local minimum. Now imagine that the initial point is this one. Then following the gradient descent process, we'll arrive at this local minimum. As you can see, the initial values of the neural network have an influence on the learning process. Let's see a third example. If the initial values of the neural network are these, then the stochastic gradient descent algorithm will take us to this local minimum, which is in fact the global minimum.

![](../Assets/photos/neural%20networks_39.PNG)  
![](../Assets/photos/neural%20networks_40.PNG)  
![](../Assets/photos/neural%20networks_41.PNG)  
![](../Assets/photos/neural%20networks_42.PNG)



# Neural Network optimization

&nbsp;&nbsp;&nbsp;We are going to learn how to optimize our neural network to approximate the values of the states. What we want to achieve is that by passing a state as input to the neural network through the input layer, the neural network will produce as output accurate estimates of the Q values for each action in that state.

![](../Assets/photos/neural%20networks_43.PNG)

&nbsp;&nbsp;&nbsp;To achieve that, we have to find the optimal values for the W parameters of the neural network, which, as you know, measure the strength of the connections between neurons.

![](../Assets/photos/neural%20networks_44.PNG)

&nbsp;&nbsp;&nbsp;The optimal values for W are those that minimize the errors that the neural network makes when estimating the Q values. But there are many ways to define the error of an estimation.

![](../Assets/photos/neural%20networks_45.PNG)

&nbsp;&nbsp;&nbsp;Two of them are the ones you see on the screen, but there are many more. All of them perfectly valid. The one on the left called Mean Absolute Error computes the average of the differences between the true values and the estimated values. The expression on the right known as the mean squared error finds the mean of the square of those differences. Depending on the error expression that we choose to minimize, we'll obtain a different solution for the W parameters, and each solution has different characteristics.

![](../Assets/photos/neural%20networks_46.PNG)

&nbsp;&nbsp;&nbsp;We are going to choose to minimize the mean squared error

![](../Assets/photos/neural%20networks_47.PNG)

&nbsp;&nbsp;&nbsp;And we'll do it based on the experience that will sample from the environment. The target value for a specific state and action is the expression that we consider the true value of that action in that state and it is represented by this expression. As you can see, it consists of the reward obtained after taking the action, plus a discounted estimate of the Q value of the next action taken in the next state reached. And of course, that estimate is produced by the neural network. And the other term in this expression is the estimated value produced by the neural network for that state. And action.

![](../Assets/photos/neural%20networks_48.PNG)

&nbsp;&nbsp;&nbsp;Now let's visualize a very simplified example of a cost function. This cost function depends on only two parameters instead of several million, as is normal working with neural networks. But by using only two parameters, we'll be able to see the loss function visually on the vertical axis. We see the value that the cost function takes for each combination of parameters. Our goal is to find the value for the parameters for which this cost function has the lowest value. In this very simplified example, the cost function reaches its minimum value when both the parameters W one and W two have the value zero.

![](../Assets/photos/neural%20networks_49.PNG)

&nbsp;&nbsp;&nbsp;Now let's look at another example, which is still a great simplification of what the true cost functions look like, but it's a little bit more complex than the previous one. This cost function has several local minima, like for example, this one over here, this one, this one and this one, but only a single global minimum. The minimum point in the cost function. In this case, it would be ideal to find the values for the parameters that produce the global minimum for the cost function. But iterative algorithms like the ones we are going to use only guarantee to find a local minimum. That is one of these. However, as a general rule, this will be enough because the estimates produced by the neural network with these parameters will be good enough to solve the control tasks.

![](../Assets/photos/neural%20networks_50.PNG)



































































































































































































































