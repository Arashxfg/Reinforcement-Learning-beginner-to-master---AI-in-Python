# Classic control tasks

https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Classic_Control_Introduction.ipynb#scrollTo=7PnV4kmDuEaS&uniqifier=2


# Working with continuous state spaces

&nbsp;&nbsp;&nbsp;In this section, we're going to learn how to solve control tasks that have continuous state spaces. So far, we've learned the basic reinforcement learning algorithms using a single control task. The five by five maze. In this task, the state is the position of the agent defined by the row and the column index. Since there are five possible values for the row and five possible values for the column, in total, there's 25 possible combinations of row and column. That is 25 possible states.

![](../Assets/photos/Continuous%20state%20spaces_1.PNG)

&nbsp;&nbsp;&nbsp;Now look at this control task. The agent, which is the golf player, must hit the ball with a club and into the hole. The further it throws the ball, the more inaccurate the shot will be. This task has the following state value function optimal state value function on the x axis. You can see the different values that the state can take. The state is the position of the ball on this. Straight line and the valid values are between negative ten and positive ten. On the y axis, you can see the optimal value for each state possible. As you can see, the closer a state is to the hole, the higher its value. That is so because the agent gets rewarded when it puts the ball inside the hole. 

![](../Assets/photos/Continuous%20state%20spaces_2.PNG)

&nbsp;&nbsp;&nbsp;But now we have a problem because we can't solve this task using the methods that we've seen so far. Because these methods use a value table in which they store the value of the states or the Q_values.

![](../Assets/photos/Continuous%20state%20spaces_3.PNG)

&nbsp;&nbsp;&nbsp;The problem is that control tasks with continuous state spaces have infinitely many possible states. If we were to store an entry in the table for each state possible, we would need a table with infinite memory.

![](../Assets/photos/Continuous%20state%20spaces_4.PNG)

&nbsp;&nbsp;&nbsp;So what options do we have? Well, in general, we have two possible solutions.  
The first one is to transform the state into a format that we can work with. This is the solution that we are going to explore in this section. In the following sections, we're going to explore the second of these options. Which consists of using algorithms capable of dealing with continuous state spaces.

![](../Assets/photos/Continuous%20state%20spaces_5.PNG)

&nbsp;&nbsp;&nbsp;The first option. Modifying the states to make them usable with our classic algorithms means converting a continuous range of values into a finite set of states. To achieve this in this section, we're going to develop two techniques. State aggregation and tile coding.

![](../Assets/photos/Continuous%20state%20spaces_6.PNG)



# State aggregation

&nbsp;&nbsp;&nbsp;We are going to learn  about the first method for extending classical reinforcement learning algorithms to continuous state bases, known as `state aggregation`.  
Let's go back to the example of the golf player that we saw in the previous section. Here is the optimal state value function fot this task. On the X axis you can see the state base and on the y axis the value of each of those state.  
Applying `sate aggregation ` to this task means grouping the values within a range into a single state. For example we would take all the values between -10 amd -8 and we would represent all this values with a single state. Then we choose all the values between -8 and -6 and we would represent all those values with another state, and so on and so far. That way we can create an entry in the value table for each one aggregation state and then use any classical reinforcement learning algorithm to solve it.

![](../Assets/photos/Continuous%20state%20spaces_7.PNG)

&nbsp;&nbsp;&nbsp;By doing that the table will store an estimate of the value of each aggregation state. That means that different states in the original state base will have the same estimate ones we aggregate the state. This puts a limit on a precision of our estimate because state that where originally different will share the same value in our table os estimate. We are responsible for choosing the number of the states in which we want to group the original range, the more state we keep the smaller the ranges of the values their group and the more accurate that estimate of the value function. But that will also make it harder to solve the task because there will be more state to evaluate. This translate into more execution time and memory requirement.

![](../Assets/photos/Continuous%20state%20spaces_8.PNG)

&nbsp;&nbsp;&nbsp;Using this technics we go from working within an infinite number of states to working with only 10.

![](../Assets/photos/Continuous%20state%20spaces_9.PNG)

&nbsp;&nbsp;&nbsp;And this technic can also applied to more complex tasks in which state has several dimensions, each of them with a continuous bally range.  
This is the case of the `Mountain Car` task. This is the task where the car has to gain momentum to reach the flag on the mountain to its right. The state consist of the vector of 2 values with the position of the car and its velocity. On the right you can see the state base for this task. Represented by the all valid combinations of the position and the velocity. On the x axis you can see all the valid values for the position of the car, ranging from -1.2 to 0.6. On the y axis you can see all the valid values for the velocity of the car, ranging from the -0.07 to 0.07.

![](../Assets/photos/Continuous%20state%20spaces_10.PNG)

&nbsp;&nbsp;&nbsp;To aggregate this 2 dimensional state base we split each dimension of the state into the number of the intervals that we want and generate a grid like this one. All the state inside a tile of this grid are aggregated into a single state using this technic we go from having infinite states to having only 25. And this process can be generalize to continuos state bases of multiple dimensions

![](../Assets/photos/Continuous%20state%20spaces_11.PNG)

https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_7_continuous_observation_spaces_complete.ipynb#scrollTo=c0c3GADpBSou



# Tile coding

&nbsp;&nbsp;&nbsp;We're going to learn the second technique that will allow us to work with continuous state spaces called `tile coding`. Let's go back to the golf player task. As you know, this is the optimal value function with the state aggregation method. We group the valid positions into a set of states that we work with. However, when we do this, we are accepting some loss of precision because we use the same value estimate for all the states within a range of values. For example, here the optimal state is larger than the estimate, and here it's smaller. In both cases, the state aggregation algorithm produces a discretization error. This loss of precision is a problem that we are going to mitigate using tile coding.

![](../Assets/photos/Continuous%20state%20spaces_12.PNG)

&nbsp;&nbsp;&nbsp;This technique is just a generalization of state aggregation in which we create a given number of independent aggregations.

![](../Assets/photos/Continuous%20state%20spaces_13.PNG)

&nbsp;&nbsp;&nbsp;And this is what it looks like. We perform several state aggregations, each one with a different color. These aggregations are independent of each other and aggregate different ranges of values. For each aggregation. We'll keep a separate value table with independent estimates of each of its aggregated values. Our estimate of the value of a state will be the average of the value stored in those different value tables. For example, our estimate of the value of this state will be the average of this estimate, this one, this one, and the red one, which should land somewhere around here. If you notice, the average of the estimates will be much closer to the real value function and smoother for most states. We can expect its estimate to be closer to the optimum.

![](../Assets/photos/Continuous%20state%20spaces_14.PNG)

&nbsp;&nbsp;&nbsp;But how can we apply this technique to problems with more complex state spaces? Well, let's look at it through the mountain car problem. As you know, this is a task in which the state has two values, two dimensions. The first one is the position of the cart and the second its velocity. This here is the two dimensional state space.

![](../Assets/photos/Continuous%20state%20spaces_15.PNG)

&nbsp;&nbsp;&nbsp;As you can see, we have created several grids that aggregate different portions of the state space. And all the states that follow within the same tile in a specific aggregation are represented by a single state. Since we have several independent aggregations. A state can belong to different tiles in different aggregations. For example, this state here belongs to the tile on the second row, second column in the red aggregation. It also belongs to the tile on the second row and second column on the green one. But it belongs to the tile on the third row. Second column on the golden one.

![](../Assets/photos/Continuous%20state%20spaces_16.PNG)

&nbsp;&nbsp;&nbsp;To carry out the tile coding technique will follow these steps. First, we are going to create a grid by dividing each dimension of the state into segments. As we see here.

![](../Assets/photos/Continuous%20state%20spaces_17.PNG)

&nbsp;&nbsp;&nbsp;Then we're going to change the size of the grid, enlarging or shrinking it by a random factor. This will help us to make different aggregations independent from one another.

![](../Assets/photos/Continuous%20state%20spaces_18.PNG)

&nbsp;&nbsp;&nbsp;Next, we'll use a displacement vector to move the grid by a small amount in each dimension of the state. For example, the state aggregation has been moved to the left by a certain amount described in the first element of the vector. And on the Y direction by the second element of this displacement vector. And on the Y direction by the second element of this displacement vector. For example, in this case, since we have two dimensions, this vector will contain one and three because twice the number of dimensions is four, and this vector only contains the odd numbers.

![](../Assets/photos/Continuous%20state%20spaces_19.PNG)

&nbsp;&nbsp;&nbsp;Finally, we'll repeat this process as many times as the number of grids that we need. Each new grid that we create will take the last one as reference. For example, we'll create the green one starting in the position of the red one, enlarging or shrinking it and then moving it based on the displacement vector and then starting at the position of the green one. We'll create the yellow one, change its size and move it around.

![](../Assets/photos/Continuous%20state%20spaces_20.PNG)


https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_7_continuous_observation_spaces_complete.ipynb#scrollTo=HZls8wqNvgqO

























