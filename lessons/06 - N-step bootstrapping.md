# N-step temporal difference methods

&nbsp;&nbsp;&nbsp;ٌWe're going to learn about the family of algorithms that lie between `Monte carlo` and `temporal different` methods. They are called `n-step temporal difference` methods.  
These algorithms learn based on experience and use a technique known as n-step bootstrapping.

![](../Assets/photos/N-step%20temporal%20difference%20methods_1.PNG)

&nbsp;&nbsp;&nbsp;To explain what n-step bootstrapping is, let's quickly review the update rule of the SARSA algorithm, which we saw in the previous section. Every time that we update a q-value estimate, we push the current estimate in the direction of the target. In an amount proportional to alpha. This target is the reward obtained after taking the action at time 't' plus the estimated q-value of the next state and the action chosen in that next state. Recall that a q-value is the expectation of future rewards from taking an action.

![](../Assets/photos/N-step%20temporal%20difference%20methods_2.PNG)

&nbsp;&nbsp;&nbsp;So here Q replaces the future rewards with an estimate, and using an estimate to update another estimate is what we know as bootstrapping.

![](../Assets/photos/N-step%20temporal%20difference%20methods_3.PNG)

&nbsp;&nbsp;&nbsp;The advantage of using an estimate to update another estimate is that we don't have to wait until the end of the episode to obtain the remaining rewards because we use an estimate to replace them. In this case, we're performing one step bootstrapping because we're using one actual reward and we estimate the rest. So we are applying our estimate one step in the future.

![](../Assets/photos/N-step%20temporal%20difference%20methods_4.PNG)

&nbsp;&nbsp;&nbsp;But we could have also taken another action, obtained another actual reward, interacting with the environment and estimated the rest by replacing them with the q-value estimate of the state and action chosen two steps into the future, or even collect three rewards and estimate the remaining ones or even 'n' rewards. All of these expressions are valid estimates of the return of the episode. The difference is how many actual rewards they include and how many we estimate using the q-values.

![](../Assets/photos/N-step%20temporal%20difference%20methods_5.PNG)

&nbsp;&nbsp;&nbsp;When we include 'n' rewards obtained from the environment to an estimate, we call it the n-step return estimate. And we'll write it like this: G from 't' to 't+n'. Well, n-step bootstrapping consists of replacing the rest of the rewards after the first 'n' reward with an estimate.

![](../Assets/photos/N-step%20temporal%20difference%20methods_6.PNG)

&nbsp;&nbsp;&nbsp;And if we use the n-step return estimate as the target of our updates, this expression is still correct and is the update rule of this new family of methods called n-step temporal difference methods. Using these methods, we'll have to wait 'n' steps into the future to update the q-value estimate of the present state, because we'll have to collect those'n' rewards to be able to compute the estimate of the return.

![](../Assets/photos/N-step%20temporal%20difference%20methods_7.PNG)



# Where do n-step methods fit

&nbsp;&nbsp;&nbsp;We're going to see how `Monte carlo` methods and `temporal different` methods are connected to this new family of `n-step` methods.  
Let's go back to `SARSA` for a moment. This here is the update rule. The target is the first reward, plus a discounted estimate of the q-value of the next action taken in the next state. But this is actually the episode return estimated in one step. That's the value to which we push our estimate of the q-values. So in fact, `SARSA` is just a special case of the `n-step` family of algorithms in general `temporal differences` methods are one special case of `n-step temporal difference` methods where 'n' equals one.

![](../Assets/photos/N-step%20temporal%20difference%20methods_8.PNG)

&nbsp;&nbsp;&nbsp;But the `SARSA` algorithm could also use any of these targets. The return estimated in two steps in three or in 'n'. If we do that, the resulting algorithm will be called n-step `SARSA`. But there is a catch. If our 'n', that is the number of real rewards that we want to incorporate in our estimate is bigger than the actual duration of the episode, then we'll have the discounted sum of every reward obtained during the episode, which is the actual return, not an estimate.

![](../Assets/photos/N-step%20temporal%20difference%20methods_9.PNG)

&nbsp;&nbsp;&nbsp;Well, it's the same that happens with `Monte carlo` methods. `Monte carlo` methods are the other extreme of this family where the 'n' is so big that it's larger than the number of steps in the episode. What this means is that we will include every single reward in our computation of the return and in fact, this update rule here is simply the update rule of the constant alpha `Monte carlo` method.

![](../Assets/photos/N-step%20temporal%20difference%20methods_10.PNG)

&nbsp;&nbsp;&nbsp;Now we can see where `n-step` methods fit. They are a family of methods that extend and encompass `Monte carlo` and temporal different methods. `Monte carlo` methods are a special case where the 'n' is bigger than the duration of the episode and temporal difference methods are the other extreme where 'n' equals one. By adjusting the value of 'n', we can push our method towards one family or the other.

![](../Assets/photos/N-step%20temporal%20difference%20methods_11.PNG)



# Effect of changing n

&nbsp;&nbsp;&nbsp;We're going to see how the choice of 'n' affects the learning process. Remember that 'n' is the value that decides how many rewards obtained interacting with the environment we're going to include in our estimate of the return. And that value is arbitrarily chosen by us.

![](../Assets/photos/N-step%20temporal%20difference%20methods_12.PNG)

&nbsp;&nbsp;&nbsp;To see its effect, imagine that we are going to play a game of darts. When making an estimate of a value, there are `two` problems that can negatively affect these estimates, and trying to mitigate one tends to make the other worse.  
The first problem that can arise when we try to estimate a quantity is `bias`. This problem happens when our estimates are systematically away from the actual quantity. As you can see, the player throws all the darts accurately because they land next to each other, but the player is aiming in the wrong direction.

![](../Assets/photos/N-step%20temporal%20difference%20methods_13.PNG)

&nbsp;&nbsp;&nbsp;The second problem is the `variance` of the estimate. This problem happens when the estimates are very different from one another, although on average they are aiming at the right quantity. In this place, although the player is aiming at the right place, their throws are very inaccurate and these problems are not mutually exclusive.

![](../Assets/photos/N-step%20temporal%20difference%20methods_14.PNG)

&nbsp;&nbsp;&nbsp;The player can be aiming at the wrong position and throwing the darts very accurately.

![](../Assets/photos/N-step%20temporal%20difference%20methods_15.PNG)

&nbsp;&nbsp;&nbsp;Well, remember that when estimating the return of the episode. Q is an estimate of future rewards. And this estimate improves during the learning process. But it doesn't have to be right from the beginning. That is at the beginning of the learning process Q Can you give us a biased estimate.

![](../Assets/photos/N-step%20temporal%20difference%20methods_16.PNG)

&nbsp;&nbsp;&nbsp;The larger 'n' is the more heavily discounted this estimate will be. What this means is that `the higher 'n', the lower potential bias our estimates will include`.

![](../Assets/photos/N-step%20temporal%20difference%20methods_17.PNG)

&nbsp;&nbsp;&nbsp;On the other hand, the problem with variance arises for other reasons. Here's the formula of the return. Not an estimate, but the actual return. The return includes every single reward observed during the episode discounted by the proper gamma value. And each one of these rewards is a random variable that depends on the state where the action was taken and the action that led to that reward. So the return is a sum of random variables. This makes the combination of rewards that form the return highly variable, because if the policy chooses a different action at the beginning of the episode, the rewards that will obtain throughout the rest of the episode can vary a lot because we'll visit different states and probably choose other actions. Although the expected return will be the same, every observation of the return will be very different from the previous one. For that reason, `the larger 'n' the greater the variance`.

![](../Assets/photos/N-step%20temporal%20difference%20methods_18.PNG)

&nbsp;&nbsp;&nbsp;Now we understand what we choose when we pick a body for 'n': we are exchanging bias for variance. The smaller the 'n', the smaller the variance. Because less rewards are included in the estimate of the return, but the greater the bias, because the estimate will be discounted less heavily. And the weight of the estimate in the overall estimated return will be higher. On the other hand, if we pick a higher value for 'n', the q-value estimate will be more heavily discounted. Which means that the bias will be lower, but our estimate of the n-step return will incorporate more rewards, which means more random variables and more variance to the overall estimation. In practice, intermediate values for 'n' achieve better results than the extremes.

![](../Assets/photos/N-step%20temporal%20difference%20methods_19.PNG)


# N-step SARSA

&nbsp;&nbsp;&nbsp;We're going to see an extension of the `SARSA` algorithm to n-step methods called `n-step SARSA`. This is simply a version of `SARSA` that uses `n-step bootstrapping`. That is, we will use as target for our updates, the estimate of the return in n-steps, where we'll have 'n' real rewards and an estimate of the following ones.

![](../Assets/photos/N-step%20temporal%20difference%20methods_20.PNG)

&nbsp;&nbsp;&nbsp;This is the update rule that we'll use to improve our estimates of the q-values. It looks identical to `SARSA`, except that now we are going to use as target the n-step estimate of the return.

![](../Assets/photos/N-step%20temporal%20difference%20methods_21.PNG)

&nbsp;&nbsp;&nbsp;As in the version of `SARSA` we've already seen, this algorithm will follow an on-policy learning strategy and will use an epsilon greedy policy that will sometimes pick a random action. Every time that we have to choose an action we will flip a coin. And with probability epsilon we will pick a random action and with probability one minus epsilon we will choose the action with the highest estimated value.

![](../Assets/photos/N-step%20temporal%20difference%20methods_22.PNG)

&nbsp;&nbsp;&nbsp;Here's the complete algorithm. It's quite similar to `SARSA`, but we are forced to make some changes to accommodate the use of n-step returns. The first thing we'll do, as always, is to initialize the policy and the table of values. Then we'll enter the main loop, we will start the episode, pick an action for that initial state, and then we'll enter the inner loop, which we will run t+n times until we have updated all the states. In each iteration if the task hasn't finished, we'll execute the action and observe the reward and the next state attained, and then for the new state we will pick another action. Then if we have enough observations to calculate the n-step return, we will compute it and use it to update the q-values. This B here is the bootstrap value. If after 'n' steps the episode is not done, the bootstrap value will be the q-value of that state and the action selected at that state. Otherwise it will be 0 because if the episode has ended, then we don't expect to obtain any additional rewards. When the process ends, we'll have a near optimal policy and near optimal q-values. As you know, near optimal, because our policy is responsible for exploring the environment as well and sometimes it will pick a random action.

![](../Assets/photos/N-step%20temporal%20difference%20methods_23.PNG)


# N-step SARSA in action

&nbsp;&nbsp;&nbsp;We're going to see the learning process of the n-step `SARSA` algorithm in practice. In our example, the value 'n' will be eight. That means to update  Q_value, you will collect eight rewards interacting with the environment and will substitute the rest by a Q_value estimate. In practice, what this means is that we'll have to wait until we make eight moves to start updating the Q_values of the first actions taken. This is the state of the environment. After taking the first action, as you can see, the agent has moved down. So the first action that this algorithm will update is this one here. But still, we don't have enough rewards to compute the 'n' step estimate of the return. So the agent will keep interacting with the environment and after two steps, it will return to its initial position. As you can see, the Q_value table will still not be affected. Then after eight moves, we can finally compute the 'n' step estimate of the return, in this case, the eight step estimate and update the Q_value of the first action that we performed after performing 100 actions will have performed 92 updates and the Q_value table will look like this. So on the one hand, using and 'n' step `SARSA`, we have to wait and steps until we are able to start the learning process. But on the other hand, those estimates will include more information from the environment instead of a single reward. They will include 'n'. And that will be a more reliable estimate of the return.

![](../Assets/photos/N-step%20temporal%20difference%20methods_24.PNG)
![](../Assets/photos/N-step%20temporal%20difference%20methods_25.PNG)
![](../Assets/photos/N-step%20temporal%20difference%20methods_26.PNG)
![](../Assets/photos/N-step%20temporal%20difference%20methods_27.PNG)


https://colab.research.google.com/github/escape-velocity-labs/beginner_master_rl/blob/main/Section_6_n_step_sarsa_complete.ipynb#scrollTo=TTJGndBmtJiD





































































