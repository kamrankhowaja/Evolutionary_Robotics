Part b)

Its a good natured optimization problem 

- Monotonic fitness landscape: The fitness function is the sum of independent contributions from each character position. Each correct character adds exactly 1 to fitness.
- No local optima: Unlike typical optimization problems, there are no traps. Every step toward the goal increases fitness, and you can never get stuck in a local maximum that isn't the global maximum.
- Guaranteed progress: With our accept condition (fitness >= previous), you'll never backtrack, ensuring monotonic improvement.


Part c)

Thoretically the estimation of expected generation can be given using probability 

Imagine you hvae k no of correct character and n-k number of wrong character so, 
- probability of selecting a wrong position is (n-k)/n , (Beacuse everytime the probability changes and fewer position remains wrong rest are updated)
- probability of choosing correct character 1/27 (Considering Space too)
- probability of improvement per generation  (n-k)(1) / n(27)

So the expected generation to go from k to k+1(next letter as correct) = (n(27))/(n-k)(1) , considering geometric distribution, as we have sequence of independent beranuli trials each with a constant of probability of success 

