Congrats on making it to the other side of the first assignment! It's not a simple task to immerse yourself in the discipline of wrangling data and algorithm and broadcast mismatches, so I'd like to take the time to really highlight the hard work you've put in thus far.

TODO(you): Pat yourself on the back.

Now, we move on to the second assignment; however, before we do that, I'd suggest perusing the A1 FAQ once more because there's a lot of useful information there that generalizes across all assignments. Done? Alright, now let's begin with A2.

Alright, so I have SKLearn / Weka / 5 R libs primed and ready from the first assignment. How do I use all that stuff for the next assignment?

You don't. The simplest way to approach the second assignment is to leverage the mlrose (Python) or ABAGAIL (Java) libraries. These implement all the algorithms you need as well as the optimization problems we'll talk about later on in this FAQ.

For mlrose, the hiive fork is preferable since it has some fixes and is more actively maintained. With that being said, if you run into any installation issues, please provide details about what you're doing on Ed as it's likely to help a lot of people out.

Note: There's another pinned, supplemental post on Ed that will be of use for the neural network portion of the assignment. Check that out as well as you get started on your implementation.

What are the randomized optimization (RO) algorithms I need to implement?

Randomized Hill Climbing (RHC)
Simulated Annealing (SA)
Genetic Algorithms (GA)
Mutual-Information-Maximizing Input Clustering (MIMIC)
Are there extra credit opportunities as with the previous assignment?

Yes, you may implement MIMIC as a point of comparison against the second optimization problem which is highlighting GA. Follow the FAQ outline as you normally would for MIMIC in that section.

Help, all this stuff about RO sounds scary! Where do I start?

Let's start by distilling the assignment to its core requirements. There's two sections which you may present in either order in your report. As with A1, there's little preference (although I provide some pointers in the A1 FAQ) on how you arrange things, as long as you clearly label what you're doing.

In one section:

You will create or steal three two optimization problems and run RHC, SA, GA, and MIMIC on your chosen optimization problems.
... with an additional constraint: Highlight each of GA, SA, and MIMIC in one of the respective optimization problems.
Hint: These optimization problems are available in the corresponding library you chose above. You'll need to experiment with various problems and choose the appropriate ones that highlight one of the respective algorithms.
In the other section:

You will compare backprop to RHC, SA, and GA while (hopefully) reusing one data set from A1.
To facilitate this, you'll need to freeze your network architecture from A1 (hidden layer size, activation functions, input / output layers, etc.) and use each of the RO algorithms as a backprop replacement for weight updates.
What kind of plots should I focus on for the optimization problem section?

You'll find that a major theme of the assignment is this notion of convergence. In fact, many of the algorithms could do better indefinitely so it's important to explicitly pick a convergence criteria or understand the underlying convergence criteria of the library you're using.

In terms of plots:

Fitness / Iteration: Since we're dealing with optimization problems, those problems will present access to a fitness score at each iteration.
Fitness / Problem Size: Focusing on a single problem size can misrepresent how well an algorithm is actually doing. To counteract this, try multiple problem sizes.
Function Evaluations: There's many ways to interpret fevals. Some students will focus on fevals / wall clock times, or fevals / iterations, etc. The real requirement is that you somehow look at fevals. Keep in mind that some algorithms may take significantly less iterations but each iteration may take significantly longer.
Wall Clock Time: Our favorite. Look into it.
And, again, hopefully it's rather clear that you should cover this notion of convergence extensively in your analysis.

Do I need to tune hyperparameters for every problem size?

That would be ideal.

It seems like there's quite a bit of variance between runs... how can I account for this?

Well, "randomized" is in the name. What I'd suggest is:

Generate 
N
N seeds that you'll reuse throughout your experiments
Take the average and provide a visualization of variance within your plot
This will counteract the possibility of picking a single run that was unlucky or lucky.

What kind of plots should I focus on for the NN section?

We're really looking for you to somehow compare your A2 results to your A1 NN results. Naturally, you might use some of the existing tools that you're already familiar with from A1. For example, things like learning curves. The iterative (or lossy) version of said visual would also fit into the whole notion of convergence that the assignment is trying convey anyway.

With that being said, if you'd like to focus on properties of bias and variance and the interactions of the RO algos with that, then feel free to do so.

Oh, and by the way, you should probably try hard to acclimate to thinking of trade-offs here. Hypothetically, a RO algorithm may actually do "better" than backprop but we should fairly and explicitly define what "better" means in terms of fitness, iterations, time, etc. Right?

Anything else I should know?

Follow the general structure that we've already set out for you from A1.

Introduce your optimization problems, NN data set, etc. and explain why they're interesting in terms of applying the RO algorithms. For example, MIMIC may exploit inherent structure within the data using dependency trees. Other algorithms are rather rudimentary but that simplicity can provide positive trade-offs for certain classes of problems.

Hypothesize, hypothesize, hypothesize. Try to set the ground work for experimentation by doing an initial walk of lecture theory and readings so you don't wander aimlessly into the forest. :-)

As with A1, it helps to be inquisitive about your results... focusing on why certain algorithms struggle with certain types of problems and why others do well on said problems. 

Finally, you should talk at length on what worked and what didn't. How did you improve performance by tuning hyperparameters? Why did one configuration work while the other did not? Is there anything you can do to improve performance even further?

You can even be theoretical about it, imagining ways you could improve things if you had more time.

What is all this stuff about bit strings?

Mitchell has a decent section with the theory behind this so that would be the best place to start.

They're preferable, but not required. I'd suggest not getting lost in the details of this; just work through the optimization problems available to you and you can always peruse the code to understand how the problem is being represented.

With that being said, you're able to create your own problems if you'd like. I'd suggest doing this after you experiment with a couple of existing problems yourself to get your bearings on problem representation.

What if I have a minimization problem?

Any minimization problem can be restated as a maximization problem and vice versa.

What does it mean to highlight an algorithm?

It'll be up to you to justify. Fitness, problem sizes, wall clock times, etc. may all factor into painting a complete picture. That is to say that most students will likely use multiple visuals or results to find trade-offs that favor a specific algorithm.

What if I don't want to reuse my data from A1?

That's fine. Just more work for you; it's OK to switch for data sets for A2 or A3.

It seems I've used some special sauce hyperparameters from SKLearn that aren't available in mlrose or ABAGAIL...

It might be difficult to do an apples to apples comparison depending on how many specialized library hyperparameters you used for your NN in A1. If that's the case, it would be appropriate to do a rough reimplementation of your A1 NN in the library you chose.

As in, run backprop *within* the A2 library you pick.

Do I need to provide a plot for everything?

As with A1, you may use your discretion in choosing which plots you'd like to present visually, in tabular format, or textually. Just make sure you include them in some way.