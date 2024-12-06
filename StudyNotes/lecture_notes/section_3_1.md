TITLE: Reinforcement Learning and Markov Decision Processes (MDPs)

1. THEORETICAL FOUNDATIONS

   - **Markov Decision Processes (MDPs):** MDPs are used to model decision-making in environments where outcomes are partly random and partly under the control of a decision maker. An MDP is defined by:
     - A set of states $S$
     - A set of actions $A$
     - Transition model $T(s, a, s') = P(s' \mid s, a)$
     - Reward function $R(s, a, s')$
     - Discount factor $\gamma \in [0, 1)$
   - **Bellman Equation:** The Bellman equation provides a recursive decomposition for computing the utility of states:
     $$ U(s) = R(s) + \gamma \max_{a} \sum_{s'} T(s, a, s') U(s') $$
     - **Bellman Optimality Equation:** For optimal policy $\pi^*$, the equation is:
       $$ U^*(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s'} T(s, a, s') U^*(s') \right] $$
   - **Stationarity Assumption:** The transition probabilities and reward function do not change over time.

2. KEY CONCEPTS AND METHODOLOGY

   A. Essential Concepts:
      - **Policy ($\pi$):** A mapping from states to actions.
      - **Value Function ($U(s)$):** Expected return (cumulative future reward) starting from state $s$.
      - **Q-Function ($Q(s, a)$):** Expected return after taking action $a$ in state $s$ and following the optimal policy thereafter.
      - **Exploration vs. Exploitation:** Balancing between exploring new actions to improve knowledge and exploiting known actions to maximize reward.

   B. Algorithms and Methods:
      - **Value Iteration:**
        - Iteratively update value estimates using the Bellman equation until convergence.
        - Complexity: $O(n^2)$ per iteration for $n$ states.
      - **Policy Iteration:**
        - Evaluate and improve policies iteratively.
        - Consists of policy evaluation and policy improvement steps.
      - **Q-Learning:**
        - Off-policy learning algorithm that seeks to find the best action to take given the current state.
        - Update rule: 
          $$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$
        - **Epsilon-Greedy Exploration:** Choose random actions with probability $\epsilon$ and exploitative actions with probability $1-\epsilon$.
        - Convergence: Requires visiting each state-action pair infinitely often with diminishing learning rates.

3. APPLICATIONS AND CASE STUDIES

   - **Grid World:** Simple environment to illustrate MDPs where an agent moves on a grid with states, actions, rewards, and transitions.
   - **Backgammon (TD-Gammon):** Uses reinforcement learning to train a neural network to play backgammon, demonstrating the practical application of RL in complex games.

4. KEY TAKEAWAYS AND EXAM FOCUS

   - Understand the components and definitions within an MDP.
   - Be able to derive and solve the Bellman equation for given MDP scenarios.
   - Comprehend the trade-offs in exploration vs. exploitation strategies.
   - Recognize the differences between model-free and model-based reinforcement learning approaches.
   - Familiarize with the convergence conditions and update rules for Q-learning.
   - Focus on the implications and solutions of reinforcement learning problems without explicit models of the environment.