# Social Interaction as Regularization for Hierarchical RL

Hierarchical Reinforcement Learning (HRL) promises temporal abstraction through goal-conditioned policies: a high-level “manager” sets subgoals for a low-level “worker” (Vezhnevets et al., ICML 2017). However, HRL notoriously suffers from **goal collapse**: subgoals degenerate to trivial solutions—either one step away from the current state (making hierarchy useless) or identical to the final goal (providing no decomposition). Regularization techniques help but don’t address the fundamental question: *what makes a goal representation “good”?*

Theories of cognitive evolution suggest that abstraction, compositionality, and language co-emerged in social species as tools for coordination (Tomasello, 2009). Communicating a goal to another agent requires that goals be sufficiently abstract to generalize across contexts, yet specific enough to be actionable. A limited vocabulary forces compositionality: describing “the red key near the blue door” requires combining concepts rather than memorizing configurations.

This project hypothesizes that **multi-agent coordination pressure can regularize HRL goal representations**. If agents must communicate subgoals to coordinate on shared tasks, the communication channel imposes structure on goal space. Goals that collapse to trivial solutions would be communicatively useless—saying “go one step left” provides no information a partner couldn’t infer. Meaningful coordination requires meaningful abstraction.

The HIRO framework (Nachum et al., NeurIPS 2018) uses full states as goals, avoiding information loss but struggling to scale. Feudal Networks (Vezhnevets et al., ICML 2017) learn latent goal spaces but are prone to collapse. Neither leverages multi-agent structure. By introducing communication requirements, we may obtain goal representations that are both informative and well-structured.

---

## Research Questions

1. **Can multi-agent coordination pressure prevent goal collapse in HRL?**  
   Compare goal distributions (entropy, coverage) between single-agent HRL and multi-agent variants with communication.

2. **Does emergent communication structure transfer to better goal representations for single-agent tasks?**  
   Train with social pressure, then evaluate goal-conditioned policies in isolation.

3. **What is the relationship between communication vocabulary size, compositionality of messages, and quality of goal representations?**  
   Do bottleneck constraints that promote compositional communication also prevent goal collapse?

4. **Which multi-agent scenarios provide the strongest regularization?**  
   Compare shared resources, explicit communication channels, and turn-taking coordination.

---

## Suggested Approach

Design multi-agent environments where coordination benefits all agents.

A concrete setup:
- Use **Minigrid** with multiple agents
- Introduce a shared “bus” resource that is cheaper per-agent when used simultaneously
- Require agents to coordinate arrival times

Agents must communicate intended subgoals to enable coordination.

Train HRL agents (Option-Critic, Feudal Networks, or HIRO-style) with and without communication channels.

Analyze goal representations:
- Do communicated goals exhibit less collapse?
- Are they more diverse and temporally extended?

Test transfer:
- Train with social pressure
- Evaluate the manager’s goal-setting in single-agent tasks

Relevant methods:
- **MADDPG** (Lowe et al., NeurIPS 2017) for centralized training with decentralized execution
- **LOLA** (Foerster et al., AAMAS 2018) for opponent-aware learning

---

## Key References

- Vezhnevets, A. et al. (2017). *Feudal Networks for Hierarchical Reinforcement Learning*. ICML.  
- Tomasello, M. (2009). *The Cultural Origins of Human Cognition*. Harvard University Press.  
- Nachum, O. et al. (2018). *Data-Efficient Hierarchical Reinforcement Learning*. NeurIPS.  
- Bacon, P.-L. et al. (2017). *The Option-Critic Architecture*. AAAI.  
- Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. NeurIPS.  
- Foerster, J. et al. (2018). *Learning with Opponent-Learning Awareness*. AAMAS.  