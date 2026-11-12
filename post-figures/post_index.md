---
title: Limitations of Weight-Space Programming: Recognizing a network is not understanding it
description: Claude Opus 5.5 and GPT-6 Astra are really good at reading weights. But how does that translate into understanding?
date: 2026-11-09
category: Research
kicker: Limitations of Weight-Space Programming
hero: HeaderWeights
---

Last week Takehi et al. published [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). They gave the weights of a small neural network, about 17,000 numbers, to GPT-6 Astra, and asked which game the network had been trained on. Astra recognized the right game in all four networks they tried.

The capability seems impressive at first glance (does a frontier model have superhuman capability of intuition of what a network is doing, without running it?). But it's worth decomposing "reading" into two different skills. First, recognition: can the model pattern-match the "shape" of the weights into something familiar, likely within its pretraining distribution? And second, computation: can the model follow the arithmetic the weights encode, from input to output, and understand what it does to some arbitrary inputs?

I trained toy networks to separate what the models recognize and compute, and found that both models are indeed strong at recognition, beyond pretraining pattern-matching. For computation and intuition, I use a twin-game study to demonstrate that the models still do fail at _really_ understanding what a network does, unless specifically prompted for it.

## The original weight-reading experiment

The neural networks in the MIT experiment are _policies_: each looks at a game board and picks a move. The board is a 10 × 10 grid with five on-off channels per cell (the player's piece, moving objects, reward targets, obstacles and the previous frame), which makes 500 inputs. Two _hidden layers_ of 32 units sit in between, for 17,253 weights in all.

```figure PlayAndRead
Left: a policy playing the game, live in your browser, with each hidden unit's output and the chance of each move at this step. Right: the start of what the model is given, the same network as printed numbers. Pick a game.
```

## The weights contain a picture of the board

How does a model get from that wall of numbers to a game? If you rearrange the first layer of weights, the answer is trivial: the different channels encode the shape of the board or where the player can move. Recognizing a famous game this way is closer to spotting an encoded photo than understanding a network.

```figure BoardPictures
The weights as printed look like noise, but folded onto the 10 × 10 board, the game's boundaries and how to play are obvious.
```

### Camouflaging the input layer

To test the model's reading ability, I obscured the neural network (without retraining), to preserve the game's function but write a used input's weights into the dark cells. On these 48 camouflaged networks, Opus recognized all 48, and Astra 43.

## Rewriting the network

So I went a step further, with eight rewrites to test the neural network recognition ability for increasingly complex networks. They cover two network shapes: the original shape, with two hidden layers of 32 tanh units, and a deeper one with three hidden layers of 24 ReLU units. Each preserves or improves on the original game-playing ability.

```figure Disguises
Each row is a different neural network rewrite. Select a row to see what changes and how many of the four games each model still recognized.
```

Astra and Opus were strong at recognizing the two-layer tanh policies, even after large changes like adding units that cancel, splitting every unit in two and mixing the inputs. But with a three-layer ReLU policy, when input weights are randomly filled, success goes down from 100% to 25% for Astra and 50% for Opus across the four networks. Adding extra do-nothing layers doesn't change this, so the random fill, and not the added depth, is what causes the degradation. On the tanh policies the same switch costs Astra only the Breakout policy, and I do not know why it costs so much more here.

With a Python sandbox on the four deepest networks, where the models could also run the network, both do slightly better: Opus recognizes 3 of 4 instead of 2, Astra 2 instead of 1. (Two Opus sandbox conversations in the recognition tests described code output that never existed. I discarded both and reran each once.)

## Recognizing a game is not understanding the network

To catch a model recognizing when it should be computing, I trained networks to play a _twin game_, with the same board and pieces as a famous game but rewards for the opposite behavior. In Catch a paddle collects falling objects; in its twin, Dodge, it must avoid them. Hunter chases the creatures that Pac-Man flees. Car-tag touches the cars that Freeway avoids.

Both models walk into the trap. Both call Car-tag Frogger, Freeway's better-known cousin, every time, and Opus calls 2 of its 4 Dodge networks Catch. Astra reads Hunter as a chase about half the time. Astra describes the right behavior for 9 of the 18 twins, Opus for 5 of the 16 it has answered so far. The skill that took Opus to 48 of 48 on the camouflaged networks takes it to 5 of 16 here.

### Ask and you shall recieve

If you warn Astra that the game might be a look-alike, it checks more carefully, going from 9 to 16 of 18 correct. Ask as well whether the player moves toward the moving objects or away from them (giving away the challenge), and it gets all 18. Replace the open question with a forced choice between the two games and Astra again gets all 18, Opus 13 of the 15 it has answered so far. The models can tell the twins apart, it's just too lazy to check unless asked.

```figure QuestionLadder
Twin networks under questions that get more pointed row by row, then with Python. Bars count answers that describe the right behavior. Switch to the famous games for comparison.
```



### Python alone does not

Given the open question and a sandbox with no rules, Astra was free to run the policy and watch. On the camouflaged twins, Python lifts Hunter (2 of 4 to 4 of 4) and nothing else: Car-tag stays at 0 of 4, still called Frogger. The hint does more than the tools. With both, Astra gets all 12 right.

## How much can a model compute in its head?

The twin experiment demonstrates that models are capable, when pushed, of computation to recognize a neural network's function. To measure how far that goes, I trained deeper networks to follow one of two opposite rules: move toward the ball when it travels right and away when it travels left, or the opposite. The first layer looks the same, but the computation is hidden deeper in the network.



```figure DepthCurve
Share of paddle networks whose rule each model gets right, with no tools. Astra stays accurate about twice as deep as Opus. Switch the x-axis to tokens to compare at equal prompt size, or to Python for the calculator test. In the Python view, check quoted numbers against what the code printed: two Opus conversations reported results from analyses they never ran.
```

I measure how deep the network has to be to get the accuracy of the model to 75%, halfway between random chance and perfect. Astra maintains 75% accuracy until 11.1 layers deep, versus 6.1 for Opus. Measured by the size of the prompt instead, it is about 115,000 tokens for Astra and 100,000 for Opus (perhaps a better metric for comparing the two, as Opus has more tokens per number).

At 8 layers, the answer requires tracing 24,000 printed numbers and nine matrix multiplications. With no calculator, Astra is right on 45 of 48 networks and Opus on only 30 of 48.

### Astra starts at the input layer and gets further

Astra and Opus often start from different ends. Astra traces further by tracing from the input layer by hand in 46 of its 60 replies. Deeper than 4 layers, it starts approximating some calculations. By 14 and 16 layers it often decides it cannot trace that far and guesses.

Opus more often starts at the output and works backward: treat the network as one long matrix product by pushing the "right" score minus the "left" score back through every weight matrix. Where Opus carries its reasoning through every layer it is right 21 times in 23. But from 8 layers on it tries this in only 4 of 35 replies. The other 31 replies call the arithmetic too long to do by hand and fall back on the first and last layers, which is a failing strategy.

```figure DepthReplies
One square per reply, colored by how far it says it traced through the network. The bar indicates it got the answer right
```

### With a calculator

Given a Python sandbox for arithmetic on the weights (but no forward pass), both models jump to 53/60 networks correct from 4 to 20 layers deep. In nearly every conversation, the model reads the network's linear approximation by multiplying the weight matrices. It also adds up the signal along every route from input to output: the same product Opus attempts in its head but rarely finishes past 8 layers, and far more routes than Astra can trace by hand. They get things wrong when the activations, which the product leaves out, change the answer.

## An unfamiliar rule in a scrambled transformer

Next, I ask the model to attempt to follow a more complicated rule on a larger neural network, a transformer with 63,360 parameters (~260,000-290,000 tokens). The rule is one the models have never seen, so recognition can't help, and the network is scrambled so that no single weight gives it away.

It reads a row of 16 tokens, and the last repeats an earlier one. It outputs a token a few places from that earlier copy: _a_ places if a yes-or-no test passes, _b_ places if not. There are 120 possible rules. Before printing, I rotated the network's internal coordinates at random, which spreads every meaning across many weights without changing what it computes.

With no tools, Astra gets the whole rule right for 6 of 16 networks (baseline: guessing gets it right with 1/120 chance) and Opus for none. On 24 other networks, Python for arithmetic on the weights (but no running the network) takes Astra from 6 right to 19. That's much better, but still not all.

Even a right answer could rest partly on a guess, most likely in the rule's _direction_: which offset goes with "yes". So the 24 networks are 12 networks and their _mirrors_. A mirror is the same network with its tokens relabeled so it takes the opposite offset. To any statistic that ignores which token is which, the two look identical. Only a model that follows the mechanism can tell them apart. Astra wasn't told that mirrors existed.

```figure MirrorPairs
Astra identifies the rule in a transformer model. With Python; switch for no tools.
```

With Python, Astra gives the network and its mirror opposite directions in all 11 pairs where its two answers name the same test and offsets. With no tools, only 4 pairs get that far, too few to say anything.

## Limits

These are toy networks: at most 20 layers and 63,360 weights. The same effort setting is not the same amount of work: on the camouflaged recognition test on max effort for both models, Opus used 8x the thinking tokens. When Opus ran out of output at max, I reran it one step lower (xhigh) and count that answer; the figures mark those counts with an asterisk. The mirror test ran on Astra only.

## Takeaways

```figure Scoreboard
Share of networks each model read correctly, test by test. Select a test to jump to its section.
```

So how well can a model read a network? It depends on what you count as reading.

Astra and Opus are both very good at recognizing a network from its weights. They do it even when the "picture" in the weights has been rewritten or camouflaged, perhaps better than any human could. Understanding falls short, for two reasons.

- **The model can compute, but takes the shortcut.** The twin experiment demonstrates models can do the math to trace a neural network. Given a forced choice between the two games, Astra gets 18 of 18 and Opus 13 of 15. Without forcing a choice, however, both models fall back to incorrect pattern-matching.
- **The model has limited capacity for "mental math" for weight tracing.** Without tools, Astra can trace a network to about 11 layers before its accuracy falls to 75%, and Opus only to 6. That is remarkable in itself: at those depths, Astra's median reply runs about 32,000 tokens of reasoning, and Opus's about 360,000. With a calculator, both models stay accurate to 20 layers, which is unsurprising.

Given a calculator and a question about what the network does, Astra's transformer result is the strongest evidence of deeper neural network understanding. Without running the network, it worked out an unfamiliar rule rule in 19 of 24 cases, even distinguishing it from a mirrored rule.

Beyond recognizing what a neural network can do, I'm excited to see more work in weight-level understanding, a capability that programming in weight space depends on. I'm excited to see future work that probes the limits of the complexity of neural networks that models can understand.
