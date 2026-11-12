# Recognizing a network is not understanding it

Code and setup for the experiments in the blog post
[*Recognizing a network is not understanding it*](https://www.yangkev.in/blog/recognizing-a-network-is-not-understanding-it).

We give small neural networks to GPT-6 Astra and Claude Opus 5.5 as raw printed weights and ask what each network
does. The experiments separate two skills: **recognizing** a network from the shape of its weights, and
**computing** what it actually does from input to output. Both models recognize remarkably well. Computing holds up
only when the question pushes them to do it, or when they have a calculator.

## The experiments

Each folder's README explains the question, the setup, how answers were scored, and the post's results.

1. [**Recognizing the game**](experiments/1-recognizing-the-game): name the game a policy plays, even with the
   board picture in its first layer camouflaged.
2. [**Rewriting the network**](experiments/2-rewriting-the-network): exact rewrites that change how the network is
   written but not what it does.
3. [**Twin games**](experiments/3-twin-games): look-alike games with opposite rewards, where recognition gives the
   wrong answer.
4. [**How deep a model can compute**](experiments/4-how-deep-can-a-model-compute): follow a rule through 2 to 20
   hidden layers, with and without Python.
5. [**A rule in a scrambled transformer**](experiments/5-transformer-rule-and-mirrors): recover an unseen rule, and
   tell a network apart from its mirror image.

## What's here

- `experiments/`: one reading guide per experiment.
- `studies/`: the runnable code and setup (games, training, networks, prompts, answer keys, pre-registrations). It
  keeps its original layout because scripts find files by relative path.
- `post-figures/`: the post's figure data and the scripts that build it.

Model replies are not included; run the readers on the stored prompts to produce new ones. See [DATA.md](DATA.md)
for everything else that is left out, and each study's README for setup.

The pre-registrations were edited for release to remove operational details, so some script hashes they record no
longer match. The answer-key hashes are unchanged.

## Credit and license

The game-recognition setup builds on *Sparks of Weight-Space Programming* (Takehi et al.,
[`RikiyaT/lm_understand-logs`](https://github.com/RikiyaT/lm_understand-logs), CC BY 4.0); their networks are
fetched, not redistributed.

Code is [MIT](LICENSE). Everything else is [CC BY 4.0](LICENSE-DATA).
