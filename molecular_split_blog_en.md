# Rethinking Molecular Dataset Splitting: From Benchmark Splits to a Tunable Split Framework

## Introduction

In machine learning, data splitting is often treated as a routine technical step: divide the dataset into training, validation, and test sets, then evaluate the model on the test set. In molecular machine learning, however, the story is much less straightforward.

The reason is simple. Molecular models are not ultimately meant to perform well on molecules already seen during training, but on new molecules drawn from shifted distributions. Whether the task is property prediction, activity screening, or lead optimization, the real value of a model lies not in memorizing the training set, but in making reliable judgments on molecules that fall outside the training distribution. For this reason, **generalization** matters more than mere fit in molecular tasks, and data splitting becomes more than a preprocessing detail: it becomes a mechanism that largely determines what “generalization” even means.

This is why random split is rarely considered sufficient. To evaluate generalization more rigorously, researchers often try to enforce a distributional gap between training and test data. Bemis–Murcko scaffold split, for example, aims to simulate structural extrapolation through scaffold-level partitioning. Fingerprint-based clustering splits, by contrast, try to create more challenging test distributions through distances in representation space. The motivation behind both approaches is reasonable: if the train and test distributions overlap too heavily, test performance may overestimate the model’s real-world utility.

The problem, however, is that benchmark splits are not neutral. Every splitting strategy defines train/test difference in its own way, and in doing so it implicitly defines what kind of variation should count as “generalization.” In other words, a split is not merely an evaluation tool; it actively shapes the conclusion of the evaluation.

That is exactly what this post is about. I will first examine the limitations of two mainstream families of molecular dataset splitting methods: scaffold-based splits represented by BM scaffold split, and representation-based splits represented by fingerprint + clustering. I will then discuss what I believe a good split should look like. Finally, I will introduce a new split framework that is still under development but already has a clear direction: instead of treating split as a discrete benchmark choice, it treats split as a continuously controllable distribution-shift generator.

The more I think about it, the more I feel that in molecular machine learning, whether a model “generalizes well” is never absolute. It is always defined relative to a particular split. So instead of asking which split is the most reasonable, perhaps the more important question is: **what kind of generalization is a split actually defining?**

## Does BM scaffold split really measure structural generalization?

In molecular machine learning, BM scaffold split is arguably one of the most classic benchmarks. Its intuition is very natural: if random split makes training and test sets too similar, then one can first extract the Bemis–Murcko scaffold of each molecule and split the dataset by scaffold. In this way, the scaffolds appearing in the test set are absent from training, and the model is forced to face a form of “structural extrapolation.”

The intuition itself is not the problem. The problem is that BM scaffold is actually a very coarse definition of molecular backbone.

The key idea of the Bemis–Murcko scaffold is to preserve ring systems and the linkers connecting them, while removing most peripheral substituents. This definition has obvious advantages: it is simple, intuitive, and easy to explain. For polycyclic drug-like molecules, it does capture part of the chemical intuition of a “core scaffold.”

But once the focus shifts from individual molecules to an entire dataset, the limitations of this definition become much harder to ignore.

I analyzed scaffold distributions on seven common DeepChem benchmark datasets: BBBP, FreeSolv, HIV, Lipophilicity, QM9, ESOL (labeled as solv in my table), and Tox21. The results were quite revealing.

First, many datasets exhibit a strongly **head-heavy + long-tail** scaffold distribution: a small number of scaffolds account for a large fraction of molecules, while a very large number of scaffolds correspond to only a handful of samples. For example:

- In **FreeSolv**, the top-1 scaffold alone covers **49.8%** of molecules, and the top 5 scaffolds together cover **81.5%**
- In **ESOL**, the top-1 scaffold accounts for **28.1%**, while the top 5 account for **58.0%**
- In **Tox21**, the top-1 scaffold accounts for **22.7%**, while the top 5 account for **44.6%**
- In **QM9**, even the top-1 scaffold already reaches **10.7%**

In terms of imbalance, the scaffold distributions of FreeSolv, QM9, ESOL, and Tox21 are especially extreme, with Gini coefficients of **0.84, 0.80, 0.72, and 0.67**, respectively. This means that “splitting by scaffold” is not really a smooth partition over a balanced structural space. It is much closer to discretely bucketing molecules under a highly imbalanced tag distribution.

Second, several datasets contain a high fraction of **\([NO\_SCAFFOLD]\)** molecules, i.e., acyclic molecules for which no typical BM scaffold can be extracted. This issue is especially severe in some cases:

- **FreeSolv**: **49.8%**
- **ESOL**: **28.1%**
- **Tox21**: **22.7%**
- **QM9**: **10.7%**

This tells us that BM scaffold is not universally expressive. It depends heavily on ring structures, which makes it more natural for polycyclic compounds. But once a dataset contains many oligocyclic or even acyclic molecules, the representation deteriorates rapidly. In the extreme, many structurally different acyclic molecules are effectively thrown into the same “no scaffold” bucket, or handled in some ad hoc way during splitting. Either way, it is difficult to argue that this is a faithful description of “structural generalization.”

Third, there is substantial heterogeneity even within a single scaffold class. Two molecules with the same BM scaffold can still differ greatly in side-chain length, functional-group composition, local bonding patterns, or overall electronic environment. Conversely, two molecules that differ only by a single carbon atom, a linker atom, or a small local modification can be forced into completely different scaffold classes.

This points to the most fundamental issue with BM scaffold: **it is essentially a discrete tagging system rather than a continuous similarity measure.**

The main limitation of a discrete tag is that it can only answer whether two molecules are in the “same class” or “different classes”; it cannot express *how similar* they are. If two molecules share exactly the same scaffold, they are treated as identical at the scaffold level. If the scaffold differs even slightly, they are treated as entirely different. The result is rather unnatural: a very small structural change can be mapped to a very large jump in split space.

So the problem with BM scaffold split is not merely that its definition is “not fine-grained enough.” More fundamentally, it collapses a naturally continuous structural similarity problem into a discrete classification problem. This gives scaffold split strong interpretability, but it does not necessarily make it a faithful proxy for structural generalization. It is better understood as a convenient and highly usable heuristic, one that nevertheless carries substantial structural artifacts.

## Fingerprint + clustering split: more continuous, but also more biased

If BM scaffold split is limited by being too coarse and too discrete, then another common family of methods—fingerprint-based clustering splits—appears more modern at first glance.

These methods typically begin by computing a molecular fingerprint such as Morgan, MACCS, or AtomPair, then defining pairwise distances in that fingerprint space, clustering the molecules accordingly, and finally assigning clusters into train/validation/test sets. Butina clustering is a representative example. Compared with BM scaffold, the apparent advantage here is obvious: there is no need to rely on pre-defined scaffold tags. Similarity is handled directly in a representation space, which makes the split naturally more continuous and more flexible.

But that is precisely where the deeper problem lies: **it depends too heavily on representation.**

A molecular fingerprint is itself a representation. Different fingerprints emphasize different aspects of molecular structure: Morgan focuses more on local environments and neighborhood expansion, MACCS emphasizes predefined structural keys, and AtomPair focuses more on atom pairs and topological distances. In other words, once a fingerprint is used to define pairwise distances, a specific notion of what counts as “similar” and “different” has already been imposed.

The model, meanwhile, also depends on a representation space. A Random Forest trained on Morgan fingerprints and a Random Forest trained on MACCS fingerprints may share the same model class, but they do not “see” molecular differences in the same way. This gives rise to a crucial question:

**What happens if the representation used by the split is very similar to the representation used by the model?**

To probe this, I ran a direct comparison on five datasets. I fixed the model class to Random Forest, trained separate models on Morgan, MACCS, and AtomPair fingerprints, and constructed Butina splits using each of the same three fingerprints. This allowed each experiment to be categorized into two cases:

- **aligned**: the split fingerprint is the same as the model fingerprint
- **unaligned**: the split fingerprint differs from the model fingerprint

The results were quite revealing. On several datasets, aligned splits led to clearly worse performance than unaligned ones. For example:

- On **HIV**, the relative effect of aligned vs. unaligned on the test set reached  
  - PR-AUC: **-16.6%**
  - ROC-AUC: **-26.8%**
- On **ESOL (solv)**, the effect was even more pronounced, with aligned performing worse in  
  - test MAE: **-30.2%**
  - test RMSE: **-20.6%**
  - val MAE: **-16.9%**
  - val RMSE: **-13.7%**
- On **Lipophilicity**, aligned also performed systematically worse on the validation set, with MAE and RMSE dropping by about **7.1%** and **6.0%**

This trend suggests that when the split fingerprint and the model fingerprint are highly aligned, the train/test gap is being deliberately opened along exactly that representation space. The similarity patterns most relied upon by the model are being directly targeted. In that setting, worse performance does not simply mean that the model “generalizes poorly”; it may instead mean that **the split is selectively making the task harder along the model’s most sensitive representational axis.**

To be clear, this phenomenon does not appear with identical strength on every dataset and every metric. For example, the trend is not stably visible on the **BBBP** test set. But I do not think BBBP should be interpreted as a real counterexample. A more plausible explanation is that on BBBP, the overall shift introduced by Butina split relative to random split is itself relatively small. The additional difference between aligned and unaligned is therefore a weaker, higher-order signal, one that can be easily drowned out by experimental noise. In other words, this is better understood as a case where **the signal is too weak and the noise overwhelms it**, rather than evidence that representation alignment has no effect.

The opposite situation is not problem-free either. When the fingerprint used for splitting differs from the fingerprint used by the model, the test distribution may fail to shift in the direction that matters most to that model. The resulting evaluation may then fail to fully probe the model’s actual generalization challenge.

This is exactly the awkwardness of fingerprint + clustering splits:  
they either over-target certain models, or fail to truly test certain models.

So although fingerprint + clustering splits are more continuous than BM scaffold and easier to parameterize, they introduce another, subtler form of bias: **representation bias**. And once that bias overlaps with model bias, the evaluation result can be seriously distorted.

## What should a good split look like?

If we step back and abstract the discussion a little further, the real issue is not “whether BM scaffold is better than Butina” or vice versa. The real question is: **what properties should a good split have in the first place?**

To me, a good molecular dataset split should satisfy at least two criteria.

### It should be as representation-independent as possible

If a split depends too strongly on a specific representation, then it becomes difficult to evaluate models fairly. Since the model itself also operates in some representation space, any large overlap between split assumptions and model assumptions will inevitably influence the result.

This is precisely why, although I criticize BM scaffold for being coarse, I still think it has one important strength: **to a considerable extent, it is representation-independent.** It does not directly bind itself to a specific fingerprint, nor does it rely on any particular model’s feature space. Instead, it uses a human-interpretable structural rule to define the split. For that reason, it does not naturally become a targeted attack on one specific family of representations in the same way fingerprint-based splits can.

Of course, being representation-independent is not enough by itself. A good split should also be:

- **interpretable**: one should be able to explain what property is being separated between train and test
- **robust**: small structural changes should not induce excessively large jumps in split behavior
- **continuous**: ideally, it should express *how different* two molecules are, not just whether they belong to the same bucket

### It should be tunable

This is an idea I have come to believe more and more strongly:  
**split should not be just a few discrete benchmark options; it should form a split space.**

If we think of different models as occupying a model space, then different splits also occupy a space. A model performing well on one particular split only tells us that it performs well under that specific kind of distribution shift. It does not imply that it will perform well under all reasonable shifts.

In that sense, a model’s generalization ability should not be determined by a single fixed split. It should be characterized by its behavior over a family of controllable splits.

From this perspective, a “good split” may not be a single perfect protocol. It may be better understood as a **parameterized split framework**. By scanning over parameters, we can continuously control both the type and the strength of train/test difference. What we get is no longer a single benchmark point, but an entire split space. Evaluation then shifts from asking “what score does the model get on this split?” to asking “how does the model behave over an interpretable family of shifts?”

I believe that is much closer to what “generalization” should mean in molecular machine learning.

## A new idea: treating split as a tunable framework

With these considerations in mind, I have been trying to design a new molecular dataset splitting method. It is still at the prototype stage, but its goals are already quite clear: **preserve chemical interpretability as much as possible, reduce dependence on any particular representation, and provide continuously tunable parameters.**

At a high level, I decompose molecular difference into two layers:

- **global level**: differences in overall scaffold and topology
- **local level**: differences in local chemical composition and functional groups

The intuition is simple. Two molecules can be very similar in global backbone while differing substantially in local functional groups. They can also share similar local chemistry while differing dramatically in overall topology. A single monolithic definition of “molecular distance” tends to mix these two regimes together. It is therefore more natural to separate them first, then control their relative importance with a parameter.

### Global level: a looser and more continuous scaffold distance than BM scaffold

At the global level, I borrow the spirit of BM scaffold but do not use the standard BM scaffold directly.

My current definition is broader: **all ring systems, all carbon atoms, all multiple bonds, and the linkages connecting these elements** are included as part of the global scaffold. The purpose is to avoid the strong ring-centric bias of standard BM scaffold and improve the expressiveness for oligocyclic and acyclic molecules.

But the more important change is not what gets extracted; it is how comparison is performed.

I do not want the global feature to remain a discrete tag, because that would immediately bring back the main limitation of BM scaffold: it can only say “same” or “different,” but cannot tell us *how similar*. So I want the global-level similarity to be a **continuous quantity**.

More specifically, for a pair of molecules, I compute the maximum common substructure (MCS) on their global scaffolds, then measure how much of the larger scaffold is covered by that common part. This yields a continuous overlap-based indicator rather than a categorical tag.

A single MCS, however, also has limitations. It captures only the largest shared block, while potentially missing several other disconnected but still meaningful shared pieces. To mitigate this, I extend the procedure into an **iterative MCS** scheme:

1. find the first maximum common substructure  
2. mask the matched part  
3. run MCS again on the remaining part  
4. repeat for several rounds (currently, I use 3 rounds as the working design)

The resulting global similarity no longer depends on a single largest match, but can more fully describe the degree of overlap between two molecules at the level of global scaffold and topology.

### Local level: local chemical distance from functional-group vectors

If the global level asks whether the backbones look similar, then the local level asks whether the chemical compositions look similar.

My approach is to scan each molecule for functional groups and represent the result as a **functional-group count vector**. Each dimension corresponds to one functional group, and the value represents how many times that group appears in the molecule. Pairwise local distance is then computed from these vectors.

In the current prototype, I experimented with RDKit’s functional-group interface to generate such vectors. In practice, however, the hierarchy is not fully reliable. A more robust direction would likely be to replace it with a **custom SMARTS vocabulary**, i.e., define a clearer and more controllable functional-group dictionary with a more stable counting convention.

This local axis matters because similar global scaffolds do not imply similar local chemical environments. Functional-group vectors provide exactly the complementary information needed here. They allow the split to account not only for whether the backbone has changed, but also for whether the local chemical semantics have shifted.

### Fusion: using a parameter to control split preference

Once we have a global distance and a local distance, the most natural step is to combine them:

\[
D_{\text{total}}=\lambda D_{\text{global}}+(1-\lambda)D_{\text{local}}
\]

where \(\lambda \in [0,1]\) is a tunable parameter.

Its interpretation is very intuitive:

- when \(\lambda\) is large, the split is driven more strongly by global scaffold difference
- when \(\lambda\) is small, the split is driven more strongly by local functional-group difference

So \(\lambda\) is not just a mathematical coefficient. It is an interpretable **shift-preference knob**. By scanning \(\lambda\), one can continuously move from a split that emphasizes global structure to one that emphasizes local chemistry.

Once the total distance matrix is built, Butina clustering or another clustering method can be used to assign train/validation/test sets. In that sense, clustering itself becomes more neutral: it is no longer responsible for defining molecular similarity, but simply for grouping molecules in a distance space that has already been defined.

What I am really trying to build here is not another fixed split, but a **parameterized split family**. In the future, perhaps model performance should not be reported only as a score under one split, but as a curve over a range of \(\lambda\) and cutoff values. That would provide something closer to a profile of model behavior over split space.

## An interactive prototype: making the distance function tangible

To make this idea more than just an abstract formula, I also built a small interactive prototype:

**https://leodistance.streamlit.app/**

The goal of this demo is not to produce the final split directly, but to help me develop an intuition for what the distance function is actually doing. It allows me to sample pairs of molecules, inspect their structures, and compute their total distance under different values of \(\lambda\). Through this kind of visualization, I can directly observe:

- which molecular pairs are dominated by **global difference**
- which pairs are dominated by **local difference**
- how their relative ordering changes when \(\lambda\) varies

I value this interactive prototype a lot because it turns Stage 3 from a purely abstract design idea into something that can be inspected, tuned, and reasoned about.

This is important methodologically. For a tunable split to be meaningful, the parameter itself has to be **interpretable**. Otherwise, tunability is just “one more hyperparameter,” not an additional controllable dimension of distribution shift. The prototype helps me build a more intuitive sense of what \(\lambda\) really means: what exactly it means for a split to emphasize scaffold difference more strongly, and what it means to emphasize local chemistry instead.

Ultimately, I do not want a black-box distance that can only be computed. I want a split generator that can also be understood.

## What remains unresolved?

Of course, this idea is still far from a mature method, and several important issues remain unresolved.

The first is the definition of the **functional-group axis**. How should the vocabulary be designed? How should counting conventions be standardized? How should hierarchical relations between groups be handled? All of these details directly affect the stability and interpretability of the local distance.

The second is the interpretability of **\(\lambda\)** itself. Although its mathematical role is simple, its practical effect on split difficulty and split semantics may not be linear or immediately intuitive across datasets. In other words, \(\lambda\) is tunable, but determining what counts as a “reasonable” setting still requires experiments and accumulated intuition.

The third is **computational cost**. Iterative MCS is expensive, and once the dataset becomes large, the pairwise distance matrix itself is already a heavy object. If this direction is to scale to larger datasets, pruning, approximation, or pre-screening strategies will likely become necessary.

But these limitations do not weaken the value of the direction. If anything, they highlight the real point: the question is no longer whether some existing split is good enough, but whether split itself can be designed as an object that is controllable, interpretable, and worthy of study in its own right.

## Closing thoughts

The more I think about it, the more convinced I am that in molecular machine learning, generalization is never an absolute concept. A model does not possess some isolated, context-free “generalization ability.” Its generalization behavior is always defined relative to a particular kind of distribution shift, and data splitting is precisely the mechanism that explicitly or implicitly creates that shift.

From this perspective, split is not a minor detail in benchmarking. It is one of the core steps that define what the benchmark means in the first place.

The limitation of BM scaffold split is that it compresses continuous structural similarity into a rough discrete tag. The limitation of fingerprint + clustering split is that it depends too strongly on representation and can impose a targeted bias on models that share its representational assumptions. These two methods represent two highly instructive but incomplete philosophies: one emphasizes interpretability at the expense of continuity, while the other emphasizes continuity at the expense of neutrality.

My sense is that the more promising direction is not to keep arguing over which fixed split is “best,” but to construct a **split framework that is interpretable, parameterized, representation-aware, yet as representation-independent as possible**. Once split is transformed from a discrete benchmark into a tunable space, we may finally be in a position to answer a question that sounds simple but has never really been made explicit:

**relative to what, exactly, is a molecular model said to generalize?**
