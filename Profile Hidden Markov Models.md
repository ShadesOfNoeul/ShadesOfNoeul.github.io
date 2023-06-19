# Position Specific Scoring Matrices (PSSM)

###### A sequence profile is a position-specific scoring matrix (or PSSM) that gives a quantitative description of a sequence motif

###### A simple PSSM is a Log odds scoring matrix that has as many columns as there are positions in the alignment, and either 4 rows (one for each DNA nucleotide) or 20 rows (one for each amino acid)

![[Pasted image 20230613203124.png]]

## Problems with PSSMs

###### PSSMs work well for fixed length motifs in which the sites are more or less independent - i.e., ungapped motifs

###### However there are other kinds of motifs for which PSSMs are not well suited

###### PSSMs cannot

 1. model positional dependencies
 2. recognize pattern instances containing insertions or deletions
 3. model variable length patterns
 4. detect boundaries
![[Pasted image 20230613203234.png]]

### 1. Positional dependencies

Do not capture positional dependencies
![[Pasted image 20230613203326.png]]
     Note: We never see QD or RH, we only see RD and QH.
     However, P(RH)=0.24, P(QD)=0.24, while P(QH)=0.16

### 2. Insertions and deletions

Hard to recognize pattern instances that contain indels
![[Pasted image 20230613203400.png]]

### 3. Variable length motifs

Cannot easily deal with variable length motifs
![[Pasted image 20230613203416.png]]

### 4. Detecting boundaries

Do not handle boundary detection problems well
 E.g. Label every element in the sequence with a 0 (not in pattern) or a 1 (in pattern)

Examples of boundary detection problems include:

- Recognition of regulatory motifs
- Recognition of protein domains
- Intron/exon boundaries
- Gene boundaries
- Transmembrane regions
- Secondary structure elements (helices and strands)

![[Pasted image 20230613203526.png]]

These shortcomings of PSSMs set the stage for a new kind of profile,
based on Markov chains, called **Hidden Markov models (HMMs)**

# Markov chains

**Markov chains** are stochastic processes that undergo transitions between a
finite series of states in a chainlike manner.

The system transverses states with probability
$$p(x_1, x_2, x_3, ...) = p(x_1) p(x_2| x_1) p(x_3| x_2) p(x_4| x_3)...$$

i.e. **Markov chains are memoryless**: the probability that the chain is in state xi at time t, depends only on the state at the previous time step and not on the past history of the states visited before time t−1.

This specific kind of "memorylessness" is called the **Markov property**.

> [!info] The **Markov property** states that:
> the conditional probability distribution for the system at the next step (and in fact at all future steps) depends only on the current state of the system, and not additionally on the state of the system at previous steps.

Markov chains, and their extension hidden Markov models (HMMs), are commonly represented by **state diagrams**, which consist of states and connecting *transitions*

![[Pasted image 20230613204023.png]]

A **transition probability** parameter ($a_{ij}$) is associated with each transition (arrow) and
determines the probability of a certain state ($S_j$) following another state ($S_i$).

> [!info] A **Markov chain** is defined by:
• a finite set of states, $S_1, S_2 ...S_N$
• a set of transition probabilities: $a_{ij} = P(q_{t+1}=S_j|q_t=S_i)$
• and an initial state probability distribution, $π_i = P(q_0=S_i)$

#### Simple Markov chain example for $x=\{a,b\}$

Observed sequence: $x = abaaababbaa$
Model:
![[Pasted image 20230613204330.png]]

**P(x) = 0.5 x 0.3 x 0.5 x 0.7 x 0.7 x 0.3 x 0.5 x 0.3 x 0.5 x 0.5 x 0.7**

- [ ] Q. Can you sketch the state diagram with labeled transitions for this model?

> [!info] Typical questions we can ask with Markov chains include:
>
> - What is the probability of being in a particular state at a particular time?
> (By time here we can read position in our query sequence)
> - What is the probability of seeing a particular sequence of states?
> (I.e., the score for a particular query sequence given the model)

- [ ] Q. What do Markov chains add over the traditional PSSM approach?
In particular how do Markov chains deal with the following PSSM weaknesses?
>
> 1. Positional dependencies
> 2. Pattern instances containing insertions or deletions
> 3. Variable length patterns, and
> 4. The detection boundaries (i.e. segmentation of sequences)

## 1. Positional dependencies

The connectivity or topology of a Markov chain can easily be designed to capture dependencies and variable length motifs.
![[Pasted image 20230613205103.png]]
Recall that a PSSM for this motif would give the sequences WEIRD and WEIRH equally good scores even though the RH and QR combinations were not observed

## 2. Pattern instances containing insertions or deletions

To address pattern instances with gaps and variable length motifs, we can construct a Markov chain to recognize a query sequences with insertions (via an extra insertion state) and deletions (via extra transitions (edges))
![[Pasted image 20230613205126.png]]

## 3. Boundary detection

Giving a sequence we wish to label each symbol in the sequence according to its class (e.g. transmembrane regions or extracellular/cytosolic)
![[Pasted image 20230613205149.png]]
Given a training set of labeled sequences we can begin by modeling each amino acid as hydrophobic (H) or hydrophilic (L)
 i.e. reduce the dimensionality of the 20 amino acids into two classes

E.g., A peptide sequence can be represented as a sequence of Hs and Ls.
 e.g. HHHLLHLHHLHL...

**Is a given sequence a transmembrane sequence?**
A Markov chain for recognizing transmembrane sequences
![[Pasted image 20230613205241.png]]

Question: Is sequence HHLHH a transmembrane protein?
  P(HHLHH) = 0.6 x 0.7 x 0.7 x 0.3 x 0.7 x 0.7 = 0.043

Problem: need a threshold,
  threshold must be length dependent
  
###### We can classify an observed sequence (O = O1, O2, ...) by its log odds ratio

![[Pasted image 20230613205347.png]]

In other words, it is more than twice as likely that **HHLHH** is a transmembrane sequence. The log-odds score is: log2(2.69) = 1.43

#### Side note: Parameter estimation

Both initial probabilities ($π(i)$)and transition probabilities ($a_{ij}$) are determined from known examples of transmembrane and non-transmembrane sequences.
![[Pasted image 20230613205435.png]]
Given labeled sequences (TM and E/C), we determine the initial probabilities $π(i)$ by
counting the number of sequences that begin with residue $i$.

To determine transition probabilities, $a_{ij}$, we first determine $A_{ij}$ (the number of transitions from state $i$ to $j$ in the training data, i.e. count the number of $ij$ pairs in the training data). Then normalize by the number of $i^*$ pairs.
![[Pasted image 20230613205501.png]]
Both initial probabilities (π(i))and transition probabilities (aij) are determined from
known examples of transmembrane and non-transmembrane sequences.
![[Pasted image 20230613205601.png]]

### Boundary detection challenge

**Given sequence of Hs and Ls, find all transmembrane regions:**

>Using our Markov models we would still need to score successive overlapping windows along the sequence, leading to a fuzzy boundary (just as with a PSSM).

To approach this question we can construct a new four state model by adding
transitions connecting the TM and E/C models
![[Pasted image 20230613205647.png]]
In a Markov chain, there is a one-to-one correspondence between symbols and
states, which is not true of our new merged four state, two symbol model.

For example, both $H_M$ and $H_{E/C}$ are associated with hydrophilic residues.

- This four-state transmembrane model is a **hidden Markov model**
![[Pasted image 20230613205729.png]]

### So what's hidden?

We will distinguish between the *observed* parts of the problem and the *hidden*
parts

- In the Markov models we have considered previously it is clear which states account for each part of the observed sequence
 Due to the one-to-one correspondence between symbols and states
- In our new model, there are multiple states that could account for each part of the observed sequence
 i.e. we don’t know which state emitted a given symbol from knowledge of the sequence and the structure of the model
  -> This is the hidden part of the problem
![[Pasted image 20230613205836.png]]
**For our Markov models**
- Given HLLH..., we know the exact state sequence (q0=SH, q1=SL, q2=SL, ...)
**For our HMM**
- Given HLLH..., we must infer the most probable state sequence
- This HMM state sequence will yield the boundaries between likely TM and E/C regions
![[Pasted image 20230613205908.png]]

### Hidden Markov models (HMMs)

![[Pasted image 20230613205929.png]]

### Example three state HMM

In this example we will use only one state for the transmembrane segment (M) and use emission probabilities to distinguish between H and L residues. We will also add separate E & C states with distinct emission probabilities.
![[Pasted image 20230613205951.png]]

#### Side note: Parameter estimation

As in the case of Markov chains, the HMM parameters can be learned from labeled training data

Note that we now have to learn the initial probabilities, transition probabilities and emission probabilities

![[Pasted image 20230613210056.png]]

![[Pasted image 20230613210116.png]]

![[Pasted image 20230613210147.png]]

### Viterbi algorithm

The **Viterbi algorithm** finds the most probable “state path” ($S^*$) (i.e. sequence of hidden states) for generating a given sequence ($x= x_1, x_2,...x_N$)
$$S^* = argmax P(x,S)$$
This process is often called decoding because we “decode” the sequence of symbols to determine the hidden sequence of states
HMMs were originally developed in the field of speech recognition, where speech is "decoded" into words or phonemes to determine the meaning of the utterance

Note that we could have used brute force by calculating $P(x|S)$ for all paths but this quickly becomes intractable for longer sequences or HMMs with a large number of states

> [!info] The Viterbi algorithm is guaranteed to find the most probable state path given a sequence and an HMM

# Three key HMM algorithms

## Viterbi algorithm

Given observed sequence x and an HMM M, composed of states S, calculate
the most likely state sequence, S*
-> $S^* = argmax P(x,S)$

## Forward algorithm

Given observed sequence x and an HMM composed of states S, calculate the
probability of the sequence for the HMM, $P(x|M)$
-> $P(x) = \sum_S P(x, S)$

### How well does a given sequence fit the HMM?

To answer this question we must sum over all possible state paths that are consistent with the sequence in question
(Because we don't know which path emitted the sequence)

The number of paths can quickly become intractable. The **forward algorithm** is a simple dynamic programming solution that makes use of the Markov property so that we don’t have to explicitly enumerate every path.

The **forward algorithm** basically replaces the maximization step of the Viterbi algorithm with sums to calculate the probability of the sequence given a HMM

## Baum-Welch algorithm

Given many observed sequences, estimate the parameters of the HMM
-> heuristic expectation maximization method to optimize of $a_{ij}$ and $e_i(a)$

The Baum-Welch algorithm is an **heuristic optimization** algorithm for learning
probabilistic models in problems that involve hidden states

If we *know* the state path for each training sequence (i.e. no hidden states with respect to the training sequences), then learning the model parameters is simple (just like it was for Markov chain models)

- count how often each transition and emission occurs
- normalize to get probabilities

If *we don’t know* the path for each training sequence, we can use the **Baum-Welch algorithm**, an expectation maximization method, which estimates counts by considering every path weighted by its probability

- start from a given initial guess for the parameters
- perform a calculation which is guaranteed to improve the previous guess
- run until there is little change in parameters between iterations

For sequence profile-HMMs we train from a MSA and hence we can estimate our probabilities from the observed sequences

##### Segmentation/boundary detection

**Given**: A test sequence and a HMM with different sequence classes
**Task**: Segment the sequence into subsequences, predicting the class of each subsequence
**Question**: What is the most probable “path” (sequence of hidden states) for generating a given sequence from the HMM?
**Solution**: Use the Viterbi algorithm

##### Classification/sequence scoring

**Given**: A test sequence and a set of HMMs representing different sequence classes
**Task**: Determine which HMM/class best explains the sequence
**Question**: How likely is a given sequence given a HMM?
**Solution**: Use the Forward algorithm

##### Learning/parameterization

**Given**: A model, a set of training sequences
**Task**: Find model parameters that explain the training sequences
**Question**: Can we find a high probability model for sequence characterization
**Solution**: Use the Forward backward algorithm

## HMM limitations

HMMs are linear models and are thus **unable to capture higher order correlations** among positions (e.g. distant cysteins in a disulfide bridge, RNA secondary structure pairs, etc).

Another flaw of HMMs lies at the very heart of the mathematical theory behind these models. Namely, that the probability of a sequence can be found from the product of the probabilities of its individual residues.

This claim is only valid if the probability of a residue is independent of the probabilities of its neighbors. In biology, there are frequently **strong dependencies between these probabilities** (e.g. hydrophobic residues clustering at the core of protein domains).

These biological realities have motivated research into new kinds of statistical models. These include hybrids of HMMs and neural nets, dynamic Bayesian nets, factorial HMMs, Boltzmann trees and stochastic context-free grammars.
