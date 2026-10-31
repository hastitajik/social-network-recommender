# Friend Recommendation with Link Prediction, Clustering and a Genetic Algorithm

A single-file Python implementation of:

> Lu et al. **A recommender system for social networks using link prediction, clustering and genetic algorithm.** *Egyptian Informatics Journal* 32 (2025) 100784.

## Pipeline

1. **Profile preprocessing** (§3.2.1): age, gender, language, occupation and education encoded into a 5-dimensional vector; age min–max normalised.
2. **Edge weighting** (Eq. 3): `W(i,j) = 1 / [α·S(Pᵢ,Pⱼ) + (1−α)·S(Cᵢ,Cⱼ)]`, mixing cosine similarity of profiles and of connection vectors.
3. **Community detection** (§3.2.2): minimum spanning tree (Prim) → remove the heaviest ⌊(N−1)/2⌋ edges → greedily merge local communities while modularity increases. New users join the nearest cluster centre.
4. **Link prediction** (Eq. 8): inside each cluster, count loop-free paths of length 2…L between non-connected users, attenuated by 1/(i−1) and normalised by the number of possible paths. Top-k candidates become recommendations. Similarity matrices are computed once per cluster.
5. **Genetic algorithm** (§3.2.4): 8-bit chromosomes encode the number of clusters and the path length; fitness is the mean of precision and recall; elitism, tournament selection, two-point crossover and bit-flip mutation.
6. **Evaluation**: 10-fold cross-validation over edges with accuracy, precision, recall and F1 (Eq. 9–12).

## Data

The paper uses the Pokec social network. This repository runs on a **synthetic** network by default: a stochastic block model (400 users, 6 communities, average degree ≈ 10.6) with community-correlated profiles, so the pipeline can be tested without downloading Pokec. `generate_synthetic_dataset` can be swapped for a loader that returns a NetworkX graph and a profile DataFrame with the same columns.

## Results on the synthetic network (10-fold CV, top-10, L = 3)

| Metric | Mean | Std |
|---|---|---|
| Precision@10 | 0.027 | 0.003 |
| Recall@10 | 0.166 | 0.019 |
| F1 | 0.045 | 0.005 |
| Accuracy | 0.972 | 0.000 |

Accuracy is high mainly because true negatives dominate; precision/recall are the informative metrics. The synthetic graph leaves only one or two held-out links per test user, which caps precision@10. These numbers are not comparable to the paper's Pokec results.

## Run it

```bash
pip install -r requirements.txt
python recommender.py          # or open social_recommender.ipynb
```

Set `run_ga = True` in `main()` to search the number of clusters and path length with the genetic algorithm instead of using the paper's reported optimum (9 clusters, path length 3).

## Author

Hasti Tajik — M.Sc. Data Science, University of Science and Culture
