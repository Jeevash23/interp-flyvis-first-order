# Literature and interpretation corrections

Sources checked on 7 October 2026. This is a focused source check, not an exhaustive novelty search.

| Source | Role in the revised study | Boundary |
| --- | --- | --- |
| [Lappalainen et al., Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/) | Released connectome-constrained model, model ensemble and previously identified tuning clusters. Figure 3 already reports opposite T4c preferred directions and a non-selective cluster. | Opposite clusters are prior work, not a new AI-06 discovery. Task success does not establish biological correctness. |
| [Mozer and Smolensky, NIPS 1, 1988 conference / 1989 proceedings](https://proceedings.neurips.cc/paper/1988/hash/07e1cd7dca89a1678042477183b7ac3f-Abstract.html) | Relevance assessment and unit-removal approximation lineage. | No claim to invent gradient-based removal estimates. |
| [Molchanov et al., ICLR 2017](https://arxiv.org/abs/1611.06440) | Taylor estimates of removal effects and comparisons with weight/activation criteria. | The present evaluation changes the model system and outcomes, not the approximation's provenance. |
| [Kramar et al., AtP*, 2024](https://arxiv.org/abs/2403.00745) | Related work on efficient attribution and false negatives in language models. | Language-model activation patching is not identical to recurrent output-gain silencing. |
| [Sundararajan et al., ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html) | Integrated gradients and completeness. | Along one cell's exact scalar intervention path, a converged derivative integral reproduces the endpoint change where the readout is well defined. It is not automatically a cheap independent predictor. |
| [Zhang and Wang, 2026 preprint](https://arxiv.org/abs/2606.09899) | Second-order corrections to attribution patching. | Verified as a preprint; not required for the scoped first-order evaluation. Transfer to this recurrence and nonsmooth readout would need separate qualification. |
| [Strother et al., Neuron 2017](https://doi.org/10.1016/j.neuron.2017.03.010) | T4-level biological intervention context; more relevant measured cells than tangential-cell assays. | Calcium and synaptic-blocking experiments still differ from voltage and acute gain attenuation. Prior FlyVis comparison prevents treating it as an untouched validation discovery. |
| [Braun, Borst and Meier, Current Biology 2023](https://doi.org/10.1016/j.cub.2023.05.007) | CT1-mediated inhibition and T5 directional tuning. | The model splits CT1 into two compartments; a cell-level model intervention must address both. No numerical biological match is claimed. |
| [Henning et al., eLife version of record, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13211874/) | C2/C3 feedback and motion processing. | Earlier audit cited the 2025 reviewed preprint. The current record is dated 26 May 2026; retain version-specific dates. |

The original Stage 1 report was already explicit about finite-set inference, biological assay mismatch, and the distinction between ranking and effect-size accuracy. The previous audit's blanket statement that interpretations were wrong was too broad. The revision preserves those qualifications while adding the missing comparisons and correcting denominator handling.

The proposed contribution is a transparent evaluation of screening usefulness and effect-size reliability across one released connectome-constrained ensemble, with explicit endpoint failures and stronger anatomical baselines. Neither additional model count nor more methods guarantees publication suitability. Venue and title should follow the final results. No claim that the study is the first of its kind is made.
