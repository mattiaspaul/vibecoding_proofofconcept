## Proof-of-concept

This stand in the MICCAI 2026 vibe coding tutorial demonstrates the use of LLMs for proof-of-concept research exploration on the example of foundation features with a non-trainable cross-image attention module that directly acts on transformation coordinates for medical image registration.

### Part 1: Literature review

General recommendations:
- Use multiple tools and cross-verify with actual PDFs.
- Supplement with direct search queries using LLM-suggested terms.
- Prompt strictly to prevent the LLM from introducing unsourced ideas.

:



> **Suggested prompt**
> 
> *"I am doing a literature review regarding novelty of a new module within medical image registration, basically I want to **tackle large deformations in a single step** by 1) feeding in moving and fixed images from a shared (foundation) feature extractor and 2) modelling cross similarities densely over a certain local neighbourhood mask without any trainable weights 3) feed a coordinate grid as values to directly obtain softmax weighted displacement prediction. Can you please verify whether this is novel or list the five most related prior works to be considered.  Keep your answer very concise - approx. 750 words."*
> 
> — **Mattias Heinrich**, *thinking about medical image registration (2026)*

<details>
<summary><b>ChatGPT </b></summary>
I would **not claim the overall method is novel yet**. The strongest conclusion from the literature search is more nuanced: **each of your three ingredients has substantial precedent, but their particular combination for single-step medical deformable registration appears much less directly represented.** In particular, the “softmax correlation → weighted coordinate grid → displacement” component is already established in optical-flow/correspondence literature, so that part alone cannot carry the novelty claim. ([IET][1])

### Five prior works I would treat as essential

1. **DINO-Reg — Song et al., 2024/2025**
   This is the **closest medical-registration precedent**. DINO-Reg uses a frozen DINOv2 foundation encoder for feature extraction and performs **training-free deformable medical image registration**. Thus, your points (1) and “no trainable registration weights” are very close conceptually. Their registration, however, uses a **discrete optimizer** to determine the deformation field rather than directly computing a local softmax-weighted coordinate expectation. ([arXiv][2])
   **This is the paper you most need to differentiate from.**

2. **GMFlow — Xu et al., 2022, “Learning Optical Flow via Global Matching”**
   This is probably the closest methodological precedent outside medical imaging. GMFlow explicitly reframes flow as **feature correlation → softmax matching distribution → differentiable matching**, rather than conventional learned flow regression. Its motivation is also directly relevant to your objective: local cost volumes constrain the search range and therefore struggle with large displacements. ([ResearchGate][3])
   Your proposed local-neighbourhood version therefore needs a clear argument for why local matching on foundation features can recover *large* deformation in one step.

3. **LoFTR — Sun et al., CVPR 2021**
   LoFTR establishes **dense/semi-dense correspondence from image features**, avoiding explicit feature detection, and uses correspondence distributions to establish pixel-level matches. It is not medical registration and uses learned Transformer-based cross-attention, but conceptually it is a very important precedent for “feature representations → dense correspondence.” ([Open Access CVF][4])

4. **KeyMorph — Yu et al., MIDL 2022; Wang et al., MedIA 2023**
   KeyMorph is particularly relevant to your **large-displacement** motivation. It argues that conventional learned registration can be vulnerable to large misalignment and instead obtains transformations from corresponding image-derived keypoints using a differentiable closed-form operation. The later version supports nonlinear/spline registration. ([Proceedings of Machine Learning Research][5])
   The distinction is important: KeyMorph predicts/learns **keypoints**, whereas you propose dense correspondence at every location.

5. **VoxelMorph — Balakrishnan et al., CVPR 2018**
   This is the canonical baseline for learned one-shot deformable medical registration: an image pair is mapped directly to a dense deformation field. It is therefore useful mainly as the **“single-pass registration” precedent**, although unlike your proposal its deformation predictor is a trained CNN. ([arXiv][6])

### Where I think your novelty actually lies

I would formulate the potential contribution **not** as “training-free registration,” “foundation features,” “dense matching,” or “softmax displacement,” because those components individually have clear precedents.

The potentially novel combination is:

> **Frozen/shared foundation features + explicitly restricted local correlation matching + parameter-free softmax expectation over a coordinate grid + direct dense deformation prediction in a single inference step, specifically designed to handle large medical-image deformations.**

The important caveat is that **local correlation + softmax + coordinate expectation is itself not new**. A recent optical-flow formulation explicitly describes converting correlations into a softmax matching distribution and multiplying that distribution by a coordinate grid to obtain expected correspondences. ([ResearchGate][7]) Earlier registration/homography work likewise uses a local correlation volume, softmax matching probabilities, local candidate coordinates, and feature flow. ([IET][1])

Consequently, your strongest novelty claim would have to rest on **the interaction of foundation representations and this parameter-free local matching mechanism in medical deformable registration**, particularly whether it genuinely resolves large deformations **without iterative refinement, learned correspondence heads, or an optimization stage**.

One particularly important concern for the paper is the apparent tension between **“local neighbourhood” and “large deformation.”** If the correspondence search is genuinely bounded to a radius \(r\), a displacement larger than \(r\) cannot be recovered in one step unless the foundation features themselves provide some mechanism for bringing corresponding structures within that neighbourhood (e.g., coarse-scale features, multi-resolution features, or semantically aligned representations). This should probably become an explicit part of your theoretical/experimental argument.

**Bottom line:** I see a **credible novelty hypothesis, but not evidence for claiming fundamental algorithmic novelty yet**. The closest prior art is overwhelmingly **DINO-Reg + GMFlow/local correlation matching**, and your paper will need an explicit component-by-component distinction from both. ([PubMed][8])

[1]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.12842?utm_source=chatgpt.com "Unsupervised deep homography with multi‐scale global attention - Hu - 2023 - IET Image Processing - Wiley Online Library"
[2]: https://arxiv.org/abs/2402.15687?utm_source=chatgpt.com "General Purpose Image Encoder DINOv2 for Medical Image Registration"
[3]: https://www.researchgate.net/publication/356601872_GMFlow_Learning_Optical_Flow_via_Global_Matching?utm_source=chatgpt.com "(PDF) GMFlow: Learning Optical Flow via Global Matching"
[4]: https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html?utm_source=chatgpt.com "CVPR 2021 Open Access Repository"
[5]: https://proceedings.mlr.press/v172/yu22a.html?utm_source=chatgpt.com "KeyMorph: Robust Multi-modal Affine Registration via Unsupervised Keypoint Detection"
[6]: https://arxiv.org/abs/1809.05231?utm_source=chatgpt.com "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
[7]: https://www.researchgate.net/publication/405076459_Rethinking_Dense_Optical_Flow_without_Test-Time_Scaling?utm_source=chatgpt.com "(PDF) Rethinking Dense Optical Flow without Test-Time Scaling"
[8]: https://pubmed.ncbi.nlm.nih.gov/40327463/?utm_source=chatgpt.com "DINO-Reg: Efficient Multimodal Image Registration With Distilled Features - PubMed"

</details>
<details>
<summary><b>Gemini 3.6 Flash</b></summary>
### Novelty Assessment

The specific combination of steps you described is **partially novel as a unified module**, but **the underlying mechanics are heavily anchored in established prior work** across medical registration and computer vision.

* **Where it is NOT novel:**
1. Using **dense correlation volume matching followed by a non-trainable spatial expectation / Softmax weighting over a local window** to derive displacement fields directly from a coordinate grid is a standard classical/hybrid technique.
2. Feeding features from **shared foundation backbones** (e.g., SAM, DINOv2) to drive zero-shot or non-trainable registration is widely used in current literature.


* **Where potential novelty lies:**
Applying this non-trainable, expectation-based grid aggregation **strictly as a single-step large-deformation module** directly fed by frozen medical foundation features (without learning any intermediate displacement regressors, spatial transformer networks, or iterative optimization steps).

---

### Top 5 Related Prior Works to Consider

#### 1. **VoxelMorph & Learning-Based Deformable Registration**

* **Reference:** Balakrishnan et al., *"VoxelMorph: A Learning Framework for Deformable Medical Image Registration"* (IEEE TMI, 2019).
* **Relevance:** The primary baseline for deep learning-based displacement field generation. While VoxelMorph uses trainable ConvNet decoders to regress spatial transformation fields, your module replaces the decoder entirely with a non-trainable dense cross-correlation + expectation operation on feature grids.

#### 2. **PWC-Net / RAFT (Dense Feature Matching via Correlation Layers)**

* **Reference:** Sun et al., *"PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume"* (CVPR, 2018) / Teed & Deng, *"RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"* (ECCV, 2020).
* **Relevance:** Foundational literature for local cost volume / dense correlation computation over search windows. These models compute dense similarity matrices across local neighborhoods, which directly mirrors your second step.

#### 3. **Soft Argmax / Softmax Weighted Coordinate Aggregation**

* **Reference:** Kendall et al., *"End-to-End Learning of Geometry and Context for Deep Stereo Regression"* (GC-Net, ICCV 2017) / Luvizon et al. (2018).
* **Relevance:** Directly establishes your step (3): using non-trainable Softmax probabilities computed over cost/correlation volumes to perform a weighted sum over target coordinate grids (soft argmax) to output continuous displacement fields directly without trainable weights.

#### 4. **uniGradICON / Foundation Models for Registration**

* **Reference:** Tian et al., *"uniGradICON: A Foundation Model for Medical Image Registration"* (MICCAI, 2024 / arXiv 2024).
* **Relevance:** Represents the current state-of-the-art in single-step, large-deformation medical registration using foundation models. It demonstrates how step (1) (extracting deep representations via shared foundation models) can generalize across large displacements, serving as your primary competitor/benchmark.

#### 5. **Discrete Optimization & Correlation-Based Medical Registration**

* **Reference:** Heinrich et al., *"MIND: Modality-Independent Neighborhood Descriptor for Multi-Modal Image Registration"* (Medical Image Analysis, 2012) & *Deeds* (2013).
* **Relevance:** Shows that computing local self-similarity descriptors (or feature maps) and searching dense local neighborhoods via non-parametric cost functions to directly construct deformation grids is a classical, highly effective approach for large deformations.

---

### Framing Recommendations for Your Literature Review

1. **Highlight the "Parameter-Free" Nature:** Frame your approach as a *non-parametric non-iterative correspondence decoder*. Emphasize that while existing approaches rely on heavy decoder networks (e.g., VoxelMorph, TransMorph) or iterative energy minimization, your method extracts displacements analytically via local soft-argmax over foundation features.
2. **Address Large Deformations Explicitly:** Standard local-window cross-attention fails under extreme motion if the search neighborhood is too small. You should justify why your local neighborhood mask is wide enough to capture large displacements without becoming computationally intractable ($O(N \cdot K^3)$ where $K$ is the window size).

</details>

<details>
<summary><b>Qwen 3.7 Plus</b></summary>
### Novelty Verdict
Your proposed module is **novel in its specific integration**, but its individual components have strong prior art. The true novelty lies in achieving **large deformations in a single step** using this exact pipeline. Prior works that use weightless softmax-weighted coordinate grids universally rely on multi-step, coarse-to-fine pyramids to handle large deformations, while single-step large-deformation methods lack this sophisticated feature-matching mechanism. 

To solidify your novelty claim, your literature review must explicitly contrast your single-step capability against the multi-step limitations of the following five most related prior works.

---

### Top 5 Most Related Prior Works

**1. Vector Field Attention (VFA) for Deformable Image Registration (2024)**  
- **Similarity**: Uses local neighborhood cross-similarity (inner product) with *no trainable weights* in the matching step. It applies a Softmax operation to create an attention map and uses a fixed coordinate grid (vector field) as the value matrix to directly retrieve a softmax-weighted displacement prediction.  
- **Difference**: VFA explicitly relies on a **multi-resolution coarse-to-fine strategy**. It cannot handle large deformations in a single step, as the local search window (e.g., 3×3×3) is too small for large displacements without prior coarse alignment.

**2. Coordinate Translator for Learning Deformable Medical Image Registration (im2grid, 2022)**  
- **Similarity**: Computes cross-correlation between fixed and moving features over candidate coordinates, normalizes matching scores via Softmax, and calculates a weighted sum of candidate coordinates to directly output a sampling grid (displacement).  
- **Difference**: Typically employs Siamese (separate weight) encoders rather than a strictly shared foundation extractor. More importantly, it uses a **coarse-to-fine hierarchical composition** of multiple Coordinate Translators, not a single-step large-deformation solution.

**3. Learning a Metric for Multimodal Medical Image Registration (Siebert et al., 2022)**  
- **Similarity**: Explicitly uses a **shared feature extractor** (SharedNet) and a dense correlation layer with **zero trainable weights** over a discrete displacement grid (11×11×11). It applies Softmax to the correlation volume to obtain differentiable, soft displacement correspondences.  
- **Difference**: This method is designed to compute a **rigid transformation** (via differentiable least-squares fitting), not a dense deformable displacement field. 

**4. CARL: A Framework for Equivariant Image Registration (Greer et al., 2024/2025)**  
- **Similarity**: Leverages a shared, translation-equivariant feature encoder and a "coordinate attention" layer that computes softmax-weighted correspondences to tackle large initial displacements.  
- **Difference**: The authors explicitly note that a single step is insufficient for very large deformations. CARL wraps the coordinate attention mechanism in a **TwoStep (coarse-to-fine) refinement framework**, using subsequent convolutional networks to refine the displacement field.

**5. Intermediate Deformable Image Registration via Windowed Cross-Correlation (Aganj et al., 2023)**  
- **Similarity**: Specifically designed to tackle **large deformations in a single step** by computing cross-correlation over a massive local neighborhood (up to half the image size), avoiding local optima traps.  
- **Difference**: Operates directly on raw image intensities. It does not utilize shared foundation model feature extractors, nor does it employ a softmax-weighted coordinate grid mechanism for displacement regression.

---

### Strategic Recommendation for Your Review
To maximize the perceived novelty of your work, frame your contribution around the **"single-step large deformation gap"**. 

State clearly: *"While weightless, softmax-weighted coordinate regression has been explored (e.g., VFA, im2grid), these methods are fundamentally limited to small deformations per step, requiring computationally expensive coarse-to-fine pyramids. Conversely, single-step large-deformation methods (e.g., Aganj et al.) lack the semantic robustness of foundation features. Our module is the first to bridge this gap, leveraging a shared foundation extractor and weightless dense cross-similarity to resolve large deformations in a single, direct softmax-weighted coordinate prediction."* 

This framing directly preempts reviewer concerns by acknowledging prior art while sharply defining your unique architectural contribution.
</details>

<details>
<summary><b>Perplexity AI</b></summary>
## Verdict

The **overall combination is not novel**. Your proposed module appears to reproduce, or closely interpolate, several established ideas:

1. shared/frozen feature extraction;
2. parameter-free dense correlation or cost-volume construction;
3. local discrete displacement candidates;
4. softmax conversion into correspondence probabilities; and
5. expectation of candidate coordinates/displacements (“soft-argmax”) to obtain a field in one feed-forward step.

The strongest prior-art overlap is with probabilistic dense displacement networks, rather than with foundation-model registration specifically. Your potentially defensible novelty would therefore need to lie in the **specific foundation encoder, neighborhood/search design, large-deformation coverage, computational formulation, or demonstrated performance**, not in the basic module structure.

## Five closest prior works

| Work | Relevant overlap | Difference from your proposal |
|---|---|---|
| **Heinrich, “Closing the Gap between Deep and Conventional Image Registration using Probabilistic Dense Displacement Networks” (2019)** | Uses shared feature extraction, a trainable-free correlation layer, densely quantised displacement labels, softmax displacement probabilities, and a weighted average of displacement labels to produce the deformation field. It explicitly targets large deformations with a single discrete prediction stage.  [arxiv](https://arxiv.org/pdf/1907.10931.pdf) | Uses learned 3D features and additional min-convolution/mean-field regularisation; its search is generally a larger discrete displacement space rather than only a local mask. |
| **Siebert, Hansen & Heinrich, “Learning a Metric for Multimodal Medical Image Registration without Supervision Based on Cycle Constraints” (2022)** | Describes a feature-extraction/correlation/registration architecture in which the correlation layer has no trainable weights and can directly yield displacement probabilities. It uses shared feature-processing layers, patchwise SSD, a discrete displacement set, softmax correspondences, and large capture ranges.  [mdpi](https://www.mdpi.com/1424-8220/22/3/1107) | Primarily estimates robust rigid transformations and learns the feature extractor through cycle constraints; it is not a frozen foundation-feature pipeline or dense deformable output. |
| **DINO-Reg (Song, Xu & Yan, MICCAI 2024)** | Introduces training-free multimodal deformable registration using a general-purpose frozen DINOv2 encoder. It demonstrates the closest precedent for replacing task-specific feature learning with foundation features.  [papers.miccai](https://papers.miccai.org/miccai-2024/231-Paper2230.html) | Registration is performed through feature pairing and iterative gradient-based/discrete optimisation, rather than your direct local cost-volume softmax expectation. |
| **FlowNet-C / volumetric correspondence networks** | FlowNet-C established the parameter-free correlation layer over a quantised displacement window. Subsequent volumetric correspondence work uses local cost volumes and soft-argmin/soft-argmax, i.e. weighted displacement expectations from softmax-normalised matching scores.  [arxiv](https://arxiv.org/pdf/1907.10931.pdf) | Primarily optical-flow literature, but the computational mechanism is directly transferable and should be cited as foundational prior art. |
| **Blendowski et al., “Weakly-supervised learning of multi-modal features for regularised iterative descent in 3D image registration” (2021)** | Closely related to shared multimodal feature learning followed by correspondence/cost-volume registration and iterative deformation estimation; it is also cited as a predecessor in the 2022 correlation-based work.  [mdpi](https://www.mdpi.com/1424-8220/22/3/1107) | Features are learned and registration remains iterative; it does not use a frozen foundation encoder with a direct coordinate-grid expectation. |

## Where novelty may remain

A careful claim could be:

> “A training-free or frozen-feature, single-pass deformable registration head that computes local dense cross-correlation and directly predicts displacement as the softmax expectation of coordinate-grid values.”

Even this should be stated cautiously. The **softmax-weighted coordinate-grid prediction** is essentially the probabilistic displacement formulation already present in Heinrich’s 2019 work, where displacement labels are weighted by predicted probabilities to obtain the continuous field.  The 2022 work independently makes the same design especially explicit for medical registration, including a trainable-free correlation layer and softmax correspondences. [arxiv](https://arxiv.org/pdf/1907.10931.pdf)

The more plausible contribution is the **combination** of that head with frozen foundation features and an appropriately designed local search mask. DINO-Reg makes the foundation-feature component non-novel in isolation, although its optimisation strategy differs. [papers.miccai](https://papers.miccai.org/miccai-2024/231-Paper2230.html)

## Recommendation

Do not frame the contribution as inventing dense correlation, softmax displacement inference, or single-step discrete registration. Instead, position it as a **new registration head or efficient synthesis of known components**, and establish novelty through:

- exact comparison with probabilistic dense displacement networks;
- comparison with DINO-Reg and other frozen-feature methods;
- ablations of local versus global search masks;
- capture-range analysis for large deformations;
- proof that the coordinate-grid expectation is differentiable and computationally cheaper;
- evaluation against iterative optimisation and coarse-to-fine baselines.

A novelty claim may survive if your method introduces a genuinely new **mask construction, search-centering mechanism, foundation-feature adaptation, uncertainty treatment, or regularity guarantee**. Based on the described ingredients alone, however, I would classify it as **incrementally novel at the system level, but not conceptually novel at the module-operation level**.
</details>

---

Have a look at individual responses from different models (all receiving the same prompt) and see how some LLMs slightly lacked relevance - DINOreg, VoxelMorph and conventional MIND+deeds registration are all not directly related to the research question. From the acutally relevant papers, there are just few that overlap - which highlights the need to explore multiple tools. 

Interestingly, of all four LLMs only Qwen 3.7 provided the papers I had in mind myself. VFA/im2grid and CARL. When asked, Gemini suggested another Google scholar search that resulted in a few more relevant hits (incl. again VFA).

<details>
<summary><b>suggested search terms </b></summary>
``("medical image registration" OR "deformable registration") AND ("large deformation" OR "large displacements") AND ("foundation model" OR "DINOv2" OR "SAM" OR "feature extractor") AND ("dense correlation" OR "cost volume" OR "cross similarity" OR "local neighborhood") AND ("softmax" OR "soft argmax" OR "expectation" OR "non-trainable" OR "parameter-free")``
</details>  

---

From here we will continue with "Coordinate Translator for Learning Deformable Medical Image Registration (im2grid, 2022)" by Yihao Liu, which is a training-free, differentiable module that matches features between fixed and moving images to output coordinate correspondences directly. It relieves convolutional neural networks (CNNs) from learning spatial coordinate systems, letting them focus purely on feature extraction. Just as we anticipated.


## Part 2: Proof-of-concept implementation

The next prompt should get us kickstarted with the actual implementation. If you want to see a co-created solution you can look into the ``solved.ipynb``. The  
