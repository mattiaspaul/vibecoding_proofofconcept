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

The next prompt should get us kickstarted with the actual implementation. If you want to see a co-created solution you can look into the ``solved.ipynb``. The snippet to read the provided data is ``
ct_slice, seg_slice = torch.load('ct_seg_slice.pth')['ct_slice'], torch.load('ct_seg_slice.pth')['seg_slice']``. 

General recommendations:
- Users should maintain full control and understand every line.
- Break coding into small, human-verifiable sub-steps to avoid agentic "spaghetti" repositories.
- Prefer inline autocomplete (next edit suggestions) over massive copy-pasting.

> **Suggested prompt**
> 
> *"I want to implement a 2D proof of concept for &quot;Coordinate Translator for Learning Deformable Medical Image Registration (im2grid, 2022)&quot; by Yihao Liu (https://github.com/yihao6/vfa/blob/main/vfa/models/im2grid.py), which is a training-free, differentiable module that matches features between fixed and moving images to output coordinate correspondences directly. Our features should be the last spatial tokens of DINOv2; assume the input is a preprocessed/normalised tensor ``ct_slice`` of size 9, 3, 301, 301 that only needs scale_factor=2 upsampling. "*
> 
> — **undefined**, *broad enough to leave some room for interpretation (think about more strict guiderails)*

<details>
<summary><b>ChatGPT</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2CoordinateTranslator(nn.Module):
    """
    Training-free 2D coordinate translator.

    Given fixed and moving images:
        fixed  = [B, 3, 301, 301]
        moving = [B, 3, 301, 301]

    returns a dense correspondence grid:
        grid = [B, 2, 301, 301]

    grid[:, 0] = x-coordinate in moving image
    grid[:, 1] = y-coordinate in moving image

    Coordinates are in pixel units.
    """

    def __init__(
        self,
        dinov2_name="dinov2_vits14",
        temperature=1.0,
        upsample_factor=2,
    ):
        super().__init__()

        self.temperature = temperature
        self.upsample_factor = upsample_factor

        # Pretrained DINOv2 backbone.
        self.dino = torch.hub.load(
            "facebookresearch/dinov2",
            dinov2_name,
        )

        # This is a training-free module.
        self.dino.eval()

        for p in self.dino.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_features(self, x):
        """
        x: [B, 3, H, W]

        returns:
            features: [B, N, D]
            Ht, Wt: spatial token dimensions
        """
        out = self.dino.forward_features(x)

        # Final normalized spatial/patch tokens.
        tokens = out["x_norm_patchtokens"]

        B, N, D = tokens.shape

        # For ViT/14:
        # N = (H / 14) * (W / 14)
        Ht = x.shape[-2] // 14
        Wt = x.shape[-1] // 14

        assert Ht * Wt == N, (
            f"Unexpected token count: N={N}, "
            f"but expected {Ht}*{Wt}={Ht*Wt}"
        )

        return tokens, Ht, Wt

    @staticmethod
    def coordinate_grid(H, W, device, dtype):
        """
        Normalized coordinates in [-1, 1].

        Returns:
            [1, H*W, 2]
        """
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )

        grid = torch.stack([x, y], dim=-1)
        return grid.reshape(1, H * W, 2)

    def forward(self, fixed, moving):
        """
        fixed, moving:
            [B, 3, 301, 301]

        returns:
            correspondence:
                [B, 2, 301, 301]

            confidence:
                [B, 301, 301]
        """

        B, C, H, W = fixed.shape

        assert C == 3
        assert fixed.shape == moving.shape

        # ------------------------------------------------------------
        # 1. 2x upsample
        # ------------------------------------------------------------

        fixed_up = F.interpolate(
            fixed,
            scale_factor=self.upsample_factor,
            mode="bilinear",
            align_corners=True,
        )

        moving_up = F.interpolate(
            moving,
            scale_factor=self.upsample_factor,
            mode="bilinear",
            align_corners=True,
        )

        Hu, Wu = fixed_up.shape[-2:]

        # ------------------------------------------------------------
        # 2. DINOv2 final spatial tokens
        # ------------------------------------------------------------

        fixed_tokens, Ht, Wt = self.extract_features(fixed_up)
        moving_tokens, _, _ = self.extract_features(moving_up)

        # [B, N, D]
        assert fixed_tokens.shape == moving_tokens.shape

        # ------------------------------------------------------------
        # 3. Normalize features
        # ------------------------------------------------------------

        fixed_tokens = F.normalize(
            fixed_tokens,
            dim=-1,
        )

        moving_tokens = F.normalize(
            moving_tokens,
            dim=-1,
        )

        # ------------------------------------------------------------
        # 4. Feature matching
        #
        # For every fixed token i:
        #
        #   p(j | i) = softmax(
        #       <F_i, M_j> / temperature
        #   )
        #
        # This is exactly the attention-style coordinate
        # translator used by im2grid.
        # ------------------------------------------------------------

        attention = torch.matmul(
            fixed_tokens,
            moving_tokens.transpose(-1, -2),
        )

        attention = attention / self.temperature

        attention = torch.softmax(
            attention,
            dim=-1,
        )

        # ------------------------------------------------------------
        # 5. Moving-image coordinate values
        # ------------------------------------------------------------

        moving_grid = self.coordinate_grid(
            Ht,
            Wt,
            device=moving_tokens.device,
            dtype=moving_tokens.dtype,
        )

        moving_grid = moving_grid.expand(B, -1, -1)

        # Expected moving coordinate for every fixed token.
        #
        # [B, N, N] @ [B, N, 2]
        #       -> [B, N, 2]
        #
        token_correspondence = torch.matmul(
            attention,
            moving_grid,
        )

        # ------------------------------------------------------------
        # 6. Convert token correspondences back to a dense image grid
        # ------------------------------------------------------------

        token_correspondence = token_correspondence.view(
            B,
            Ht,
            Wt,
            2,
        )

        # [B, Ht, Wt, 2] -> [B, 2, Ht, Wt]
        token_correspondence = token_correspondence.permute(
            0, 3, 1, 2,
        )

        # Upsample the sparse DINO correspondence field to the
        # 602x602 image.
        correspondence_up = F.interpolate(
            token_correspondence,
            size=(Hu, Wu),
            mode="bilinear",
            align_corners=True,
        )

        # ------------------------------------------------------------
        # 7. Bring it back to original 301x301 coordinates
        # ------------------------------------------------------------

        correspondence = F.interpolate(
            correspondence_up,
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )

        # ------------------------------------------------------------
        # 8. Confidence = maximum matching probability
        # ------------------------------------------------------------

        confidence = attention.max(dim=-1).values

        confidence = confidence.view(
            B,
            Ht,
            Wt,
        )

        confidence = F.interpolate(
            confidence.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

        return correspondence, confidence
```

</details>

<details>
<summary><b>Qwen 3.7 Plus</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoordinateTranslator(nn.Module):
    """Training-free, differentiable attention-based coordinate translator."""
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # query: (B, N, 1, C), key: (B, N, K, C), value: (B, N, K, D)
        attention = torch.matmul(query, key.transpose(-1, -2)) # (B, N, 1, K)
        attention = self.softmax(attention)
        out = torch.matmul(attention, value) # (B, N, 1, D)
        return out

class PositionalEmbedding2D(nn.Module):
    def __init__(self, num_channels, matching_channels):
        super().__init__()
        assert matching_channels >= 4, "matching_channels must be >= 4 for 2D cos/sin encoding"
        self.tensor_proj = nn.Conv2d(num_channels, matching_channels, 1)
        self.tensor_proj.weight.data.zero_()
        self.tensor_proj.bias.data.zero_()
        self.grid_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, grid):
        x = self.tensor_proj(x)
        x[:, :4, ...] += grid * self.grid_scale
        return x

class Im2gridDecoder2D(nn.Module):
    def __init__(self, feature_dim=384, matching_channels=64):
        super().__init__()
        self.positional_embedding = PositionalEmbedding2D(num_channels=feature_dim, matching_channels=matching_channels)
        self.coordinate_translator = CoordinateTranslator()

    def get_positional_embedding(self, tensor, padding=0):
        pads = [padding, padding] if isinstance(padding, int) else padding
        dims = tensor.shape[2:] # H, W
        vectors = [torch.arange(start=0-pad, end=dim+pad, device=tensor.device) for (dim, pad) in zip(dims, pads)]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).unsqueeze(0).type(torch.float32)
        
        max_dim = max(dims)
        grid = 2 * (grid / (max_dim - 1)) - 1.0
        # Map each coordinate dimension to a unit vector (cos, sin)
        grid = torch.cat((torch.cos(grid * math.pi / 2.0), torch.sin(grid * math.pi / 2.0)), dim=1)
        return grid.repeat(tensor.shape[0], 1, 1, 1).detach()

    def tensor_to_patch_token(self, x, kernel=3, stride=1):
        # x: (B, C, H, W) -> patches: (B, C, H_out, W_out, kernel, kernel)
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.flatten(start_dim=4)          # (B, C, H_out, W_out, kernel*kernel)
        patches = patches.flatten(start_dim=2, end_dim=3) # (B, C, H_out*W_out, kernel*kernel)
        token = patches.permute(0, 2, 3, 1)             # (B, H_out*W_out, kernel*kernel, C)
        return token

    def separate_spatial_dim(self, x, dims):
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1) 
            x = x.view(x.shape[0], x.shape[1], dims[0], dims[1])
        return x

    def combine_spatial_dim(self, x):
        if len(x.shape) == 4:
            x = x.flatten(start_dim=2)
            x = x.transpose(-1, -2)
        return x

    def forward(self, F, M):
        B, C, H, W = F.shape
        device = F.device
        
        # 1. Initialize identity grid normalized to [-1, 1]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        identity_grid = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        composed_grids = [identity_grid]
        local_grids = []

        # 2. Sample moving image features with the current composed grid
        M_sampled = F.grid_sample(M, composed_grids[-1].permute(0, 2, 3, 1), mode='bilinear', align_corners=True)
        
        # 3. Prepare moving image tokens (with padding for patch extraction)
        m_padded = F.pad(M_sampled, pad=(1, 1, 1, 1), mode='constant', value=0)
        grid_m = self.get_positional_embedding(M_sampled, padding=1)
        m_embedded = self.positional_embedding(m_padded, grid_m)
        token_m = self.tensor_to_patch_token(m_embedded, kernel=3, stride=1)
        
        # 4. Prepare fixed image tokens
        grid_f = self.get_positional_embedding(F, padding=0)
        f_embedded = self.positional_embedding(F, grid_f)
        token_f = self.combine_spatial_dim(f_embedded).unsqueeze(-2)
        
        # 5. Prepare coordinate grid tokens (value for attention)
        H_pad, W_pad = H + 2, W + 2
        y_coords_pad, x_coords_pad = torch.meshgrid(
            torch.linspace(-1, 1, H_pad, device=device),
            torch.linspace(-1, 1, W_pad, device=device),
            indexing='ij'
        )
        grid_g = torch.stack([x_coords_pad, y_coords_pad], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        token_g = self.tensor_to_patch_token(grid_g, kernel=3, stride=1)
        
        # 6. Coordinate translation (Attention: Query=Fixed, Key=Moving, Value=Grid)
        # token_f: (B, H*W, 1, C), token_m: (B, H*W, 9, C), token_g: (B, H*W, 9, 2)
        token = self.coordinate_translator(token_f, token_m, token_g).squeeze(-2) # (B, H*W, 2)
        tensor_grid = self.separate_spatial_dim(token, dims=(H, W))               # (B, 2, H, W)
        
        # 7. Compose grids
        new_grid = F.grid_sample(composed_grids[-1], tensor_grid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)
        composed_grids.append(new_grid)
        local_grids.append(tensor_grid)
        
        # 8. Apply the requested scale_factor=2 upsampling to the final grid
        upsampled_grid = F.interpolate(composed_grids[-1], scale_factor=2, mode='bilinear', align_corners=True)
        
        # 9. Convert grid from [-1, 1] normalized space to pixel spacing
        dims = torch.tensor(upsampled_grid.shape[2:]).view(1, 2, 1, 1).to(device)
        pixel_grid = (upsampled_grid + 1) / 2 * (dims - 1)
        
        return pixel_grid, composed_grids, local_grids

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super().__init__()
        # Loads pretrained DINOv2 (downloads weights on first run)
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.patch_size = 14
        
    def forward(self, x):
        # x: (B, 3, H, W)
        H, W = x.shape[2:]
        
        # DINOv2 strictly requires dimensions to be multiples of patch_size (14)
        # Interpolate 301x301 -> 308x308 (22 * 14)
        new_H = ((H + 13) // 14) * 14
        new_W = ((W + 13) // 14) * 14
        
        if new_H != H or new_W != W:
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            
        # Get features from the last transformer block
        features = self.model.get_intermediate_layers(x, n=1)[0] # (B, N, D)
        
        # Drop the CLS token (index 0) to retain only spatial tokens
        spatial_tokens = features[:, 1:, :] # (B, H_feat * W_feat, D)
        
        # Reshape to 2D spatial feature map: (B, D, H_feat, W_feat)
        B, N, D = spatial_tokens.shape
        h_feat = new_H // self.patch_size
        w_feat = new_W // self.patch_size
        spatial_tokens = spatial_tokens.permute(0, 2, 1).reshape(B, D, h_feat, w_feat)
        
        return spatial_tokens

class Im2grid2D(nn.Module):
    def __init__(self, feature_dim=384, matching_channels=64):
        super().__init__()
        # feature_dim=384 for ViT-S, 768 for ViT-B, 1024 for ViT-L, 1536 for ViT-g
        self.feature_extractor = DINOv2FeatureExtractor(model_name='dinov2_vits14')
        self.decoder = Im2gridDecoder2D(feature_dim=feature_dim, matching_channels=matching_channels)
        
    def forward(self, fixed_img, moving_img):
        F = self.feature_extractor(fixed_img)
        M = self.feature_extractor(moving_img)
        pixel_grid, composed_grids, local_grids = self.decoder(F, M)
        return pixel_grid, composed_grids, local_grids


# ==========================================
# Proof of Concept Usage
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Assume input is a preprocessed/normalised tensor `ct_slice` of size 9, 3, 301, 301
    batch_size = 9
    ct_slice = torch.randn(batch_size, 3, 301, 301).to(device)
    
    # Dummy fixed image (e.g., an atlas or reference slice) of the same size
    fixed_img = torch.randn(batch_size, 3, 301, 301).to(device)
    
    # 2. Initialize the model (using ViT-Small dimensions as default)
    model = Im2grid2D(feature_dim=384, matching_channels=64).to(device)
    model.eval()
    
    # 3. Forward pass (training-free, differentiable)
    with torch.no_grad():
        pixel_grid, composed_grids, local_grids = model(fixed_img, ct_slice)
        
    print(f"Input ct_slice shape:      {list(ct_slice.shape)}")
    print(f"Output pixel_grid shape:   {list(pixel_grid.shape)} (Upsampled by 2x from feature resolution)")
    print(f"Number of composed grids:  {len(composed_grids)}")
    print(f"Composed grid shapes:      {[list(g.shape) for g in composed_grids]}")
    
    # Optional: Verify differentiability (remove no_grad to test)
    # loss = pixel_grid.sum()
    # loss.backward() # This will successfully backpropagate through the attention and grid sampling
```
End of text
</details>

<details>
<summary><b>Gemini 3.1 Pro</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Im2GridCoordinateTranslator(nn.Module):
    """
    Training-free, differentiable Coordinate Translator (im2grid).
    Calculates a dense correspondence grid by matching features and computing
    a weighted sum over the moving image's spatial coordinates.
    """
    def __init__(self):
        super().__init__()

    def forward(self, feat_fixed, feat_moving):
        """
        Args:
            feat_fixed: Tensor of shape (B, C, H, W)
            feat_moving: Tensor of shape (B, C, H, W)
        Returns:
            deformed_grid: Tensor of shape (B, H, W, 2) in range [-1, 1] 
                           ready for F.grid_sample.
        """
        B, C, H, W = feat_fixed.shape
        N = H * W
        
        # 1. Flatten spatial dimensions for attention matrix operations
        # Shape becomes (B, H*W, C)
        Q = feat_fixed.view(B, C, N).permute(0, 2, 1)  
        K = feat_moving.view(B, C, N).permute(0, 2, 1) 
        
        # Optional: L2 normalization is often beneficial for DINO features 
        # to ensure stable cosine similarities, though the original paper 
        # utilizes standard Scaled Dot-Product scaling.
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        # 2. Compute matching scores and probabilities 
        # Using a fixed temperature for L2-normalized vectors (e.g., 0.05).
        # If omitting L2 norm, use the paper's original scaling: temp = C ** 0.5
        temp = 0.05
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / temp
        attn_probs = F.softmax(attn_logits, dim=-1) # (B, N, N)
        
        # 3. Construct the Identity Grid G_I (the "Values" in the attention mechanism)
        # Coordinates range from [-1, 1] for PyTorch's F.grid_sample compatibility
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=feat_fixed.dtype, device=feat_fixed.device),
            torch.linspace(-1, 1, W, dtype=feat_fixed.dtype, device=feat_fixed.device),
            indexing='ij'
        )
        # Stack into (H, W, 2) representing (x, y) coordinates
        grid_identity = torch.stack([gx, gy], dim=-1) 
        
        # Flatten identity grid: (H*W, 2) and expand to batch size -> (B, N, 2)
        V = grid_identity.view(1, N, 2).expand(B, -1, -1)
        
        # 4. Generate the expected correspondences 
        # G = Softmax(Q * K^T) * G_I
        G = torch.bmm(attn_probs, V)
        
        # Reshape back to spatial structure: (B, H, W, 2)
        deformed_grid = G.view(B, H, W, 2)
        
        return deformed_grid


class RegistrationPipelinePoC(nn.Module):
    """
    End-to-end wrapper bridging DINOv2 feature extraction with the im2grid module.
    """
    def __init__(self, dinov2_model='dinov2_vits14'):
        super().__init__()
        # Load DINOv2. Vit-S operates with a patch size of 14.
        self.encoder = torch.hub.load('facebookresearch/dinov2', dinov2_model)
        self.encoder.eval()
        self.translator = Im2GridCoordinateTranslator()
        
    def extract_dinov2_spatial_tokens(self, x):
        """
        Upsamples the input and extracts 2D spatial features from DINOv2.
        """
        # Upsample 301x301 -> 602x602
        x_up = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.encoder.forward_features(x_up)
            
        # Extract patch tokens: shape (B, N, C)
        patch_tokens = features['x_norm_patchtokens'] 
        B, N, C = patch_tokens.shape
        
        # 602 / 14 = 43. The resulting grid is 43x43
        H_feat = x_up.shape[2] // 14
        W_feat = x_up.shape[3] // 14
        
        # Reshape tokens back into a spatial 2D grid: (B, C, H, W)
        spatial_features = patch_tokens.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
        
        return spatial_features

    def forward(self, fixed_ct, moving_ct):
        # 1. Extract feature grids
        feat_fixed = self.extract_dinov2_spatial_tokens(fixed_ct)
        feat_moving = self.extract_dinov2_spatial_tokens(moving_ct)
        
        # 2. Output coordinate mapping using im2grid
        deformed_grid = self.translator(feat_fixed, feat_moving)
        
        return deformed_grid

# ==========================================
# Example Usage with provided specifications
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulating the provided preprocessed tensor (9 slices, 3 channels, 301x301)
    ct_slice = torch.randn(9, 3, 301, 301).to(device)
    
    # For a registration PoC, split the batch into a fixed target and moving sources
    # (e.g., matching the 8 moving slices to the 1st fixed slice)
    fixed_tensor = ct_slice[0:1].expand(8, -1, -1, -1)  # Shape: (8, 3, 301, 301)
    moving_tensor = ct_slice[1:9]                       # Shape: (8, 3, 301, 301)
    
    pipeline = RegistrationPipelinePoC().to(device)
    
    # Predict the displacement field (deformed grid)
    # Output shape will be (8, 43, 43, 2)
    predicted_grid = pipeline(fixed_tensor, moving_tensor)
    
    print(f"Input moving tensor:  {moving_tensor.shape}")
    print(f"Predicted coordinate grid: {predicted_grid.shape}")
```

</details>

You'll notice that Qwen goes off track, does not confirm to the non-trainable part, introduces positional embeddings etc. Both Gemini and ChatGPT solve the task, yet the latter bloats up the code base immensely - making it harder to follow. Looking at the human written solved.ipynb code with only next edit suggestions, we see how much more compact it remains.  

## Verify code with cross-tool prompts 

General recommendations
- Let another LLM check the code for bugs and conceptional shortcomings
- Be aware that without strict guidance this blurs the line between your own ideas and letting the LLM do research for you

> **Naive check and improve prompt**
>
> *Another LLM provided the following code for a proof-of-concept 2D image registration tool, could you analyse potential practical limitations and suggest improvements* 
>

Which will recommend a number of useful improvements, however, also sway to trainable solutions (diffusion regularisation loss), question DINOv2 in general and surprisingly state "But 602 is not divisible by 14.". 

> **Suggested follow-up**
> 
> *now improve the translator with a local attention mask - ie only consider displacements of 7 voxels max. add a B-spline smoothing to reduce high frequency displacements*

and 
> *how to evaluate everything given corresponding multi-label segmentations*

ChatGPT again delivers a huge amount of boiler plate code, wheras Gemini's suggestion remains readable.

<details>
<summary><b>Gemini 3.1 Pro</b></summary>

```python 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Im2GridCoordinateTranslator(nn.Module):
    """
    Training-free, differentiable Coordinate Translator (im2grid)
    enhanced with a local attention mask to restrict maximum displacement.
    """
    def __init__(self, max_displacement=7):
        """
        Args:
            max_displacement: The maximum allowed displacement in feature-grid voxels.
        """
        super().__init__()
        self.max_displacement = max_displacement
        # Cache the mask if H and W remain constant to avoid recomputing it every forward pass
        self._cached_mask = None 
        self._cached_shape = None

    def _get_local_mask(self, H, W, device):
        """
        Generates an (N, N) boolean mask where True indicates the distance 
        between point i and point j is <= max_displacement.
        """
        if self._cached_shape == (H, W) and self._cached_mask is not None:
            return self._cached_mask.to(device)

        # 1. Create grid of integer pixel coordinates (0 to H-1, 0 to W-1)
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        
        # 2. Flatten to shape (N, 2) where N = H * W
        coords = torch.stack([gy, gx], dim=-1).view(-1, 2)
        
        # 3. Compute pairwise distances
        # Using p=float('inf') yields Chebyshev distance, ensuring the displacement 
        # is strictly <= max_displacement on BOTH the X and Y axes independently.
        # (For a circular radius instead of a square window, use p=2.0)
        dist_matrix = torch.cdist(coords, coords, p=float('inf'))
        
        # 4. Create the boolean mask
        mask = dist_matrix <= self.max_displacement
        
        self._cached_shape = (H, W)
        self._cached_mask = mask
        
        return mask

    def forward(self, feat_fixed, feat_moving):
        """
        Args:
            feat_fixed: Tensor of shape (B, C, H, W)
            feat_moving: Tensor of shape (B, C, H, W)
        Returns:
            deformed_grid: Tensor of shape (B, H, W, 2) in range [-1, 1] 
        """
        B, C, H, W = feat_fixed.shape
        N = H * W
        
        # 1. Flatten spatial dimensions
        Q = feat_fixed.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        K = feat_moving.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        # 2. Compute matching scores
        temp = 0.05
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / temp # (B, N, N)
        
        # 3. Apply Local Attention Mask
        mask = self._get_local_mask(H, W, attn_logits.device)
        
        # Broadcast mask to match batch size and fill invalid pairs with -inf
        # The diagonal (self-matching) is always distance 0, so every row 
        # has at least one valid entry, preventing NaN outputs from Softmax.
        mask = mask.unsqueeze(0).expand(B, -1, -1) 
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        
        # 4. Convert to probabilities
        attn_probs = F.softmax(attn_logits, dim=-1) # (B, N, N)
        
        # 5. Construct Identity Grid 
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, dtype=feat_fixed.dtype, device=feat_fixed.device),
            torch.linspace(-1, 1, W, dtype=feat_fixed.dtype, device=feat_fixed.device),
            indexing='ij'
        )
        grid_identity = torch.stack([gx, gy], dim=-1) # (H, W, 2)
        
        V = grid_identity.view(1, N, 2).expand(B, -1, -1) # (B, N, 2)
        
        # 6. Generate the expected correspondences
        G = torch.bmm(attn_probs, V)
        deformed_grid = G.view(B, H, W, 2)
        
        return deformed_grid
```

<\details>

