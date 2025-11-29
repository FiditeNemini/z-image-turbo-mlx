# DECOUPLED DMD: CFG AUGMENTATION AS THE SPEAR, DISTRIBUTION MATCHING AS THE SHIELD

**Authors:** Dongyang Liu¹'², Peng Gao¹, David Liu¹'², Ruoyi Du¹, Zhen Li¹, Qilong Wu¹, Xin Jin¹, Sihan Cao¹, Shifeng Zhang¹, Hongsheng Li²'⚭, Steven HOI¹

¹Tongyi Lab, Alibaba Group  
²The Chinese University of Hong Kong

⚭jingpeng.gp@alibaba-inc.com, ⚭hsli@ee.cuhk.edu.hk

GitHub: https://github.com/Tongyi-MAI/Z-Image

## ABSTRACT

Diffusion model distillation has emerged as a powerful technique for creating efficient few-step and single-step generators. Among these, Distribution Matching Distillation (DMD) and its variants stand out for their impressive performance, which is widely attributed to their core mechanism of matching the student's output distribution to that of a pre-trained teacher model. In this work, we challenge this conventional understanding. Through a rigorous decomposition of the DMD training objective, we reveal that in complex tasks like text-to-image generation, where CFG is typically required for desirable few-step performance, the primary driver of few-step distillation is not distribution matching, but a previously overlooked component we identify as **CFG Augmentation (CA)**. We demonstrate that this term acts as the core "engine" of distillation, while the **Distribution Matching (DM)** term functions as a "regularizer" that ensures training stability and mitigates artifacts.

We further validate this decoupling by demonstrating that while the DM term is a highly effective regularizer, it is not unique; simpler non-parametric constraints or GAN-based objectives can serve the same stabilizing function, albeit with different trade-offs. This decoupling of labor motivates a more principled analysis of the properties of both terms, leading to a more systematic and in-depth understanding. This new understanding further enables us to propose principled modifications to the distillation process, such as decoupling the noise schedules for the engine and the regularizer, leading to further performance gains. Notably, our method has been adopted by the Z-Image project to develop a top-tier 8-step image generation model, empirically validating the generalization and robustness of our findings.

## 1 INTRODUCTION

Diffusion models have rapidly risen to prominence in generative modeling, achieving state-of-the-art performance and producing images of unprecedented quality and diversity. This success has sparked widespread interest across both academia and industry. However, the power of these models comes at a significant cost: their iterative sampling process, often requiring dozens to hundreds of neural network evaluations, is computationally expensive and slow, hindering their use in real-time applications.

To address this limitation, a flurry of research has focused on converting the original diffusion model into a few-step generator. Typical technical routes include:
- **Direct distillation**: where the student is expected to replicate the trajectory of the teacher
- **Consistency distillation**: which enforces self-consistency along the sampling trajectory
- **Adversarial distillation**: which leverages an adversarial objective to match the student's output distribution to target, showing impressive results in high-resolution synthesis

Among these diverse approaches, score-based distillation, notably represented by Diff-Instruct, Distribution Matching Distillation (DMD), and other variants, has been recognized as exceptionally promising. Its advantages are twofold: not only does it achieve state-of-the-art performance, but it is also underpinned by an elegant theoretical framework. Specifically, the method is framed as minimizing an Integral Kullback-Leibler (IKL) divergence between the student's output distribution (p_fake) and the teacher's target distribution (p_real):

$$L_{IKL}(p_{real}, p_{fake}) = \int_0^1 KL(p_{real,\tau} || p_{fake,\tau}) d\tau \quad (1)$$

In practice, the gradient of this objective is estimated using a pair of "real" (teacher) and "fake" (student-tracking) models, making the training process feasible.

However, a dark cloud has loomed over the interpretation of this elegant method: the use of Classifier-Free Guidance (CFG) on the real model. According to the theoretical derivation, the ideal estimator for the target real score should be the prediction from the real model itself, with no involvement of CFG. However, empirical evidence overwhelmingly shows that on complex tasks like text-to-image, DMD-like methods yield good results only with a high CFG scale. Even if we were to boldly assume that CFG somehow produces a higher-quality substitute for the original score, the asymmetric application of CFG, in which only the real model but not the fake model is equipped with CFG, still creates a stark inconsistency between theory and practice. Overall, the usage of CFG breaks the integrity of the original, rigorous theoretical derivation of matching two distributions.

This strongly suggests that the current understanding of DMD's success is likely incomplete or inaccurate. In this paper, we aim to redefine the understanding of how DMD and similar algorithms work. Through a rigorous decomposition of the practical DMD training objective that utilizes real score CFG, we reveal that its effectiveness is not driven by a single mechanism, but by a clear division of labor between two distinct components:

1. **CFG Augmentation (CA)**: A previously overlooked term that directly applies the CFG signal to the student's output. We demonstrate that this component acts as the core engine of distillation, responsible for converting a multi-step model into a high-quality few-step generator.

2. **Distribution Matching (DM)**: A mechanism that perfectly aligns with the theoretical derivation (Eq. 1). While existing works have proved its independent distillation capability in simple tasks like low-resolution CIFAR, we show that for complex tasks, its primary function converts more to a powerful regularizer that ensures training stability and mitigates artifacts.

This decoupled framework challenges the prevailing narrative and provides a more accurate explanation for the success of DMD-like methods. We substantiate our claims through a series of carefully designed experiments, including independent investigations of the effect of each component, and demonstrating that the DM regularizer, while highly effective, can be conceptually replaced by simpler statistical constraints or more complex GANs. This explicit decoupling also enables a more principled and in-depth analysis of the properties and inner workings of each component. Finally, armed with this deeper understanding, we propose principled improvement by proposing decoupled renoising schedules for CA and DM, respectively, leading to further performance gains, and demonstrating the practical value of our new perspective.

## 2 RELATED WORK

### Few-Step Diffusion Distillation
Aims to reduce the inference cost of diffusion models. Trajectory-matching approaches train a student model to replicate the teacher's sampling path in fewer steps, with consistency distillation as a renowned branch. Another prominent direction is GAN-based distillation, which leverages an adversarial objective to match the student's output distribution with the teacher's or with real data.

### Score-based Distillation
Was initially proposed for 3D generation. Diff-Instruct pioneered its application in few-step diffusion distillation, and DMD was among the first to successfully apply this principle to large-scale text-to-image models. Following works have explored different distribution metrics or combining this principle with other distillation paradigms. Notably, the adoption of CFG in real score is a common practice among these works, but this choice is rarely officially discussed. An exception is Luo (2024), which models the CFG term as an extra reward function after distillation. We are the first to decouple the role of this CFG term during distillation and to reveal its dominance in multi-to-few-step conversion.

## 3 REVISITING AND DECOMPOSING DMD

The goal of Distribution Matching Distillation (DMD) is to train a student generator, denoted as $G_θ$, to emulate the output distribution of a pre-trained, frozen teacher diffusion model in a few-step or even single-step inference process. The training is guided by minimizing a loss function, Eq. 1, whose gradient with respect to the generator's parameters θ can be estimated by:

$$\nabla_θ L_{DMD-theory} = \mathbb{E}_{z_t,\tau,x_τ}\left[-\left(s^{cond}_{real}(x_τ) - s^{cond}_{fake}(x_τ)\right)\frac{\partial G_θ(z_t)}{\partial θ}\right] \quad (2)$$

In this paper, we follow the flow matching notations and define t = 0 with pure noise and t = 1 with clean data. $z_t$ denotes the prepared generator input at noise level t. For single-step generation, t is 0 and $z_t$ is random noise; for few-step generation, $z_t$ can be obtained by going through the previous steps, a technique called "backward simulation". The generator $G_θ$ takes $z_t$ and makes the image prediction $G_θ(z_t)$, which is then renoised to $x_τ$ with a sampled noise level τ. After renoising, $x_τ$ would be fed to two diffusion models for score estimates: $s^{cond}_{real}$, the "real score" estimated by the original multi-step teacher model; and $s^{cond}_{fake}$, the "fake" score estimate from an auxiliary "fake" model that is trained concurrently on the generator's outputs. The subscript "cond" indicates the score is conditioned on a text input.

However, Eq. 2 usually leads to poor performance in practice, and a subtle modification is involved:

$$\nabla_θ L_{DMD} = \mathbb{E}_{z_t,\tau,x_τ}\left[-\left(s^{cfg}_{real}(x_τ) - s^{cond}_{fake}(x_τ)\right)\frac{\partial G_θ(z_t)}{\partial θ}\right] \quad (3)$$

The only difference between Eq. 2 and Eq. 3 is that the real score $s^{cond}_{real}$ is replaced with $s^{cfg}_{real}$, where

$$s^{cfg}_{real}(x_τ) = s^{uncond}_{real}(x_τ) + α\left(s^{cond}_{real}(x_τ) - s^{uncond}_{real}(x_τ)\right) \quad (4)$$

$s^{cond}_{real}$ and $s^{uncond}_{real}$ are the conditional and unconditional scores from the real model, respectively, and α is the CFG guidance scale (typically α > 1). Despite the introduction of discrepancy between theory and practice, this modification empirically yields substantially better results. Interestingly, this substitution has been largely overlooked in prior literature, often dismissed as a mere implementation detail rather than a fundamental deviation from the original theory. However, we will show that **the seemingly minor CFG detail actually signifies a fundamentally different mechanism independent to distribution matching**.

For clarity, in the rest of the paper, we use the acronym "DMD" to specially refer to in-practice-used and CFG-involved algorithm as defined in Eq. 3. In contrast, we use the term "Distribution Matching" or its abbreviation "DM" to refer exclusively to the theoretical principle of matching two distributions, which strictly adheres to the formulation in Eq. 1 and Eq. 2. We will show that the success of DMD is from the cooperation of two different mechanisms, with Distribution Matching playing a crucial, yet secondary, role as a regularizer rather than the primary distillation engine.

### 3.1 DECOMPOSING THE DMD GRADIENT

To scrutinize the underlying mechanisms of the DMD algorithm, we begin by substituting the definition of Classifier-Free Guidance (Eq. 4) into the DMD gradient formula (Eq. 3):

$$\nabla_θ L_{DMD} = \mathbb{E}\left[-\left[s^{uncond}_{real}(x_τ) + α\left(s^{cond}_{real}(x_τ) - s^{uncond}_{real}(x_τ)\right) - s^{cond}_{fake}(x_τ)\right]\frac{\partial G_θ(z_t)}{\partial θ}\right] \quad (5)$$

With simple rearrangement, we can decompose Eq. 5 into two distinct components:

$$\nabla_θ L_{DMD} = \mathbb{E}\left[-\left[\underbrace{\left(s^{cond}_{real}(x_τ) - s^{cond}_{fake}(x_τ)\right)}_{\Delta_{real-fake} \text{ (Distribution Matching)}} + \underbrace{(α-1)\left(s^{cond}_{real}(x_τ) - s^{uncond}_{real}(x_τ)\right)}_{\Delta^{cfg}_{real} \text{ (CFG Augmentation)}}\right]\frac{\partial G_θ(z_t)}{\partial θ}\right] \quad (6)$$

This decomposition reframes the DMD objective as a sum of two terms:

1. **Distribution Matching (DM, $\Delta_{real-fake}$)**: The first term, $s^{cond}_{real} - s^{cond}_{fake}$, strictly aligns with theoretical deduction of matching two distributions (Eq. 1 and 2).

2. **CFG Augmentation (CA, $\Delta^{cfg}_{real}$)**: The second term, $(α-1)(s^{cond}_{real} - s^{uncond}_{real})$, directly applies a scaled CFG signal as a gradient to the student's output. This component was typically overlooked.

This separation motivates an ablation study to isolate the true contribution of each component. We investigate three training configurations: (1) the full DMD objective ($\Delta_{real-fake} + \Delta^{cfg}_{real}$), (2) CFG Augmentation only ($\Delta^{cfg}_{real}$), and (3) Distribution Matching only ($\Delta_{real-fake}$).

#### 3.1.1 ABLATION STUDY: ENGINE VS. REGULARIZER

As illustrated in Fig. 2, our experiments reveal a clear division of labor between the two components. Training with CA alone is remarkably effective at converting the multi-step model into a few-step generator. Besides, the generated results also demonstrate high similarity in content to the full DMD objective, indicating the dominant role of the CA term in DMD loss. In contrast, even though it is improper to conclude that the DM term is totally incapable of doing the multi-step to few-step conversion (since in the 4-step experiment it indeed makes relatively reasonable images), a significant performance gap exists towards the CA setting, as indicated by both image visualizations and numerical indicators (Image Reward and HPS v2.1).

However, we also observe that training with CA alone is unsustainable. While initially effective, the generated images progressively suffer from artifacts such as over-saturation and high-frequency noise, eventually leading to training collapse. The introduction of the Distribution Matching term eliminates these issues, enabling stable training over extended periods and yielding higher-quality final results. These empirical findings lead to two fundamental conclusions:

1. **CFG Augmentation is the engine for few-step conversion.** The ability of the distilled generator to produce high-quality samples in a few steps is almost entirely attributable to the $\Delta_{cfg}$ term.

2. **Distribution Matching is a regularizer for training stability.** The $\Delta_{real-fake}$ term, while not the primary driver of distillation, plays a crucial role as a regularizer that prevents the training process from diverging and ensures the quality of the final output.

This insight fundamentally challenges the prevailing understanding of DMD-like methods: the conversion to a few-step generator is not primarily an act of matching distributions but rather a direct consequence of "baking" the CFG pattern into the student generator's predictions (we elaborate on this point in Sec. A), which is irrelevant to the fake model.

### 3.2 DISTRIBUTION MATCHING: A GOOD, BUT NOT THE ONLY, REGULARIZER

To further validate the aforementioned division between engine and regularizer, we investigate whether alternative regularization schemes can effectively stabilize the CFG Augmentation (CA) engine. This section demonstrates that even a simple non-parametric statistical constraint can prevent training collapse, thereby substantiating the role of Distribution Matching (DM) as a regularizer. Concurrently, we highlight that DM is an exceptionally well-suited regularizer for this task, exhibiting clear advantages over both simpler non-parametric methods and more complex GAN-based approaches.

#### Non-Parametric Mean-Variance Regularization
As shown in Fig. 3, training with the CA engine leads to a monotonic increase in the variance of generated images, finally reaching unreasonably large values. This inspires us to design the simplest regularization term of constraining the mean and variance of the generator's output. Specifically, we apply a Kullback-Leibler (KL) divergence loss that aligns the per-image mean μ and variance σ² of the student's output $G_θ(z_t)$ with target statistics ($μ_{target}$, $σ²_{target}$). For SDXL experiments, we use $μ_{target} = 0.075$ and $σ²_{target} = 0.81$, which are the averaged statistics of sampled real data. For a batch of B generated images, the loss is defined as:

$$L_{KL} = \frac{1}{B}\sum_{i=1}^{B}\frac{1}{2}\left(\frac{σ²_i + (μ_i - μ_{target})²}{σ²_{target}} - 1 - \log\frac{σ²_i}{σ²_{target}}\right) \quad (7)$$

where $(μ_i, σ²_i)$ are the mean and variance of the i-th generated image.

As shown in Fig. 3 and Fig. 6, this simplest non-parametric regularization proves remarkably effective at stabilizing the training process, allowing the CA engine to operate durably, keeping the quality indicators at a relatively high level. This result strongly reinforces the hypothesis that the primary role of the DM term is a regularizer. However, the final image quality, while stable, falls noticeably short of that achieved with DM. This suggests that the artifacts induced by the CA engine are more complex than what can be captured by mean&variance alone.

#### GAN-based Regularization
A more powerful candidate for regularization is a GAN discriminator, a technique already employed in several diffusion distillation works. Following existing methods, we use a discriminator initialized from the weights of the pre-trained teacher model. Experiments (Fig. 3) confirm that a GAN can indeed function as a regularizer, effectively controlling image variance and eliminating certain artifacts. However, GAN requires image data during the distillation process. Besides, the approach introduces significant challenges in training stability, as the training still collapsed after 4k iterations.

Our investigation suggests that while distribution matching is not the only possible regularizer, it represents a sweet spot—offering a more powerful corrective signal than simple statistical constraints, while being substantially more stable and less complex than GANs. The comparison suggests a trade-off between stability and potential performance. The increasing complexity from statistical constraints to GANs correlates with decreasing training robustness, which echoes the claim in Diff-Instruct that score-matching can be viewed as a more stable alternative to GANs, especially when the distributions have disjoint supports. Conversely, greater complexity may offer a higher performance ceiling. This aligns with practices in VAE training or advanced few-step distillation, where models are often first trained with a stable objective before being fine-tuned with a GAN loss to achieve peak performance.

## 4 MECHANISTIC ANALYSIS OF CA AND DM

The decomposition of the DMD objective into a CA engine and a DM regularizer enables an in-depth exploration of their respective inner workings. This section aims to answer two fundamental questions: first, how exactly does the CA engine drive the multi-step to few-step conversion? And second, by what mechanism does the DM regularizer ensure the stability of this process?

### 4.1 DISSECTING THE CA ENGINE: THE ROLE OF THE RE-NOISING SCHEDULE

To understand the working mechanism of the CA engine, we investigate a central question: How does the choice of re-noising timestep τ influence the effect of CA? Since τ determines the noise level at which the CFG signal is computed, it serves as a powerful probe into the engine's behavior.

To isolate this effect, we design an experiment where a single-step generator is trained using only the CA term ($\Delta^{cfg}_{real}$). We then systematically vary the sampling range of the re-noising timestep τ, starting from the noisiest end of the spectrum and gradually expanding towards cleaner timesteps.

Results in Fig. 4 (a) reveal a clear and consistent pattern. When τ is restricted to a noisy range (e.g., [0.0, 0.05]), the CA engine primarily enhances low-frequency information, such as broad color blocks and overall composition. As the range of τ expands to include cleaner timesteps, the generated images progressively gain richer, higher-frequency details, like sharp edges and fine textures. This leads to a crucial conclusion: **the CA engine, when applied at a specific noise level τ, primarily enhances the image content corresponding to that level**. This conclusion is further supported by the fact that the training collapses when τ is restricted to clean timesteps only ([0.7, 1.0]): high-frequency details are meaningless if low-frequency general structure has not yet been determined.

This finding has a critical implication for multi-step generation. Consider a generator at step t operating on an input $z_t$, which already contains resolved information for noise levels below t. Is it still necessary, or even detrimental, to apply CA with a re-noising range containing τ < t? Our analysis suggests this would be redundant, potentially over-enhancing already established features and leading to artifacts. We therefore hypothesize that an optimal CA schedule should act as a **focused engine, concentrating its power on the remaining, unresolved aspects of the image by constraining its re-noising schedule to τ > t**. Experimental validation is provided in Sec. 4.3.

### 4.2 UNDERSTANDING THE DM REGULARIZER: A CORRECTIVE MECHANISM

Having established CA as the engine, we now turn to the DM regularizer and ask: How does it counteract the artifacts introduced by CA and ensure training stability?

To gain insight into this corrective mechanism, we design a specific diagnostic experiment. We continue to train the generator using only the CA engine, a setting we know is unstable and produces artifacts. However, we introduce an auxiliary "fake" model as a non-interfering observer. This fake model is trained concurrently on the generator's outputs—just as in standard DMD—but its score estimate is not used to update the generator. This setup allows us to witness when artifacts occur, how a potential DM gradient ($s^{cond}_{real} - s^{cond}_{fake}$) would act to correct them.

Fig. 4 (b) offers an informative observation. The image generated by the CA-only generator exhibits clear high-frequency checkerboard artifacts. When this artifact-laden image is re-noised and fed to the two score models, the artifact is conspicuously absent in the prediction from the frozen real model, yet it persists in the prediction from the observer fake model. This occurs because the fake model, by tracking the generator's output distribution, has learned to replicate its characteristic failure modes. Consequently, in the DM gradient, the artifacts present in the fake prediction ($s^{cond}_{fake}$) form a negative term. Applying this gradient to the generator's output would thus encourage a change that actively cancels out these artifacts. This provides a concrete illustration of DM's corrective mechanism.

This understanding also clarifies the role of the re-noising timestep τ within the DM regularizer: it controls the scope of correction. A large τ (cleaner image) allows the real and fake scores to diverge primarily on subtle, high-frequency details. In contrast, a small τ (noisier image) destroys most details, forcing the scores to diverge on more fundamental, low-frequency elements like composition and color. This gives the DM regularizer an opportunity to correct global issues.

This leads to our final hypothesis regarding the optimal renoising schedule for DM in a few-step setting. Even late-stage generation steps, which primarily add high-frequency details, can still suffer from low-frequency artifacts like color oversaturation, either inherited from previous steps or induced by an imperfect CA schedule. To address these global issues, the DM regularizer requires a global perspective. Therefore, we propose that the optimal DM schedule is different from that of CA: **DM should function as a comprehensive regularizer, always spanning the full noise range ($τ_{DM} ∈ [0, 1]$), irrespective of the generator's current timestep t**.

### 4.3 VALIDATING THE DECOUPLED SCHEDULE HYPOTHESIS

Our mechanistic analysis in Sec. 4.1 and Sec. 4.2 has led to a central hypothesis: for optimal few-step distillation, the CA engine and the DM regularizer require distinct, decoupled re-noising schedules. Specifically, we proposed that the CA schedule should be constrained ($τ_{CA} > t$) to act as a focused engine, while the DM schedule should remain global ($τ_{DM} ∈ [0, 1]$) to serve as a comprehensive regularizer. In this section, we empirically validate this proposal.

To facilitate this investigation, we first generalize the DMD gradient (Eq.6) to a τ-decoupled form. This allows us to assign independent re-noising schedules, $τ_{CA}$ and $τ_{DM}$, to the CA and DM components, respectively. The resulting "decoupled DMD" (d-DMD) gradient is formulated as:

$$\nabla_θ L_{d-DMD} = \mathbb{E}\left[-\left[\left(s^{cond}_{real}(x_{τ_{DM}}) - s^{cond}_{fake}(x_{τ_{DM}})\right) + (α-1)\left(s^{cond}_{real}(x_{τ_{CA}}) - s^{uncond}_{real}(x_{τ_{CA}})\right)\right]\frac{\partial G_θ(z_t)}{\partial θ}\right] \quad (8)$$

This modification allows us to decouple the renoising schedule of DM and CA, allowing principled experimental analysis. With this formulation, we design an ablation study to evaluate four distinct schedule configurations for a 4-step generator:

- ➀ **Coupled-Shared**: The original DMD approach where $τ_{CA} = τ_{DM}$, sampled from [0, 1].
- ➁ **Decoupled-Full**: Both schedules are independent but cover the full range, $τ_{CA}, τ_{DM} ∈ [0, 1]$.
- ➂ **Decoupled-Constrained**: Both schedules are independent and constrained, $τ_{CA}, τ_{DM} > t$.
- ➃ **Decoupled-Hybrid**: The engine is constrained while the regularizer is not, $τ_{CA} > t, τ_{DM} ∈ [0, 1]$.

The results, presented in Tab. 1 for the Lumina-Image-2.0 model, provide strong evidence for our hypothesis. First, we confirm that merely decoupling the schedules while keeping them global ➁ yields negligible impact compared to the baseline ➀, demonstrating that the benefit comes from the schedule's range, not just its independence. More importantly, both configurations with constrained schedules (➂ and ➃) significantly outperform the baselines across multiple benchmarks. Crucially, our proposed Decoupled-Hybrid setting ➃ consistently achieves the best overall scores, validating our core proposal.

The qualitative results in Fig. 5 offer further visual confirmation. Compared to the global schedule (➁, top row), constraining the CA engine (➂, middle row) introduces richer, finer details, confirming the benefit of a focused engine. However, this configuration still suffers from color oversaturation, a low-frequency artifact that its constrained DM regularizer fails to correct. In stark contrast, our Decoupled-Hybrid setting (➃, bottom row) retains these enhanced details while effectively mitigating the saturation artifacts, yielding the most visually appealing and natural-looking images. These observations are decisively corroborated by a comprehensive user study (Sec. C), where model ➃ achieved a unanimous 100% preference rate in model-level comparisons. 15 annotators consistently justified their choice by its ability to generate richer details, a more realistic and less "greasy" appearance, and fewer structural deformities. Furthermore, in a three-way image-level ranking, model ➃ was ranked first in 59.8% of cases, significantly outperforming the next-best model (➂ at 33.8%).

Experiments on SDXL (Tab. 2) situate our findings within the broader landscape. For a rigorous comparison with DMD2, we adopted their exact training configuration, including the GAN loss, and only replaced their re-noising schedule with our Decoupled-Hybrid approach. The results demonstrate a clear advantage, confirming the effectiveness of our proposed schedule.

In summary, these results strongly validate that our engine-regularizer decomposition not only provides a deeper understanding but also unlocks tangible performance improvements, demonstrating the practical value of our new perspective.

## 5 CONCLUSION AND LIMITATIONS

In this work, we challenge the conventional understanding of DMD practice in complex tasks like text-to-image, revealing a functional decoupling with CFG Augmentation (CA) as the primary engine for few-step conversion and Distribution Matching (DM) as the regularizer. This new perspective allowed us to observe the distinct properties of each component and propose a principled improvement–a decoupled re-noising schedule.

However, a fundamental question remains unanswered: why does CA possess such a remarkable ability to convert a diffusion model into a few-step generator? We find that providing a precise answer is highly challenging, partly because the mechanism of CFG itself remains largely enigmatic. For interested readers, we share our high-level, preliminary understanding and explanation of this issue in Sec. A. Nevertheless, we acknowledge that a significant gap remains towards a rigorously accurate explanation, and we intend to explore this topic further in our future work.

---

## APPENDIX A: DISCUSSION: WHY DOES CA WORK?

In this section, we attempt to build a conceptual bridge to understand the efficacy of CFG Augmentation (CA) as the "engine" of few-step distillation. We do so by drawing a parallel between diffusion models and Large Language Models (LLMs) under a unified view of sequential generation.

### A.1 A PARALLEL PROBLEM: WHY CAN'T LLMS PERFORM N-TOKEN PREDICTION?

We begin by considering a parallel question in the domain of LLMs: why must they generate text token by token, rather than predicting the next N tokens simultaneously? Consider the prompt, "The richest person in the world is". Both "Elon Musk" and "Bill Gates" are plausible completions. A model cannot simultaneously predict the first and second tokens, because the choice of the second token (e.g., "Musk" or "Gates") is strictly conditional on the choice of the first ("Elon" or "Bill"). An attempt to predict both at once would risk generating incoherent combinations like "Elon Gates" or "Bill Musk", or more likely, an averaged and meaningless representation.

The fundamental reason for this limitation is that the model's role is to predict a probability distribution for the next token, P(token₁|prompt). It cannot, however, control the outcome of the probabilistic sampling process that selects a single token from this distribution. This external, uncontrollable sampling event breaks the model's predictive flow. No matter how powerful the model, it cannot bypass this external intervention to predict token₂, because any prediction it makes could conflict with the yet-undetermined outcome of token₁.

### A.2 A GENERAL FRAMEWORK FOR SEQUENTIAL GENERATION

We can abstract this to a higher level. Any sequential generator, at each step, faces three types of information:

- **Type 1 (Determined)**: Information that is already fixed and known (e.g., the input prompt).
- **Type 2 (Directly Determinable)**: Information for which the model can predict a distribution of possibilities for the very next step.
- **Type 3 (Directly Undeterminable)**: Information that can only be determined after some Type 2 information is resolved and becomes Type 1.

The process of sequential generation is the iterative conversion of information from Type 2 to Type 1, which in turn allows Type 3 to become Type 2. Importantly, we claim that **the existence of Type 3 information, whose resolution is contingent on an uncontrollable external decision on Type 2 information, is the core reason for iterative generation**.

### A.3 FROM UNCONTROLLABLE INTERVENTION TO A DETERMINISTIC PATTERN

This analysis also points to a potential strategy for converting a sequential generator into a one-step generator: if the random, external decision-making process could be replaced with a deterministic decision pattern, and this pattern could be "baked into" the generator itself. For Type 2 information, what was previously a random variable with non-zero entropy would become a determinable value. This would collapse the entire decision tree into a single, predictable path, allowing all information to be resolved in one go.

### A.4 CONNECTING BACK TO DIFFUSION MODELS AND CFG AUGMENTATION

We posit that diffusion models are also sequential generators, which first establish low-frequency global composition (e.g., the object is a cat, not a dog) before adding high-frequency details (e.g., the texture of the fur). The relationship between composition and detail mirrors that of "Elon/Bill" and "Musk/Gates". We note that this viewpoint has been formally established by Dieleman (2024).

Crucially, we argue that **Classifier-Free Guidance (CFG) acts as an external intervention analogous to probabilistic sampling**. While CFG is a deterministic bias, not a stochastic process, it is equally unpredictable from the model's perspective: The model is trained without awareness of CFG, and at inference, it cannot control the negative prompt or guidance scale (α) that will be applied. Furthermore, CFG, like sampling, transforms the model's prediction from an averaged expectation into a specific, shifted value.

Our central hypothesis is this: **CFG represents a specific, deterministic decision pattern. The CA term in the DMD objective is the mechanism that "bakes" this decision pattern into the student generator's predictions.** By doing so, it transforms the uncontrollable external force of CFG into an internalized, predictable behavior. The generation process, which was a tree of possibilities, collapses into a single, direct path.

Returning to our LLM example, what CFG Augmentation does is akin to telling the model: "Given the current input, the external process will always choose 'Elon' as the first token. Therefore, you can safely assume the first token is 'Elon' and directly predict 'Musk'." This is, we believe, the source of CA's power in enabling few-step image generation.

We acknowledge that the preceding discussion remains at the level of high-level ideas and that our hypothesis—that "CFG represents a specific, deterministic decision pattern"—is a strong assumption. We share this perspective here primarily to stimulate further investigation into this fundamental question and to provide a potential reference point for future work. We also intend to conduct a more in-depth study of this problem in our future research.

---

## APPENDIX B: PSEUDO-CODE

### Algorithm 1: Original & Decoupled DMD Training Procedure

**Require:** Pre-trained teacher model $s_{real}$, CFG scale α, number of steps N, proxy loss weight λ  
**Ensure:** Trained few-step generator $G_θ$

1. Initialize student generator and fake model from the teacher
   - $G_θ ← s_{real}$
   - $s_{fake} ← s_{real}$

2. **while** not converged **do**

3. **Generator Update Step**
   - Sample a generation step t from the few-step schedule {t₁, ..., tₙ}
   - Prepare generator input $z_t$ (e.g., via backward simulation for t > t₁)
   - Generate an image: $x_{gen} ← G_θ(z_t)$

4. **if** 'decoupled_schedule' **then**
   - **Decoupled DMD behavior**
   - Sample CFG augmentation noise level $τ_{CA} ∼ U(t, 1)$
   - Sample Distribution Matching noise level $τ_{DM} ∼ U(0, 1)$
   - Re-noise the generated image for both schedules:
     - $x_{τ_{CA}} ← \text{renoise}(x_{gen}, τ_{CA})$
     - $x_{τ_{DM}} ← \text{renoise}(x_{gen}, τ_{DM})$
   - **with** torch.no_grad():
     - Calculate scores for the Distribution Matching (DM) term
       - $s^{cond,DM}_{real} ← s_{real}(x_{τ_{DM}}, τ_{DM}, \text{text})$
       - $s^{cond,DM}_{fake} ← s_{fake}(x_{τ_{DM}}, τ_{DM}, \text{text})$
     - Calculate scores for the CFG Augmentation (CA) term
       - $s^{cond,CA}_{real} ← s_{real}(x_{τ_{CA}}, τ_{CA}, \text{text})$
       - $s^{uncond,CA}_{real} ← s_{real}(x_{τ_{CA}}, τ_{CA}, '')$
   - Compute the two components of the update direction
     - $\Delta_{DM} ← s^{cond,DM}_{real} - s^{cond,DM}_{fake}$
     - $\Delta_{CA} ← (α - 1)(s^{cond,CA}_{real} - s^{uncond,CA}_{real})$
     - $\Delta_{total} ← \Delta_{DM} + \Delta_{CA}$

5. **else**
   - **Original DMD behavior**
   - Sample a single noise level τ ∼ U(0, 1)
   - Re-noise the generated image: $x_τ ← \text{renoise}(x_{gen}, τ)$
   - **with** torch.no_grad():
     - $s^{cond}_{real} ← s_{real}(x_τ, τ, \text{text})$
     - $s^{uncond}_{real} ← s_{real}(x_τ, τ, '')$
     - $s^{cond}_{fake} ← s_{fake}(x_τ, τ, \text{text})$
     - $s^{cfg}_{real} ← s^{uncond}_{real} + α(s^{cond}_{real} - s^{uncond}_{real})$
   - Compute the combined update direction
     - $\Delta_{total} ← s^{cfg}_{real} - s^{cond}_{fake}$

6. Update generator by minimizing the proxy loss
   - $L_{proxy} ← ||G_θ(z_t) - \text{stop\_grad}(G_θ(z_t) + λ\Delta_{total})||²$
   - Update $G_θ$ by minimizing $L_{proxy}$

7. **Fake Model Update Step**
   - This step can be run multiple times per generator update (TTUR)
   - Sample a new noise level τ' ∼ U(0, 1)
   - Generate a new image with detached gradient: $x'_{gen} ← \text{stop\_grad}(G_θ(z_t))$
   - Re-noise the new image: $x'_{τ'} ← \text{renoise}(x'_{gen}, τ')$
   - $L_{denoise} ← ||s_{fake}(x'_{τ'}, τ') - x'_{gen}||²$
   - Update $s_{fake}$ using $∇L_{denoise}$

8. **end while**

---

## APPENDIX C: USER STUDY

To further validate the effectiveness of our proposed Decoupled-Hybrid schedule (➃), we conducted a comprehensive user study comparing the four few-step models from the ablation in Table 1. The study was divided into two parts: a per-image ranking evaluation and a per-model side-by-side comparison.

### C.1 PER-IMAGE RANKING EVALUATION

In this part, we compared the visual quality of models ➁ (Decoupled-Full), ➂ (Decoupled-Constrained), and ➃ (Decoupled-Hybrid). We randomly selected 500 prompts from the HPSv2 benchmark and generated one image for each prompt using the three models. We then invited 10 professional annotators to perform a forced ranking of the three images in each set based on their overall quality (e.g., a ranking of 2>3>1, with ties disallowed). To prevent positional bias, the order of the three images within each triplet was randomized, so annotators could not determine which model generated which image based on its position.

The quantitative results are presented in Table 3 and Table 4. As shown, our proposed Decoupled-Hybrid schedule (➃) demonstrates a significant and consistent advantage over the other configurations. It achieved a first-place ranking in 59.8% of the evaluations, far surpassing the 33.8% of the next-best model, ➂. The pairwise comparison in Table 4 further confirms this superiority, where model ➃ achieved win rates of 60.6% and 83.4% against models ➂ and ➁, respectively.

**Table 3: Overall performance and rank distribution from the per-image user study**

| Model | Avg. Rank ↓ | 1st Place (%) ↑ | 2nd Place (%) | 3rd Place (%) |
|-------|-------------|-----------------|---------------|---------------|
| ➁ Decoupled-Full | 2.748 | 6.4 | 12.4 | 81.2 |
| ➂ Decoupled-Constrained | 1.692 | 33.8 | 63.2 | 3.0 |
| ➃ Decoupled-Hybrid | 1.560 | 59.8 | 24.4 | 15.8 |

**Table 4: Pairwise win rates (%) from the per-image ranking evaluation**

|  | ➁ Decoupled-Full | ➂ Decoupled-Constrained | ➃ Decoupled-Hybrid |
|---|---|---|---|
| ➁ Decoupled-Full | – | 8.6 | 16.6 |
| ➂ Decoupled-Constrained | 91.4 | – | 39.4 |
| ➃ Decoupled-Hybrid | 83.4 | 60.6 | – |

### C.2 PER-MODEL SIDE-BY-SIDE COMPARISON

We further conducted a per-model comparison to gauge the overall preference between different schedules. In this setup, we performed three separate side-by-side comparisons: ➀ (Coupled-Shared) vs. ➃, ➁ (Decoupled-Full) vs. ➃, and ➂ (Decoupled-Constrained) vs. ➃. For each comparison, we randomly sampled 200 prompts from the HPSv2 benchmark (using a different random seed for each pair) and displayed the generated images in a fixed two-column layout. We then asked 15 annotators to review all 200 image pairs and make a single, model-level judgment on which model (left or right) performed better overall. They were also asked to provide a brief justification for their choice.

**Table 5: Quantitative results from the per-model, side-by-side user study with 15 annotators**

| Comparison | # Prompts | Preference for Model ➃ (%) |
|------------|-----------|---------------------------|
| ➀ (Coupled-Shared) vs. ➃ | 200 | 100% |
| ➁ (Decoupled-Full) vs. ➃ | 200 | 100% |
| ➂ (Decoupled-Constrained) vs. ➃ | 200 | 100% |

The quantitative results, summarized in Table 5, show a decisive victory for our proposed Decoupled-Hybrid model (➃). It was unanimously preferred in all three head-to-head comparisons, achieving a 100% win rate. The justifications provided by the annotators shed light on the reasons for this strong preference. The most frequently cited advantages of our model (➃) were its ability to generate richer details, produce a more realistic/less over-saturated/not greasy texture/coloring, and exhibit fewer anatomical or structural deformities.

---

## APPENDIX D: ADDITIONAL EXPERIMENTAL RESULTS

### D.1 DETAILED RESULTS OF RE-NOISING SCHEDULE ABLATION

In Tab. 1 of the main text, we have already presented the overall performance comparison for different re-noising schedule configurations. In Tab. 6 and Tab. 7, we additionally provide the fine-grained results on the HPS v2.1 and HPS v3 benchmarks, respectively.

**Table 6: Detailed results on the HPS v2.1 benchmark**

| Method | Concept-Art | Photo | Anime | Paintings | Average |
|--------|-------------|-------|-------|-----------|---------|
| Original (50 steps) | 30.35 | 28.24 | 31.74 | 30.47 | 30.20 |
| ➀ τ_CA = τ_DM ∈ [0,1] (DMD) | 30.31 | 29.95 | 31.72 | 30.45 | 30.61 |
| ➁ τ_CA ∈ [0,1], τ_DM ∈ [0,1] | 30.31 | 30.25 | 31.85 | 30.34 | 30.69 |
| ➂ τ_CA > t, τ_DM > t | 31.48 | 30.87 | 32.98 | 31.51 | 31.71 |
| ➃ τ_CA > t, τ_DM ∈ [0,1] | 32.37 | 30.87 | 33.61 | 32.31 | 32.29 |

**Table 7: Detailed results on the HPS v3 benchmark**

| Method | Animals | Architecture | Arts | Characters | Design | Food |
|--------|---------|--------------|------|------------|--------|------|
| Original (50 steps) | 9.562 | 10.418 | 9.292 | 10.822 | 8.358 | 10.160 |
| ➀ τ_CA = τ_DM ∈ [0,1] (DMD) | 10.377 | 11.537 | 9.077 | 11.061 | 8.112 | 11.520 |
| ➁ τ_CA ∈ [0,1], τ_DM ∈ [0,1] | 10.338 | 11.539 | 8.929 | 11.101 | 8.059 | 11.572 |
| ➂ τ_CA > t, τ_DM > t | 11.309 | 12.388 | 9.690 | 11.975 | 8.694 | 12.096 |
| ➃ τ_CA > t, τ_DM ∈ [0,1] | 11.873 | 12.899 | 10.351 | 12.562 | 9.308 | 12.425 |

| Method | Nat. Sce. | Others | Plants | Products | Science | Transportation |
|--------|-----------|--------|--------|----------|---------|----------------|
| Original (50 steps) | 9.781 | 9.623 | 9.881 | 9.409 | 8.477 | 9.659 |
| ➀ τ_CA = τ_DM ∈ [0,1] (DMD) | 10.138 | 10.797 | 10.528 | 10.994 | 8.417 | 11.503 |
| ➁ τ_CA ∈ [0,1], τ_DM ∈ [0,1] | 10.046 | 10.787 | 10.600 | 11.049 | 8.358 | 11.484 |
| ➂ τ_CA > t, τ_DM > t | 11.177 | 11.553 | 11.390 | 11.508 | 8.905 | 12.256 |
| ➃ τ_CA > t, τ_DM ∈ [0,1] | 11.592 | 12.138 | 11.846 | 11.841 | 9.451 | 12.782 |

### D.2 QUALITATIVE COMPARISON OF DIFFERENT REGULARIZERS

In Fig. 3 of the main text, we have already traced the quantitative indicators of combining the CFG Augmentation (CA) with different regularizers. [Figure 6 would show sample visualizations from this experiment across different training steps comparing No Regression, Mean-Var Regression, Distribution Matching, and GAN approaches.]