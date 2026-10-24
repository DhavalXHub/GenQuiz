# GenQuiz Technical & AI/ML Glossary

> **Document Version:** 1.0.0
> **Target Project:** GenQuiz — Adaptive AI Quiz & Assessment Platform
> **Audience:** Computer Science / AI/ML Engineering Students, Reviewers, Viva Examiners

---

### A

#### Adaptive Assessment
- **Layman Definition:** A quiz system that doesn't just give a static grade, but identifies your weak spots and suggests targeted study areas.
- **Technical Definition:** An evaluation methodology that dynamically maps student performance across granular cognitive categories (topics and Bloom's taxonomy levels) and triggers remediation logic when mastery drops below an established numerical threshold.
- **GenQuiz Example:** In `quiz_evaluator.py`, `evaluate_quiz_attempt()` calculates a topic-level score percentage; any topic below 75% is flagged as `needs_review` and assigned customized pedagogical recommendations in the results dashboard.

#### Attention Mechanism
- **Layman Definition:** A technique that allows an AI model to pay attention to the most important words in a sentence when generating a response.
- **Technical Definition:** A mathematical function computing a weighted sum of value vectors ($V$) based on the softmax-normalized dot product between query ($Q$) and key ($K$) vectors: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
- **GenQuiz Example:** The underlying base model `Qwen2.5-1.5B` employs Multi-Head and Grouped Query Attention (GQA). LoRA adapters modify the linear projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of these attention heads.

#### Autoregressive Generation
- **Layman Definition:** Generating text one word at a time, where each new word depends on all the words written before it.
- **Technical Definition:** A probabilistic sequence generation process where the conditional probability distribution of the next token $x_t$ is conditioned on the entire prefix history: $P(X) = \prod_{t=1}^T P(x_t | x_1, \dots, x_{t-1})$.
- **GenQuiz Example:** When `local_model_service.py` executes `model.generate()`, the causal language model predicts the next token in the JSON response sequentially until `<|im_end|>` or `max_new_tokens=512` is reached.

---

### B

#### BitsAndBytes (BnB)
- **Layman Definition:** A library that shrinks giant AI models to run on smaller, affordable graphics cards without losing much intelligence.
- **Technical Definition:** An open-source low-level CUDA library facilitating 8-bit and 4-bit matrix multiplication and quantization algorithms (e.g., LLM.int8(), NF4) with minimal degradation in model perplexity.
- **GenQuiz Example:** Configured in `GenQuiz_FineTuning.ipynb` and `local_model_service.py` via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`.

#### Bloom's Taxonomy
- **Layman Definition:** A framework ranking educational questions from simple memorization to deep critical thinking.
- **Technical Definition:** A hierarchical pedagogical classification system categorizing intellectual behavior into six cognitive levels: *Remember*, *Understand*, *Apply*, *Analyze*, *Evaluate*, and *Create*.
- **GenQuiz Example:** Tagged in `genquiz_dataset.csv` and enforced in `quiz_generator.py` JSON schema (`bloom_level`). `quiz_evaluator.py` aggregates scores across these levels to inform students if they struggle with conceptual understanding versus factual recall.

---

### C

#### ChatML (Chat Markup Language)
- **Layman Definition:** A special format using system, user, and assistant tags so the model knows who is speaking.
- **Technical Definition:** A structured prompt formatting standard developed by OpenAI using special boundary tokens (`<|im_start|>role\ncontent<|im_end|>`) to delineate system prompts, user queries, and assistant responses.
- **GenQuiz Example:** Generated in `prepare_dataset.py` for `train_sft.jsonl`, formatted via `tokenizer.apply_chat_template()` in `local_model_service.py`.

#### Cosine Annealing Learning Rate
- **Layman Definition:** A schedule that gradually lowers the training speed in a smooth wave shape as the model learns.
- **Technical Definition:** A learning rate decay schedule where the learning rate $\eta_t$ decreases following a cosine curve from maximum $\eta_{\text{max}}$ down to $\eta_{\text{min}}$: $\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$.
- **GenQuiz Example:** Specified in `GenQuiz_FineTuning.ipynb` via `SFTConfig(lr_scheduler_type="cosine")`, decaying from $2 \times 10^{-4}$ at step 10 down to $3.08 \times 10^{-7}$ at step 50.

#### Cross-Entropy Loss
- **Layman Definition:** A mathematical score measuring how surprised or wrong the AI was in guessing the correct next word.
- **Technical Definition:** The loss function for multi-class classification measuring divergence between predicted probability distribution $\hat{y}$ and true one-hot distribution $y$: $\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$.
- **GenQuiz Example:** Computed during SFT in `GenQuiz_FineTuning.ipynb`. Training loss dropped from $2.705$ (step 5) to $0.643$ (step 50); evaluation loss stabilized at $0.919$.

#### CUDA (Compute Unified Device Architecture)
- **Layman Definition:** NVIDIA's platform that lets programs run high-speed calculations on NVIDIA graphics cards.
- **Technical Definition:** A parallel computing platform and application programming interface model created by NVIDIA enabling general-purpose computing on NVIDIA GPUs (GPGPU).
- **GenQuiz Example:** Checked in `local_model_service.py` via `torch.cuda.is_available()`. If detected, weights load into `cuda:0` VRAM; otherwise, system falls back safely without crashing.

---

### D

#### Distractor
- **Layman Definition:** The incorrect options in a multiple-choice question designed to test true understanding.
- **Technical Definition:** Plausible but incorrect alternatives included alongside the correct target answer in an assessment item to evaluate discrimination ability and reveal misconceptions.
- **GenQuiz Example:** Stored in `option_a`, `option_b`, `option_c`, `option_d` in `genquiz_dataset.csv`. Validated in `quiz_generator.py` to ensure all 4 options are distinct strings and do not contain boilerplate placeholders.

#### Double Quantization
- **Layman Definition:** Compressing the compression settings themselves to save even more GPU memory.
- **Technical Definition:** The practice of quantizing the quantization constants (scale factors) themselves, reducing memory consumption by roughly 0.37 bits per parameter.
- **GenQuiz Example:** Configured with `bnb_4bit_use_double_quant=True` in `BitsAndBytesConfig` inside `local_model_service.py`.

---

### E

#### Exact Match (EM)
- **Layman Definition:** A strict grading rule where the student's answer must match the answer key character-for-character.
- **Technical Definition:** A deterministic evaluation metric assigning a binary score $\{0, 1\}$ based on string equality after standard text normalization (case folding, whitespace trimming).
- **GenQuiz Example:** Used in `quiz_evaluator.py:evaluate_mcq()` for MCQ and True/False questions. Strips punctuation and checks if `normalized_selected == normalized_correct`.

---

### F

#### Few-Shot In-Context Learning (ICL)
- **Layman Definition:** Giving an AI a few good examples inside the prompt so it knows the format and style you want.
- **Technical Definition:** Conditioning a pre-trained frozen language model on $k$ input-output demonstration pairs within its context window without updating any model weights.
- **GenQuiz Example:** Managed in `dataset.py:get_few_shot_examples()`. GenQuiz pulls 2–3 rows from `genquiz_dataset.csv` and embeds them inside the prompt prompt string sent to Groq or local LLMs.

#### Flash Attention
- **Layman Definition:** A mathematical optimization that makes Transformer attention run faster and use less memory on the GPU.
- **Technical Definition:** An IO-aware exact attention algorithm that tiles the computation between fast GPU SRAM and High Bandwidth Memory (HBM) to avoid materializing the full $N \times N$ attention matrix.
- **GenQuiz Example:** Supported natively in `Qwen2.5` architecture; instantiated with `torch.bfloat16` during training.

---

### G

#### Gradient Accumulation
- **Layman Definition:** Simulating a large batch size on a small GPU by adding up gradients over multiple small steps before updating weights.
- **Technical Definition:** An optimization technique where forward and backward passes are computed for micro-batches, accumulating gradients over $N$ steps before executing `optimizer.step()` and `optimizer.zero_grad()`.
- **GenQuiz Example:** Configured in `GenQuiz_FineTuning.ipynb` as `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`. This yields an effective batch size of $1 \times 4 = 4$, enabling training within 15 GB Google Colab T4 GPU VRAM.

#### Gradient Checkpointing
- **Layman Definition:** Saving memory by not storing intermediate calculation steps and re-calculating them on the fly during backpropagation.
- **Technical Definition:** A memory-saving technique that trades compute for memory by storing activations only at selected checkpoints in the computational graph and recomputing intermediate activations during the backward pass.
- **GenQuiz Example:** Enabled via `gradient_checkpointing=True` in `GenQuiz_FineTuning.ipynb` SFT training configuration.

---

### H

#### Hallucination
- **Layman Definition:** When an AI invents facts or states incorrect information with total confidence.
- **Technical Definition:** The generation of content by an LLM that is nonsensical or unfaithful to the provided source context (extrinsic hallucination) or objectively false according to world knowledge (intrinsic hallucination).
- **GenQuiz Example:** Guarded against in `quiz_generator.py:validate_question_answer_alignment()`. If an uploaded PDF discusses Cybercrime but the model produces an answer about Transformers positional encoding, the alignment validator intercepts and rejects the generation.

---

### J

#### Jaccard Similarity
- **Layman Definition:** A formula that measures how much two sets of words overlap compared to their total combined words.
- **Technical Definition:** A set similarity coefficient defined as the size of the intersection divided by the size of the union of two sets: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$.
- **GenQuiz Example:** Used in `local_model_service.py:generate_quiz_local_batched()` to detect and eliminate duplicate questions between successive inference batches when generating 15+ questions.

---

### L

#### LoRA (Low-Rank Adaptation)
- **Layman Definition:** Freezing the original AI and attaching small, trainable adapter layers to customize it efficiently.
- **Technical Definition:** A parameter-efficient fine-tuning (PEFT) method that decomposes the weight update matrix $\Delta W \in \mathbb{R}^{d \times k}$ into two low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ where $r \ll \min(d, k)$, computing $W' = W_0 + \frac{\alpha}{r}(B \cdot A)$.
- **GenQuiz Example:** GenQuiz configures $r=16$, $\alpha=32$, targeting 7 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). This trains 18,464,768 parameters (only 1.1820% of the 1.56B base model).

---

### N

#### NormalFloat4 (NF4)
- **Layman Definition:** A 4-bit data format custom-built for neural network weights that follow a bell-curve (normal) distribution.
- **Technical Definition:** An information-theoretically optimal quantile quantization data type for zero-mean, unit-variance normally distributed neural network weights, dividing values into 16 bins with equal probability mass.
- **GenQuiz Example:** Specified in `local_model_service.py` via `bnb_4bit_quant_type="nf4"`.

---

### P

#### PEFT (Parameter-Efficient Fine-Tuning)
- **Layman Definition:** Techniques to adapt huge pre-trained models by updating fewer than 2% of their weights.
- **Technical Definition:** A suite of adaptation algorithms (including LoRA, Prefix Tuning, IA3, and Prompt Tuning) that freeze the majority of pre-trained model parameters and train only a tiny subset of auxiliary weights.
- **GenQuiz Example:** Implemented via Hugging Face's `peft` library (`peft==0.21.2` in `requirements.txt`).

#### Pydantic Schema Validation
- **Layman Definition:** An automated code guard that inspects AI-generated text to verify it has every required field before showing it to users.
- **Technical Definition:** Runtime data parsing and validation in Python using type annotations. Throws explicit validation errors if data types, keys, or constraints fail.
- **GenQuiz Example:** Defined in `quiz_generator.py` (`QuizSchema`, `QuestionItemSchema`). Validates that `question`, `options`, `correct_answer`, `explanation`, and `difficulty` are present and valid.

---

### Q

#### QLoRA (Quantized Low-Rank Adaptation)
- **Layman Definition:** Combining 4-bit model compression with LoRA adapter training so you can fine-tune LLMs on a free Google Colab GPU.
- **Technical Definition:** An extension of LoRA that quantizes the frozen base model weights to 4-bit NormalFloat precision while backpropagating gradients through high-precision (bfloat16) low-rank adapter matrices.
- **GenQuiz Example:** Used in `GenQuiz_FineTuning.ipynb` to fine-tune `Qwen2.5-1.5B` on Google Colab T4 GPU in ~127 seconds.

---

### R

#### RAG (Retrieval-Augmented Generation)
- **Layman Definition:** Searching a database of documents for relevant snippets and pasting them into the AI prompt to answer questions.
- **Technical Definition:** An architectural pattern combining an external vector retrieval mechanism (dense embeddings + similarity search) with an LLM generator to ground responses in external reference knowledge.
- **GenQuiz Distinction:** GenQuiz uses **Direct Context-Window Stuffing and In-Memory Keyword Prioritization** (`pdf_processor.py`), *not* a vector database RAG. GenQuiz extracts text directly with PyMuPDF and fits the filtered sections directly into the model's 32k context window.

---

### S

#### Semantic Evaluation
- **Layman Definition:** Grading a short answer based on whether the student understood the concept, even if they used completely different words.
- **Technical Definition:** An automated scoring approach assessing conceptual equivalence, key concept coverage, and semantic alignment using an LLM or cross-encoder rubric, rather than surface-level lexical string matching.
- **GenQuiz Example:** Implemented in `quiz_evaluator.py:evaluate_short_answer()` using a 3-tier rubric ($1.0, 0.5, 0.0$) evaluating correctness, missing concepts, and misconceptions. Falls back to token-overlap scoring if the evaluator LLM is unreachable.

#### Supervised Fine-Tuning (SFT)
- **Layman Definition:** Teaching a model to follow instructions and output specific formats using prompt-and-answer training pairs.
- **Technical Definition:** The process of training a pre-trained base language model on curated pairs of instructions and ideal responses using causal language modeling cross-entropy loss over target tokens.
- **GenQuiz Example:** Conducted via `trl.SFTTrainer` in `GenQuiz_FineTuning.ipynb` using 40 training items from `data/train_sft.jsonl`.

---

### T

#### Temperature
- **Layman Definition:** A dial controlling how creative or strict the AI is when choosing words.
- **Technical Definition:** A hyperparameter scaling the logits $z_i$ prior to softmax normalization: $P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$. Lower values ($T \to 0$) approach argmax greedy decoding; higher values flatten probabilities.
- **GenQuiz Example:** Set to $T=0.4$ in `local_model_service.py` to balance structured JSON syntax determinism with sufficient linguistic variety for natural question phrasing.
