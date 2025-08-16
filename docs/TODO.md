# TODO

List of benchmarks to add (or not to add, that's the question), as well as other things to do.

Currently based on UbiAI's course section on benchmarks, as well as the Qwen 3 and Gemma 3 technical paper lists of benchmarks.

## Benchmarks to Consider

Source: https://github.com/ubiai-incorporated/ubiai_courses/tree/master/Lesson_1_Getting_Started_with_LLMs/3_benchmarking_models

* NarrativeQA: [arXiv 1712.07040](https://arxiv.org/abs/1712.07040) (Dec. 2017)
  * What it tests: Long-document comprehension, narrative reasoning.
  * Structure: Questions based on entire stories or summaries, requiring multi-paragraph reasoning.
  * Why it's valuable: Challenges models to stay coherent across long spans of text, great for real-world documents.
* HotpotQA: [arXiv 1809.09600](https://arxiv.org/abs/1809.09600) (Sep. 2018)
  * What it tests: Multi-hop reasoning and retrieval.
  * Structure: Questions that require reading multiple Wikipedia passages to find an answer.
  * Why it's valuable: Simulates tasks where no single paragraph has the full answer.
* DROP: [arXiv 1903.00161](https://arxiv.org/abs/1903.00161) (Mar. 2019)
  * What it tests: Discrete reasoning over paragraphs (dates, numbers, comparisons).
  * Structure: Contexts followed by questions with precise, often numerical answers.
  * Why it's valuable: Measures both reading comprehension and logical reasoning.

### General Tasks

MMLU variants:

* MMLU: [arXiv 2009.03300](https://arxiv.org/abs/2009.03300) (Sep. 2020)
  * What it tests: a model's general capabilities in a variety of tasks, requiring extensive world knowledge
* MMLU-Pro: [arXiv 2406.01574](https://arxiv.org/abs/2406.01574) (Jun. 2024)
  * Difference from MMLU: "extend the mostly knowledge-driven MMLU benchmark by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options. Additionally, MMLU-Pro eliminates the trivial and noisy questions in MMLU."
* MMLU-Redux [arXiv 2406.04127](https://arxiv.org/abs/2406.04127) (Jun. 2024)
  * Difference from MMLU: addresses numerous errors in ground truth of MMLU (ideally replace MMLU entirely)
* Multilingual variants: MMMLU, Global MMLU-Lite (see Multilingual Tasks)

Other:

* SuperGPQA: [arXiv 2502.14739](https://arxiv.org/abs/2502.14739) (Feb. 2025)
  * What it tests: graduate-level knowledge and reasoning capabilities across 285 disciplines
* BBH (BIG-Bench Hard): [arXiv 2210.09261](https://arxiv.org/abs/2210.09261) (Oct. 2022)
  * What it tests: 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH). These are the task for which prior language model evaluations did not outperform the average human.
* LiveBench: [arXiv 2406.19314](https://arxiv.org/abs/2406.19314) (Jun. 2024)
  * What it tests: (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis
* SimpleQA: [arXiv 2411.04368v1](https://arxiv.org/abs/2411.04368v1) (Nov. 2024)
  * What it tests: evaluates the ability of language models to answer short, fact-seeking questions.

### Math & STEM Tasks:

* GPQA: [arXiv 2311.12022](https://arxiv.org/abs/2311.12022) (Nov. 2023)
  * What it tests: biology, physics, chemistry knowledge. High quality and extremely difficult. Experts with PhDs get 65% accuracy (Qwen 3 235B-A22B Base gets 47.5%)
* GSM8K: [arXiv 2110.14168](https://arxiv.org/abs/2110.14168) (Oct. 2021)
  * What it tests: grade school math word problems
* MATH: [arXiv 2103.03874](https://arxiv.org/abs/2103.03874) (Mar. 2021)
  * What it tests: 12,500 challenging competition mathematics problems
* GPQA-Diamond: the diamond set, composed of 198 questions from [GPQA](https://arxiv.org/abs/2311.12022)

### Math & Text Reasoning

* MATH-500: [HF](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (Nov. 2024)
  * What it tests: sub-set of MATH from OpenAI's Let's Verify Step by Step.
* AIME (American Invitational Mathematics Examination): [2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | [2025](https://huggingface.co/datasets/opencompass/AIME2025)
* ZebraLogic: [arXiv 2502.01100](https://arxiv.org/abs/2502.01100) (Feb. 2025)
  * What it tests: logic grid puzzles derived from constraint satisfaction problems (CSPs)
* AutoLogi: [arXiv 2502.16906](https://arxiv.org/abs/2502.16906) (Feb. 2025)
  * What it tests: "synthesizing open-ended logic puzzles, and use it to develop a bilingual benchmark"; "program-based verification and controllable difficulty levels, enabling more reliable evaluation that better distinguishes models' reasoning abilities"

### Coding Tasks:

* EvalPlus: [arXiv 2305.01210](https://arxiv.org/abs/2305.01210) (May 2023)
  * What it tests: "is the code generated really correct? To answer this, we propose EvalPlus -- a code synthesis evaluation framework to rigorously benchmark the functional correctness of LLM-synthesized code."
* MultiPL-E: [arXiv 2208.08227](https://arxiv.org/abs/2208.08227) (Aug. 2017)
  * What it tests: "a system for translating unit test-driven code generation benchmarks to new languages. We create the first massively multilingual code generation benchmark by using MultiPL-E to translate two popular Python code generation benchmarks to 18 additional programming languages"
* MBPP: [arXiv 2108.07732](https://arxiv.org/abs/2108.07732) (Aug. 2016)
  * What it tests: "The Mostly Basic Programming Problems (MBPP) dataset contains 974 programming tasks, designed to be solvable by entry-level programmers."
* CRUX-O (CRUXEval): [arXiv 2401.03065v1](https://arxiv.org/abs/2401.03065v1) (Jan. 2024)
  * What it tests: Code Reasoning, Understanding, and eXecution Evaluation in python. Each function comes with an input-output pair, leading to two natural tasks (hence the Qwen 3 task was the output variant, presumably).
* BirdBench (aka Bird-SQL): [arXiv 2305.03111](https://arxiv.org/abs/2305.03111) (May 2023)

### Agent & Coding:

* BFCL v3 [blog post](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) (Sep. 2024)
  * What it tests: Multi-Turn & Multi-Step Function Calling Evaluation
  * Remarks: Would be cool to create a minimalistic environment that can deal with the given tools and such. May be possible to test without such a tool though. Looks like it has both "live" and non-live variants?

### Multilingual Tasks:

* MGSM: [arXiv 2210.03057](https://arxiv.org/abs/2210.03057) (Oct. 2022)
  * What it tests: Multilingual Grade School Math (MGSM) benchmark, by manually translating 250 grade-school math problems from the GSM8K dataset (Cobbe et al., 2021) into ten typologically diverse languages.
* MMMLU [HF](https://huggingface.co/datasets/openai/MMMLU) (Sep. 2024)
  * What it tests: translated the MMLU’s test set into 14 languages using professional human translators
  * Remarks: may restrict which languages are tested -- alternatively, show score per language
* INCLUDE: [arXiv 2411.19799](https://arxiv.org/abs/2411.19799) (Nov. 2024)
  * What it tests: a comprehensive knowledge- and reasoning-centric benchmark across 44 written languages that evaluates multilingual LLMs for performance in the actual language environments where they would be deployed; argues that existing benchmarks simply translate from English and ignore cultural and regional knowledge of the environment(s).
* Multi-IF: [arXiv 2410.15553](https://arxiv.org/abs/2410.15553) (Oct. 2024)
  * What it tests: Multi-Turn and Multilingual Instructions Following
  * Remarks: "hybrid framework combining LLM and human annotators" (i.e. probably not feasible)
* Polymath: [arXiv 2410.14702](https://arxiv.org/abs/2410.14702) (Oct. 2024)
  * What it tests: Multi-modal mathematical reasoning.
* LogiQA: [arXiv 2007.08124](https://arxiv.org/abs/2007.08124) (Jul. 2020)
  * What it tests: Machine Reading Comprehension with Logical Reasoning; expert-written questions for testing human Logical reasoning.
* Global MMLU-Lite: [HF](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite) (Dec. 2024)
  * Multilingual MMLU: 200 Culturally Sensitive (CS) and 200 Culturally Agnostic (CA) samples per language

### Alignment Tasks:

* IFEval "STRICT PROMPT": [arXiv 2311.07911](https://arxiv.org/abs/2311.07911) (Nov. 2023)
  * What it tests: "focuses on a set of "verifiable instructions" such as "write in more than 400 words" and "mention the keyword of AI at least 3 times". We identified 25 types of those verifiable instructions and constructed around 500 prompts, with each prompt containing one or more verifiable instructions"

(there are more, but they use external resources such as Creative Writing v3 or Chat Arena, etc.)

