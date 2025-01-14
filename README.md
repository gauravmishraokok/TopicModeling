
# Topic Modeling Insights on lmsys-chat-1m Dataset Using BERTopic

[Notebook Link](https://colab.research.google.com/drive/1V_vJt-1qsvT-ZPgdl0_ll21ZrPrWR1oA?usp=sharing)

## Task at hand and Why?

- This Notebook provides a comprehensive analysis using **BERTopic**, a state-of-the-art topic modeling technique, applied to the [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset on Hugging Face.
- The primary goal of this analysis is to delve into the dataset to **uncover and understand the prevalent topics** discussed in user interactions.
- This insight is **crucial for training new models and finetuning older models** on most sought after topics.

## Dataset Overview

- The [**lmsys-chat-1m**](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset comprises over one million user interaction entries, making it a rich source for understanding natural language processing in **conversational AI**.
- As the dataset is **request access** and not completely open, The notebook needs to have a basic authentication token, It can be done using **Google Colab Secrets** for safety reasons.
![Notebook Image](https://i.imgur.com/DW7Okuq.jpeg)
- Following are the columns of the dataset ->
**Columns/Dataset Features:** **`conversation_id, model, conversation, turn, language, openai_moderation, redacted`**
**Rows/Number of Datapoints:** **`1,000,000`**

![Dataset Image](https://i.imgur.com/qscD8vB.png)

- Each entry in the "conversation" represents an **array of dictionaries** of both user prompts and assistant responses.
- General structure of the conversation looks like ->

$$\begin{array}{l}
\text{[} \\
\quad \{ \text{"content": "User Prompt 1", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response 1", "role": "assistant"} \}, \\
\quad \{ \text{"content": "User Prompt 2", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response 2", "role": "assistant"} \}, \\
\quad \ldots, \\
\quad \{ \text{"content": "User Prompt n", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response n", "role": "assistant"} \} \\
\text{]}
\end{array}$$

- Although due to resource constraints of Google Colab, I have used only ***10%*** subset of the original dataset.

## Loading the Data ->

### 1. Conversion of Hugging Face Dataset to Pandas DataFrame

#### Why?:
[Pandas](https://pandas.pydata.org/) DataFrames offer a more intuitive interface for data manipulation and are generally faster due to the optimized nature of the library.
#### Challenge:
The dataset in question was substantial in size, leading to **excessive RAM consumption** when attempting to load the entire dataset into memory for conversion at once. This issue was causing the computational **notebook to crash**.
#### Implementation Solution:
To mitigate the memory overload issue, a **batching system was implemented**. The process involves:
1. Extracting segments of the dataset of smaller size.
2.  Sequentially appending each batch to a Pandas DataFrame.
3. Iterating over 1 & 2 until n% of the data is loaded.

## Data Preprocessing
### *1. Handling Multilingual Data in the Dataset*

#### Challenge: Multilingual Data
The dataset contains **multilingual data**, which can lead to inconsistencies and inaccuracies in the analysis, as it complicates the linguistic processing and may skew the interpretation of topics.

#### Implementation Solution:
A **language based filtering** step was implemented to retain **only English** language texts.

This approach ensures that the dataset is **homogenized**, reducing complexity and improving the predictability of the model outcomes.

The language based filtered out data is stored in `df_english`.

### *2. Handling Multiple Interaction Turns in Conversations & Extraction of Text from the Complex Structure*

#### First Challenge : Multiple Interaction Turns
The dataset includes conversations that feature **multiple interaction turns** between a user and an assistant within a single conversation entry.
For effective topic modeling, it is essential to **capture the essence of all user prompts** throughout the conversation, not just the initial ones.

#### Second Challenge: Complex Dataset Structure and Targeted Content Extraction

The dataset exhibits a **complex nested structure** and for effective implementation of [BERTopic](https://maartengr.github.io/BERTopic/index.html), which requires a **simplified array of strings format**, it is crucial to perform selective content extraction.

The challenge lies in efficiently **isolating and extracting only the user prompts** from this intricate conversation structure, as these prompts contain the primary content necessary for our topic modeling analysis.  

#### Data Format:
Each conversation row is structured as follows:

$$\begin{array}{l}
\text{[} \\
\quad \{ \text{"content": "User Prompt 1", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response 1", "role": "assistant"} \}, \\
\quad \{ \text{"content": "User Prompt 2", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response 2", "role": "assistant"} \}, \\
\quad \ldots, \\
\quad \{ \text{"content": "User Prompt n", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response n", "role": "assistant"} \} \\
\text{]}
\end{array}$$

#### Implementation Solution:
To address this challenge, the `concatenate_user_messages()` function was developed.

This function **extracts all user prompts** from a conversation by matching the "role" of each dictionary with "user" in the array and then **concatenates these messages into a single continuous string**.

This method ensures that every part of the user's input is considered, providing a comprehensive basis for subsequent topic modeling and analysis.

The function `concatenate_user_messages` is applied to each conversation  within the `df_english['conversation']` column **iteratively** and stored in a **new column** named `combined_user_prompts`.

### *3. Handling Redacted Information*

#### Challenge:
The dataset contains **extensive privacy-oriented redactions**, where personal identifiers have been systematically replaced with standardized placeholders (e.g., NAME_1, NAME_2).

This redaction process affects **more than 25% of the dataset entries**, presenting a significant challenge for topic modeling accuracy.

#### Impact of the Challenge on the Result:
The prevalence of these redacted placeholders poses a **critical methodological challenge** for topic modeling:
- The **high frequency** of standardized placeholders can lead to their misidentification as significant topics
- BERTopic's algorithm may **incorrectly prioritize** these non-contextual elements due to their repetitive nature
- The **semantic value** of the content is potentially compromised when these placeholders are treated as meaningful tokens

#### Illustration:
Consider the following example:
```text
"NAME_1 went to the park on a run"
```
In this context, while **"park"** and **"run"** carry genuine t**opical significance** related to outdoor activities and leisure, the **placeholder "NAME_1"** represents **semantically irrelevant information** that could skew topic modeling results.

#### Implementation Solution:
To address this challenge, an **efficient Regular Expression (RegEx) based approach** was implemented. This solution:
- Provides **O(n) time complexity** for identifier removal
- Eliminates the need for complex NLP-based name recognition systems which would be much slower.
- Ensures **consistent and accurate removal** of standardized placeholders.

### *4. Handling Numerical Noise in Dataset*

#### Challenge:
The dataset contains **extensive numerical elements** within user prompts, particularly in instructional contexts (e.g., "write 500 words about..." or "list 10 ways to...").

While these numbers are relevant for instruction purposes, they represent **non-contextual noise** for topic modeling analysis.

#### Impact of the Challenge on the Result:
The presence of these numerical values poses several challenges for effective topic modeling:
- **Repetitive numerical patterns** can be mistakenly identified as significant topics.
- The **semantic clarity** of the content is potentially diluted by numerical noise as they add **unnecessary dimensionality**.

#### Illustration:
Consider the following example:
```text
"Write 500 words about climate change and list 3 main impacts"
```
In this context, while **"climate change"** and **"impacts"** carry genuine **topical significance**, the numbers **"500"** and **"3"** represent **semantically irrelevant information** that could interfere with accurate topic identification.

#### Implementation Solution:
To address this challenge, a **streamlined Regular Expression (RegEx) based function** was implemented. This solution:
- Maintains the **integrity of the textual content** while eliminating numerical noise.
- Implements a **simple yet effective cleaning mechanism** with O(n) computation.

### *5. Managing Conversational Length Imbalance*

#### Challenge:
The dataset exhibits **significant length disparities** in conversation turns, with some entries containing **exceptionally long dialogue sequences** (up to 214 turns) compared to the typical range of **1-5 turns**.
This imbalance presents a substantial challenge for effective topic modeling.

#### Impact of the Challenge on the Result:
The presence of these lengthy conversations introduces several critical concerns:
- **Disproportionate influence** on topic distribution due to sheer content volume.
- **Representation bias** towards themes present in longer conversations
- Very high number of turns create a **significantly skewed distribution** compared to typical conversations.

#### Implementation Solution:
To address this challenge, a **strategic truncation approach** was implemented:
- Limiting conversations to **150 words or less**
- Affecting **less than 0.2% of the dataset** (approximately 20 conversations in 100,000)
- Maintaining **dataset integrity** while reducing computational overhead

#### Alternative Solutions :
Some other ways of doing it would have been
 - Undersampling : Causes Loss of Data for 99.8% of Data.
 - Oversampling: Causes very very high domination of 0.2% of data and is bad for topic modelling.
 - Data Augmentation: High Computation and Unpredictable Behaviour as loss of context is possible.
 - Choosing Data Normalization by length is practical and affects only the miniscule 0.2% of data.

### *6. Stop Words Elimination Using spaCy*

#### Challenge:
The dataset contains **abundant stop words** (such as 'the', 'is', 'at', 'which') that add minimal semantic value to the text analysis. These common words appear with **high frequency across all documents**, potentially creating noise in our topic modeling process.

#### Impact of the Challenge on the Result:
- Creates **unnecessary computational overhead** during processing
- Leads to **less distinctive topic clusters** due to the prevalence of common words

#### Illustration:
Consider the following example:
```text
Original: "The cat is sitting on the mat in the garden and it is sleeping"
After Stop Word Removal: "cat sitting mat garden sleeping"
```
In this context, while the original sentence contains 13 words, the processed version retains only **5 key content words** that carry the actual semantic meaning.

The removal of stop words maintains the core meaning while significantly reducing the text length.

#### Implementation Solution:
A **spaCy-based stop words removal** approach was implemented. This solution:
- Leverages spaCy's **built-in stop words list** for efficient filtering
- Optimizes the dataset for **faster topic modeling performance**

### *7. Text Lemmatization Using spaCy*

#### Challenge:
The dataset contains **multiple variations of words** (such as 'running', 'runs', 'ran') that represent the same core concept.

These variations create **unnecessary complexity and dimensionality** in our topic modeling process, Thus mapping all of them to a base word is good idea.

#### Impact of the Challenge on the Result:
The presence of varied word forms affects topic modeling efficiency in several ways:
- Creates **redundant feature dimensions** in the vector space.
- Leads to **scattered semantic relationships** for essentially same concepts.
- Reduces the **effectiveness of term frequency** calculations due to split word counts.

#### Illustration:
Consider the following example:
```text
Original: "The children are running and playing in the gardens"
After Lemmatization: "child run and play in garden"
```
In this example, while maintaining the core meaning, lemmatization transforms 'children' to 'child', 'running' to 'run', 'playing' to 'play', and 'gardens' to 'garden', **reducing word variations while preserving semantic content.**

#### Implementation Solution:
A **spaCy-based lemmatization** approach was implemented. This solution:
- Utilizes spaCy's **sophisticated morphological analysis** for accurate lemmatization
- Maintains **semantic consistency** while reducing vocabulary size
- Enhances topic modeling by **consolidating related word forms**

### 8. Sequential Text Processing Pipeline from steps 1 to 7

#### Application

In this step, All the preprocessing functions were serially applied on the `combined_user_prompts` and then stored in a new column called `processed_text`.

```javascript
df_english['processed_text'] = (df_english['combined_user_prompts']
    .progress_apply(remove_redacted_names)
    .progress_apply(truncate_to_first_n_words)
    .progress_apply(remove_stop_words)
    .progress_apply(lemmatize_text)
    .progress_apply(remove_digits))
```

### Illustration ->
```python
# Original text: "NAME_1 spent 25 minutes running in the park yesterday"

# After sequential processing:
# 1. remove_redacted_names() -> "spent 25 minutes running in the park yesterday"
# 2. truncate_to_first_n_words() -> "spent 25 minutes running in the park yesterday"
# 3. remove_stop_words() -> "spent 25 minutes running park yesterday"
# 4. lemmatize_text() -> "spend 25 minute run park yesterday"
# 5. remove_digits() -> "spend minute run park yesterday"

# And Now, The text is ready for topic modelling.
```

# Topic Modeling for Processed Data using BERTopic

## What is BERTopic?

BERTopic is a cutting-edge topic modeling technique that uses transformer-based embeddings, such as BERT, combined with clustering and dimensionality reduction techniques to uncover hidden topics in textual data. Unlike traditional models like Latent Dirichlet Allocation (LDA), BERTopic leverages contextual embeddings, enabling it to produce more semantically meaningful topics.

---

### Key Components of BERTopic

#### 1. **BERT Embeddings**
- **What it does:** Utilizes transformer models (e.g., `paraphrase-mpnet-base-v2`) to generate contextual embeddings that represent text in a high-dimensional space.
- **Advantage:** Captures semantic nuances of text, outperforming traditional bag-of-words models.

#### 2. **Dimensionality Reduction**
- **Technique:** UMAP (Uniform Manifold Approximation and Projection)
- **Purpose:** Reduces high-dimensional embeddings into a lower-dimensional space while preserving semantic relationships.
- **Parameters:**
  - `n_neighbors`: Controls local vs. global structure preservation.
  - `n_components`: Number of dimensions in the reduced space.
  - `min_dist`: Controls clustering density.
  - `metric`: Defines distance calculations (e.g., `cosine`).

#### 3. **Clustering**
- **Technique:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
- **Purpose:** Groups similar embeddings into clusters.
- **Parameters:**
  - `min_cluster_size`: Minimum documents per cluster.
  - `min_samples`: Controls outlier sensitivity.
  - `metric`: Defines the clustering distance measure (e.g., `euclidean`).

#### 4. **Topic Representation**
- **Technique:** c-TF-IDF (Class-based Term Frequency-Inverse Document Frequency)
- **Purpose:** Identifies the most representative terms for each topic.
- **Result:** Enhances interpretability of discovered topics.

---

### Workflow Diagram

Below is a visual representation of BERTopic’s workflow:

![BERTopic Workflow](https://github.com/MaartenGr/BERTopic/blob/master/docs/img/algorithm.png?raw=true)

---

## Configuration of Our BERTopic Model

Below is a detailed explanation of the configurations used in our BERTopic model:

| **Component**         | **Parameter**        | **Value**              | **Effect**                                                                 |
|-----------------------|---------------------|------------------------|---------------------------------------------------------------------------|
| **Embedding Model**   | `embedding_model`   | `paraphrase-mpnet-base-v2` | Provides contextual embeddings with high semantic accuracy.             |
| **UMAP**              | `n_neighbors`       | 20                     | Balances local and global structure preservation.                        |
|                       | `n_components`      | 8                      | Sets the dimensionality of the reduced space.                           |
|                       | `min_dist`          | 0.1                    | Lower values create denser clusters.                                    |
|                       | `metric`            | `cosine`               | Measures similarity using cosine distance.                              |
| **HDBSCAN**           | `min_cluster_size`  | 50                     | Ensures meaningful cluster sizes.                                       |
|                       | `min_samples`       | 10                     | Balances sensitivity to outliers.                                       |
|                       | `metric`            | `euclidean`            | Defines clustering distance calculations.                               |
| **Topic Representation** | `top_n_words`       | 20                     | Highlights the top 20 words for each topic.                             |

---

## Results: Prevalent Topics

Below are some of the prominent topics discovered by our BERTopic model:

| **Topic ID** | **Name**                       | **Count** | **Representative Terms**                          |
|--------------|--------------------------------|-----------|--------------------------------------------------|
| 0            | `Roleplay`           | 3,987     | story, girl, character, game, roleplay          |
| 1            | `AI Assistant Queries`         | 2,809     | assistant, completion, repeat, system, instruction |
| 2            | `Programming`         | 1,079     | import, int, self, const, return, class         |
| 3            | `Business and Industry`    | 962       | china, ltd, co, introduction, chemical          |
| 4            | `Coding Assistance`            | 721       | code, function, loop, debug, variable           |
| 5            | `Educational Resources`        | 643       | book, tutorial, explain, learn, teach           |
| 6            | `Physics & Mathematics`        | 489       | equation, solve, gravity, acceleration, theorem |

---

## Visualizations

1. **Intertopic Distance Map**
   - **Description:** Visualizes the relationships between topics in a 2D space using UMAP-reduced embeddings.
   - **Key Takeaways:**
     - Topics positioned closer together share higher semantic similarity.
     - Dense clusters indicate closely related or overlapping topics.

   ![Intertopic Distance Map](https://i.imgur.com/G2tlZm8.png)

2. **Topic Word Scores Bar Chart**
   - **Description:** Displays the top representative words for selected topics along with their c-TF-IDF scores.
   - **Key Takeaways:**
     - Highlights the most significant terms contributing to each topic.
     - Provides a quick comparison of word importance across topics.

   ![Topic Word Scores Barchart](https://i.imgur.com/HHK4Q54.png)

3. **Similarity Matrix**
   - **Description:** Shows the semantic similarity scores between topics.
   - **Key Takeaways:**
     - Darker cells indicate higher similarity between topic pairs.
     - Useful for identifying groups of interrelated topics.

   ![Similarity Matrix](https://i.imgur.com/vKqoUKN.png)

4. **Topic Probability Distribution**
   - **Description:** Visualizes the distribution of topic probabilities for documents in the dataset.
   - **Key Takeaways:**
     - Shows the relative dominance of topics across the corpus.
     - Highlights the prevalence of dominant and secondary topics in the dataset.

## Additional Insights

- **Topic Evolution:** The model enables dynamic topic modeling, making it possible to track changes in topics over time or across specific intervals in the dataset.
- **Application Scope:**
  - **Customer Feedback Analysis:** Extract recurring themes from user reviews or customer service logs.
  - **Content Categorization:** Automatically classify documents, articles, or blogs into relevant categories.
  - **Academic Research:** Reveal trends, methodologies, and key topics across scientific literature.
  - **AI Training:** Provides a foundation for training and fine-tuning conversational AI models based on prevalent topics.

---

## Conclusion

This Notebook provides a comprehensive analysis using **BERTopic**, a state-of-the-art topic modeling technique, applied to the [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset on Hugging Face. The **primary goal** was to delve into the dataset to **uncover and understand the prevalent topics discussed in user interactions**. These insights are crucial for training new models and fine-tuning older models on the most sought-after topics.

 The task involved uncovering dominant themes and their distributions in a dataset with multilingual content and varying conversational structures. The robust preprocessing pipeline solved subproblems such as:
  - **Handling multilingual data** to ensure consistent insights.
  - **Resolving conversational imbalances** due to varying turn lengths.
  - **Managing redacted or noisy information** and addressing numerical anomalies in the data.
  - **Streamlining text** using lemmatization and stop word elimination through spaCy.

The sequential processing workflow enabled a clean, consistent dataset that could be effectively modeled.

### Key Achievements

1. **Topic Insights:** Dominant themes such as **AI Assistant Queries, Programming Concepts, Educational Resources**, and others were identified and visualized using inter-topic distance maps and probability distributions.
2. **Advanced Preprocessing:** Successfully addressed multilingual challenges, redacted data issues, and conversational imbalances.
3. **Versatile Applications:** The model's insights are applicable across domains like content categorization, conversational AI training, and customer feedback analysis.
4. **Visualization Mastery:** Interactive visualizations provided deep insights into topic relationships, term significance, and dataset structure.

### Future Scope

- **Real-time Evolution Tracking:** Enhance the pipeline to dynamically track topic changes across time.
- **Multilingual Integration:** Extend support for diverse datasets with varied linguistic and structural patterns.
- **Customized Applications:** Tailor the workflow to meet specific industry needs, such as domain-specific categorization or customer experience optimization.
- **Robust Training for Robust Performance** : Training the model more robustly covering 100% of the dataset and handle the outlier cases better for better insights.

The methodologies demonstrated here highlight the power of modern NLP techniques in managing and interpreting complex datasets, paving the way for innovative, actionable applications in both research and industry.

## Acknowledgments

- [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) - A dataset of user interactions with an AI chatbots.
- [BERTopic](https://github.com/MaartenGr/BERTopic) - A state-of-the-art topic modeling technique.
