You will be given a scientific text (abstract -maybe introduction included) of a research article. The goal of this task is to identify scientific terms and their definitions in the text.

This task is formulated as a combination of Named Entity Recognition (NER) and Relation Extraction (RE).

The annotation process consists of three steps:
1- Identify TERMS - words or phrases that represent scientific or technical concepts.
2 Identify DEFINITIONS - text spans that explain the meaning of those terms.
3- Link the TERM to its corresponding DEFINITION to form a term- definition pair.

A TERM usually names a concept, method, model, or phenomenon in the domain (e.g., neural network, gradient descent, transformer).
A DEFINITION is a phrase or clause that explains what the term means (e.g., a machine learning model composed of layers of interconnected artificial neurons).
Your goal is to accurately identify both elements and connect them, ensuring that each TERM is linked to the correct DEFINITION within the text.

The texts may contain multiple term–definition pairs, and each valid pair should be annotated separately.

When annotating, focus on:
- selecting precise spans
- capturing complete definitions
- linking the correct term–definition pairs

Details are given as follows.


ENGLISH GUIDELINE
Patterns:
1- X is Y
Example: A [neural network]Term is [a computational model composed of layers of interconnected artificial neurons]Definition.

2- X refers to Y
Example: [Reinforcement learning]Term refers to [a training method where an agent learns by interacting with an environment and receiving rewards]Definition.

3- X is defined as Y
Example: [Gradient descent]Term is defined as [an optimization algorithm used to minimize a model’s loss function]Definition.

4- Appositive Definitions
Example: [Dropout]Term, [a regularization technique]Definition, randomly disables neurons during training.

5- Parenthetical Definitions
Example: [LLM]Term ([large language model]Definition) is a neural network trained on massive text datasets.

Rules
1- Select Minimal Span: Only highlight the exact term and definition.
2- Capture Complete Definitions: Definitions should explain the concept clearly.
3- One Term per Definition Instance: If a term is defined multiple times in the text, annotate each instance separately.
4- Skip Non-Definitions: If a term appears without explanation, do not annotate it.


Examples:
1- [Overfitting]Term is [a situation where a model performs well on training data but poorly on unseen data]Definition.
2- A [transformer]Term is [a neural network architecture that relies on self-attention mechanisms]Definition.
3- An [embedding]Term is [a vector representation of data, such as words or images]Definition.
4- [Backpropagation]Term is [an algorithm used to compute gradients in neural networks]Definition.
5- [Reinforcement learning]Term is [a machine learning paradigm where an agent learns by receiving rewards or penalties]Definition.

TURKISH GUIDELINE
Patterns
1- X ... -dir / -dır / -dur / -dür
Example: [Sinir ağı]Term, [birbirine bağlı yapay nöron katmanlarından oluşan bir model]Definitiondir.
2- X ... olarak tanımlanır
Example: [Gradyan inişi]Term, [kayıp fonksiyonunu minimize etmek için kullanılan bir optimizasyon algoritması]Definition olarak tanımlanır.
3- X ... ifade eder
Example: [Aşırı öğrenme]Term, [modelin eğitim verisine aşırı uyum sağlamasını]Definition ifade eder.
4- Appositive Structure
Example: [Dropout]Term, [bir düzenlileştirme tekniği]Definition, eğitim sırasında bazı nöronları rastgele devre dışı bırakır.
5- Abbreviation
Example: [CNN]Term, [evrişimli sinir ağı]Definition anlamına gelir.

Examples:
1- [Geri yayılım]Term, [sinir ağlarında gradyan hesaplamak için kullanılan bir algoritma]Definitiondır.
2- [Gömme]Term, [kelimeleri yoğun vektörler olarak temsil eden bir yöntem]Definitiondir.
3- [Transformer]Term, [dikkat mekanizmasına dayanan bir sinir ağı mimarisi]Definitiondir.
4- [Takviye öğrenmesi]Term, [bir ajanın ödül sinyalleri aracılığıyla öğrenmesini sağlayan bir makine öğrenmesi yaklaşımı]Definitiondır.

