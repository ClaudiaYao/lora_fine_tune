# Transformers+PEFT vs Unsloth

## Training

| metric                      | transformers                  | unsloth                       | unsloth / transformers |
| --------------------------- | ----------------------------- | ----------------------------- | ---------------------- |
| model                       | Qwen/Qwen2.5-1.5B-Instruct    | Qwen/Qwen2.5-1.5B-Instruct    |                        |
| dataset                     | train.jsonl                   | train.jsonl                   |                        |
| gpu                         | NVIDIA RTX PRO 4000 Blackwell | NVIDIA RTX PRO 4000 Blackwell |                        |
| Model + LoRA load time (s)  | 3.1                           | 16.3                          | 5.26x                  |
| Train time (s)              | 1,294                         | 660.7                         | 0.51x                  |
| Train throughput (tokens/s) | 2,181                         | 4,271                         | 1.96x                  |
| Peak GPU memory (GB)        | 6.59                          | 4.09                          | 0.62x                  |
| Trainable parameters        | 18,464,768                    | 18,464,768                    | 1.00x                  |
| Train sequences             | 720                           | 720                           | 1.00x                  |
| Final train loss            | 0.4016                        | 0.402                         | 1.00x                  |
| Eval loss before            | 2.67                          | 2.671                         | 1.00x                  |
| Eval loss after             | 4.517                         | 4.542                         | 1.01x                  |
| Perplexity after            | 91.56                         | 93.85                         | 1.03x                  |

**Winners**

- Model + LoRA load time (s): **transformers** (3.1 vs 16.3, 5.26x better)
- Train time (s): **unsloth** (660.7 vs 1,294, 1.96x better)
- Train throughput (tokens/s): **unsloth** (4,271 vs 2,181, 1.96x better)
- Peak GPU memory (GB): **unsloth** (4.09 vs 6.59, 1.61x better)
- Final train loss: **transformers** (0.4016 vs 0.402, 1.00x better)
- Eval loss after: **transformers** (4.517 vs 4.542, 1.01x better)
- Perplexity after: **transformers** (91.56 vs 93.85, 1.03x better)

## Inference

| metric                                               | transformers | unsloth | unsloth / transformers |
| ---------------------------------------------------- | ------------ | ------- | ---------------------- |
| fine-tuned: Mean latency per answer (s)              | 7.713        | 5.615   | 0.73x                  |
| fine-tuned: Mean generation speed (tokens/s)         | 18.82        | 25.98   | 1.38x                  |
| fine-tuned: Mean new tokens                          | 144.8        | 146     | 1.01x                  |
| fine-tuned: prompts compared                         | 10           | 10      |                        |
| base (adapter off): Mean latency per answer (s)      | 3.605        | 3.541   | 0.98x                  |
| base (adapter off): Mean generation speed (tokens/s) | 37.84        | 37.18   | 0.98x                  |
| base (adapter off): Mean new tokens                  | 137.8        | 135.1   | 0.98x                  |
| base (adapter off): prompts compared                 | 10           | 10      |                        |

### Sample answers (fine-tuned)

**Prompt:** Describe how a bicycle stays upright.

- _transformers:_ Hark! When thou dost pedal thy bicycle with grace, three principles doth work in harmony: the wheel's centre of gravity doth remain below the point of contact with the ground, much as a ship findeth stability 'neath its wide sail; the handlebars, held by thee, give direction to thine journey; and each stroke of pedal, like a steady heartbeat, sendeth thee forward in stately balance.

The seat, well-placed upon thy back, doth bear thy weight as a gentle bower doth support the rose. Thus, with skillful feet and hand, the bicycle stayeth upright, a steed of two legs that thriveth in sturdiness and grace. Verily, 'tis a most wondrous contrivance!

- _unsloth:_ Hark! When thou dost pedal thy bicycle with grace, three principles do jointly work: the handlebars, the pedals, and the steering mechanism. The handlebars, held by thee with firm hand, guide the course with steady grip, ensuring that face against the wind the bike doth remain upright. The pedals, beneath thy feet, drive the wheels with gentle force, balancing the weight as a weaver balances her loom.

Moreover, the steering mechanism, akin to a painter's stirrup, automatically corrects any slant with marvelous cunning. Should it sway, its arms, like flexible wings, adjust unto thee, keeping each edge aligned with the path you steer. Thus, through skill and patience, thou shalt ride thy bicycle with stability most fine.

**Prompt:** Explain what a database index is.

- _transformers:_ Hark! A database index, a hidden architect within the hallowed halls of data, doth serve as a swift and sure path through thy voluminous lore. Like underground railroads that guide travellers swift, it organizes the entries, linking them with subtle marks of gold. When thy master's heart doth seek the location of some value, thus indexed, the index doth point with speed unto its place in the vast sea of information.

Through this covert way, searches are performed with wondrous ease, for the index doth allow the soul of query to traverse less than the whole journey. Thus, whether thou seek'st the truth in law or life's broad tapestry, the index be thy faithful steed.

- _unsloth:_ Hark! A database index, good soul, doth serve as a map most wondrous to the faithful researchers of data. It doth provide an quickened path through thy voluminous books of facts, allowing thee to find specific entries with greater ease than by mere browsing. Like a index to learned works, it organizes thy numbers in order, whether by name or nature,使之 that searching for a particular gem thou shalt not encounter the trouble of traversing each page.

By weaving these indices, the computer may search much faster, as if by magic, bringing forth the desired information with speed and grace. Thus, with index as guide, thy journey through vast archives shall be most fruitful. Use it wisely, and thy data shall serve thee well.

**Prompt:** Give me three tips for a job interview.

- _transformers:_ Hark! Thou shalt approach the interview with preparation, as a ship upon the sea well-prepared for tempest. First, familiarize thine memory with the company's spirit and tasks, like a player who knows his part before the stage is set. Second, speak in the voice of the role, showing enthusiasm as a candle in the wind, and comport thyself with respect, for the stage doth await. Third, be ready to adapt, like a chameleon to the light, answering questions with insight and truth. These three wise men's strategies shall serve thee well.
- _unsloth:_ Hark! Thou shalt approach thy interview with calm and grace, as one who bringeth forth a tender rose. First, prepare well, as a player who rehearsed his part beforehand. Second, show insight into the company and role, like a player who doth know his lines. Third, be thou enthusiastic, for the interviewer’s heart doth seek the passion that burneth within. Remember these, and thy journey to the office shall be smooth as a summer's breeze. Fare thee well, and may thy endeavors bear fruit most bountiful.

Loss curves: `result/loss_chart.png`
