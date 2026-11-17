# Qwen3.8-27B: the 64 unnamed knowledge clusters

Checkpoint `Qwen3.8-27B-BF16-q8-mlx2@2dd8d304`. Clusters from `atlas-clusters-696591553` (no labels used to
form them). Functional tests from window 2, `boxes-de6adef06`, branch `claude/amoeba-27b-80bc6a`.

How to read it:
- **Best add:** the task that gains most when the cluster is added to the 53.7% export core, minus the
  gain from a random twin with the same per-layer neuron counts.
- **Largest removal loss:** the task that loses most when the cluster is removed from the full model,
  again minus the twin.
- Each task uses 32 held items, so one item is 0.031. A label needs an add gain of at least 0.1. Gains under
  about 0.2 (six items or fewer) are weak evidence. Only C32, C60 and C22 are clearly real.
- "english" means adding the cluster lowers English cost by more than its twin does. "unclassified"
  means nothing showed on the six tasks tested: capitals (three splits), 3-digit arithmetic, French and
  recall. It does not mean the cluster does nothing.
- **Likely content:** my reading of the top tokens, the most selective corpus source and the median
  layer. It is a hypothesis, not a test result.
- Source selectivity: how much more often the cluster fires on that corpus source than on average.

|Label|Cluster|Neurons|Median layer|Most selective source (×)|Best add: task, gain vs twin|Largest removal loss vs twin|English Δ vs twin|Top tokens|Likely content (untested reading)|
|---|---|---|---|---|---|---|---|---|---|
|arithmetic|C32|7,605|45|arithmetic (2.8)|arithmetic +0.72|arithmetic +0.75|+0.003|  1 0 . 2||
|arithmetic|C44|3,881|47|repo_markdown (1.5)|arithmetic +0.16|arithmetic +0.06|+0.003|0   1 2 4||
|arithmetic|C35|10,362|30|recall (2.4)|arithmetic +0.16|capitals +0.03|+0.017|. ,  the  to  and||
|french|C60|13,500|28|french (4.2)|french +0.53|french +0.44|+0.002|: French , \n .||
|french|C11|11,816|27|tinystories_train (2.3)|french +0.16|arithmetic +0.06|-0.006|. ,  to  the  and||
|french|C37|10,288|36|recall (3.9)|french +0.12|capitals +0.03|+0.002|.  the : ,  to||
|french|C48|8,828|58|repo_markdown (2.0)|french +0.12|arithmetic +0.03|+0.004|   the , 1 2||
|french|C51|8,952|33|tinystories_train (1.7)|french +0.12|capitals_fresh +0.03|-0.004|. ,  you  "  and||
|english|C10|11,278|32|tinystories_train (2.2)|capitals_fresh +0.03|capitals_phrasing +0.03|-0.027|.  to  and ,  the||
|english|C62|9,231|33|tinystories_train (2.4)|capitals +0.00|capitals +0.03|-0.024| to  the .  and ,||
|capitals_phrasing|C22|6,802|25|capitals (2.6)|capitals_phrasing +0.25|capitals_fresh +0.22|+0.013| is  the  of  capital ,||
|unclassified|C2|14,952|25|repo_markdown (1.5)|arithmetic +0.09|capitals +0.03|+0.016| the , . 2 1||
|unclassified|C49|4,099|44|arithmetic (1.2)|arithmetic +0.09|arithmetic +0.03|+0.004|. ,  the  of \n\n||
|unclassified|C21|8,623|28|repo_markdown (1.9)|capitals_fresh +0.09|arithmetic +0.03|-0.008|1 , 2    the||
|unclassified|C54|11,000|44|donor_generated (1.9)|french +0.09|capitals +0.03|+0.009| the  of , .  and||
|unclassified|C30|5,023|40|donor_generated (1.6)|capitals_fresh +0.09|capitals +0.00|+0.005| the  of  in ,  is||
|unclassified|C45|4,690|38|donor_generated (1.5)|french +0.09|capitals +0.03|-0.003| the  of ,  to  a||
|unclassified|C58|3,611|43|donor_generated (1.3)|arithmetic +0.09|capitals_phrasing +0.03|+0.003|   the , 2  of||
|unclassified|C56|11,126|27|french (1.9)|arithmetic +0.06|capitals +0.00|+0.014|. , \n \n\n  the||
|unclassified|C23|5,223|46|arithmetic (1.6)|arithmetic +0.06|capitals_fresh +0.03|+0.003|0 1 2   .||
|unclassified|C20|12,962|33|recall (2.0)|french +0.06|capitals +0.03|+0.011|. ,  the   1||
|unclassified|C33|7,149|40|repo_markdown (1.7)|french +0.06|capitals +0.03|+0.003|\n ,  ` ` :|inline code and backticks|
|unclassified|C52|6,710|34|repo_markdown (1.6)|capitals_fresh +0.06|capitals_phrasing +0.03|-0.003| | , . \n  |markdown tables (pipes)|
|unclassified|C53|14,796|29|repo_markdown (1.4)|arithmetic +0.06|capitals_phrasing +0.03|+0.010|. ,  the  to  and||
|unclassified|C24|2,106|40|donor_generated (1.4)|arithmetic +0.06|capitals +0.03|+0.000| the  of  periodic ,  in|science facts ("periodic")|
|unclassified|C38|1,962|40|donor_generated (1.2)|arithmetic +0.06|capitals +0.00|+0.001| a  the  of ,  to||
|unclassified|C3|6,218|36|tinystories_train (1.7)|arithmetic +0.06|capitals_fresh +0.03|+0.003| the  and  to  a  it||
|unclassified|C17|9,173|42|donor_generated (1.5)|capitals_fresh +0.06|capitals +0.03|-0.005| of ,  the  a .||
|unclassified|C16|12,933|35|repo_markdown (1.6)|french +0.03|arithmetic +0.06|-0.002|. ,  the  is  and||
|unclassified|C9|3,039|55|repo_markdown (1.5)|arithmetic +0.03|capitals_phrasing +0.03|-0.001|rel ier 0 2 .|late-layer word pieces ("rel", "ier")|
|unclassified|C13|15,193|21|tinystories_train (1.5)|arithmetic +0.03|capitals +0.00|+0.007|.  to ,  the  and||
|unclassified|C15|3,104|40|donor_generated (1.5)|arithmetic +0.03|arithmetic +0.03|-0.002| the  of ect  t  plate|technical words ("plate")|
|unclassified|C46|3,372|38|donor_generated (1.4)|arithmetic +0.03|capitals_phrasing +0.03|-0.005| the  of  a , ets||
|unclassified|C19|4,286|39|donor_generated (1.3)|arithmetic +0.03|capitals_fresh +0.03|-0.003| the  a  of ,  to||
|unclassified|C26|15,388|2|tinystories_train (2.5)|arithmetic +0.03|capitals +0.03|+0.009| upon adge :  time .|story openings ("once upon a time"), layer 2|
|unclassified|C40|2,572|44|recall (2.2)|arithmetic +0.03|capitals +0.00|+0.003|.  the   , \n\n||
|unclassified|C43|14,265|32|repo_markdown (2.2)|capitals +0.03|arithmetic +0.03|-0.008|\n , .  the  |markdown line structure|
|unclassified|C28|12,727|37|repo_markdown (2.2)|capitals +0.03|arithmetic +0.12|-0.001|\n `  ` ,  |code / markdown backticks|
|unclassified|C6|8,266|39|repo_markdown (2.1)|french +0.03|capitals +0.03|+0.019| the \n > 2  |markdown quotes (">")|
|unclassified|C55|6,088|27|french (2.1)|french +0.03|capitals_fresh +0.03|+0.002|. ,  the  of \n\n||
|unclassified|C25|7,803|36|tinystories_train (2.0)|arithmetic +0.03|capitals_phrasing +0.00|+0.004| the  and ,  a  to||
|unclassified|C50|5,085|32|french (1.9)|capitals_fresh +0.03|arithmetic +0.03|+0.004|. \n\n , : \n||
|unclassified|C12|10,167|16|french (1.6)|capitals +0.03|capitals +0.00|+0.006|. ,  the \n\n  to||
|unclassified|C61|13,699|40|repo_markdown (1.6)|french +0.03|capitals_fresh +0.03|+0.007| the , .  of  and||
|unclassified|C57|11,450|32|donor_generated (1.6)|capitals_fresh +0.03|capitals_phrasing +0.03|-0.007| the . ,  of  and||
|unclassified|C36|9,018|36|repo_markdown (1.6)|capitals_fresh +0.03|capitals +0.00|+0.013|  . , 0 2||
|unclassified|C7|8,957|8|repo_markdown (1.5)|capitals +0.03|capitals +0.03|+0.004|1 3 2 y 0|early-layer digits and letters (layer 8)|
|unclassified|C42|10,854|21|tinystories_train (1.5)|french +0.03|capitals_fresh +0.03|+0.001|. , \n\n  the  of||
|unclassified|C8|6,607|60|repo_markdown (1.5)|capitals +0.03|arithmetic +0.06|-0.003|.  the 0 2 1|late-layer digits|
|unclassified|C5|3,491|41|donor_generated (1.4)|capitals_fresh +0.03|capitals +0.03|-0.002| the  of  and 1  a||
|unclassified|C31|3,630|40|donor_generated (1.4)|arithmetic +0.03|capitals_phrasing +0.03|-0.004| the  of ,  and  in||
|unclassified|C47|3,531|40|donor_generated (1.3)|capitals +0.03|arithmetic +0.03|+0.000| the  of ,  a  to||
|unclassified|C1|4,652|36|donor_generated (1.3)|french +0.03|capitals_fresh +0.00|-0.002|\n\n . , ** :|markdown headings and bold|
|unclassified|C14|20,094|32|donor_generated (1.2)|capitals +0.03|capitals_fresh +0.03|+0.008| the , .  of  and||
|unclassified|C18|7,435|3|french (1.1)|capitals_fresh +0.03|capitals +0.03|-0.002|: adge ak ull ad|early-layer subword pieces (layer 3)|
|unclassified|C39|7,895|37|repo_markdown (2.1)|capitals +0.00|capitals +0.00|+0.005|2 X x  mlx 1|code identifiers ("mlx", x)|
|unclassified|C27|6,344|39|recall (1.7)|capitals +0.00|capitals_phrasing +0.00|+0.010|. \n\n  the  is ,||
|unclassified|C34|7,364|3|french (1.7)|capitals_phrasing +0.00|capitals +0.00|+0.008|: adge \n\n ull 2|early-layer subword pieces (layer 3)|
|unclassified|C59|5,879|37|donor_generated (1.6)|capitals +0.00|capitals_phrasing +0.03|-0.002| the  of . \n\n ,||
|unclassified|C63|3,841|28|tinystories_train (1.6)|capitals +0.00|capitals_fresh +0.03|+0.004| and .  the ,  to||
|unclassified|C0|4,912|40|donor_generated (1.6)|capitals_phrasing +0.00|capitals +0.03|-0.008| the  of ,  in th||
|unclassified|C4|4,843|35|donor_generated (1.5)|capitals_phrasing +0.00|capitals_phrasing +0.03|+0.005|\n\n : yclopedia  the  entry|encyclopedia entries|
|unclassified|C29|7,995|33|tinystories_train (1.5)|capitals +0.00|capitals_fresh +0.03|+0.004|. ,  to  not  and||
|unclassified|C41|7,315|39|donor_generated (1.4)|capitals +0.00|capitals_fresh +0.03|+0.015|.  to ,  the  a||
