## Contrastive Search for Encoder-Decoder Models

In this folder, we illustrate how to apply contrastive search on models (e.g. BART and T5) with encoder-decoder structure.

****
### Catalogue:
* <a href='#bart'>1. BART</a>
    * <a href='#bart_contrastive_search'>1.1. Contrastive Search</a>
    * <a href='#bart_greedy_search'>1.2. Greedy Search</a>
    * <a href='#bart_beam_search'>1.3. Beam Search</a>
* <a href='#t5'>2. T5</a>
    * <a href='#t5_contrastive_search'>2.1. Contrastive Search</a>
    * <a href='#t5_greedy_search'>2.2. Greedy Search</a>
    * <a href='#t5_beam_search'>2.3. Beam Search</a>


****

<span id='bart'/>

### 1. BART:

Here, we provide an example of how to apply different search methods on BART model.

```python
import torch
import sys
sys.path.append(r'./SimCTGBART/')
from simctgbart import SimCTGBART

# load BART model fine-tuned on CNN/DailyMail
model_path = "facebook/bart-large-cnn"
model = SimCTGBART(model_path)
tokenizer = model.tokenizer
if torch.cuda.is_available():
    model.cuda()
model.eval()

# prepare an example article
ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
```

<span id='bart_contrastive_search'/>

#### 1.1. Contrastive Search:

```python
with torch.no_grad():
    beam_width, alpha, decoding_len = 5, 0.5, 64
    ids = torch.LongTensor(tokenizer.encode(ARTICLE, add_special_tokens=False)).unsqueeze(0)
    dids = torch.LongTensor([tokenizer.eos_token_id, tokenizer.bos_token_id]).unsqueeze(0)
    if torch.cuda.is_available():
        ids, dids = ids.cuda(), dids.cuda()
        
    response = model.fast_contrastive_search(ids, dids, beam_width, alpha, decoding_len)
    id_list = []
    for idx in response:
        if idx == tokenizer.eos_token_id:
            break
        else:
            id_list.append(idx)
    print(tokenizer.decode(id_list))
    
    '''
    Liana Barrientos, 39, pleaded not guilty to two counts of "offering a false instrument for filing in the first degree" 
    In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    '''
```

**[Note]** In this example, we only apply an **off-the-shelf** BART model from huggingface and it was **not** trained with contrastive training (i.e. SimCTG). We highly recommend the users to use SimCTG to train the BART model before applying contrastive search on it.


<span id='bart_greedy_search'/>

#### 1.2. Greedy Search:

```python
with torch.no_grad():
    response = model.greedy_search(ids, decoding_len)
    print(tokenizer.decode(response))
    
    '''
    </s><s>Liana Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is 
    believed to still be married to four men, and at one time, she was married to eight men at once. In 2010, she stated 
    it was her "first and only" marriage</s>
    '''
```

<span id='bart_beam_search'/>

#### 1.3. Beam Search:


```python
with torch.no_grad():
    beam = 5
    response = model.beam_search(ids, beam, decoding_len)
    print(tokenizer.decode(response))

    '''
    </s><s>Liana Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree" 
    In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still 
    be married to four men, and</s>
    '''
```

****

<span id='t5'/>

### 2. T5:

Here, we provide an example of how to apply different search methods on T5 model.

```python
import torch
import sys
sys.path.append(r'./SimCTGT5/')
from simctgt5 import SimCTGT5

# load T5 model fine-tuned on CNN/DailyMail
model_path = "flax-community/t5-base-cnn-dm"
model = SimCTGT5(model_path)
tokenizer = model.tokenizer
if torch.cuda.is_available():
    model.cuda()
model.eval()

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
```

<span id='t5_contrastive_search'/>

#### 2.1. Contrastive Search:

```python
with torch.no_grad():
    beam_width, alpha, decoding_len = 5, 0.5, 64
    ids = torch.LongTensor(tokenizer.encode(ARTICLE, add_special_tokens=False, truncation=True, max_length=512)).unsqueeze(0)
    dids = torch.LongTensor([tokenizer.pad_token_id]).unsqueeze(0)
    if torch.cuda.is_available():
        ids, dids = ids.cuda(), dids.cuda()
    response = model.fast_contrastive_search(ids, dids, beam_width, alpha, decoding_len)
    id_list = []
    for idx in response:
        if idx == tokenizer.eos_token_id:
            break
        else:
            id_list.append(idx)
    print(tokenizer.decode(id_list))
    '''
    Liana Barrientos faces up to four men in a Bronx, New Jersey case. Prosecutors say her marriages were part of an immigration scam.
    '''
```

**[Note]** In this example, we only apply an **off-the-shelf** T5 model from huggingface and it was **not** trained with contrastive training (i.e. SimCTG). We highly recommend the users to use SimCTG to train the BART model before applying contrastive search on it.


<span id='t5_greedy_search'/>

#### 2.2. Greedy Search:

```python
with torch.no_grad():
    response = model.greedy_search(ids, decoding_len)
    print(tokenizer.decode(response))
    
    '''
    <pad> Liana Barrientos has been married 10 times, nine of them in the Bronx. She is facing two criminal counts 
    of "offering a false instrument for filing in the first degree" If convicted, Barrientos faces up to four men 
    in the case.</s>
    '''
```

<span id='t5_beam_search'/>

#### 2.3. Beam Search:


```python
with torch.no_grad():
    beam = 5
    response = model.beam_search(ids, beam, decoding_len)
    print(tokenizer.decode(response))

    '''
    <pad> Liana Barrientos has been married 10 times, nine of them in the Bronx. At one time, she was married to eight 
    men at once, prosecutors say.</s>
    '''
```
