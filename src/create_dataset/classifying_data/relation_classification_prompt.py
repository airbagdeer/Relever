relation_classification_prompt = """ 
You are a **binary classifier** that determines whether a paragraph provides evidence for a specific relation between two entities.

---

### **Task Description**

You will be given:

* A **Sentence** describing a relation between:

  * an **Abstract Entity Type** (a class of entities), and

  * a **Constant Entity** (a specific, well-defined entity).

* A **Paragraph** that may or may not provide evidence for this relation.

Your task is to determine whether the paragraph **evidences the specified relation** between an **instance of the abstract entity type** and the **constant entity**.

---

### **Definitions**

* **Abstract Entity Type**: A semantic class of entities (e.g., *Energy companies*, *Universities*, *Airlines*).  
   The paragraph may mention a concrete instance of this class (e.g., *Equinor*, *University of Sussex*).

* **Constant Entity**: A specific, well-defined entity (e.g., *the European Union market*, *the Asia-Pacific region*).

* **Relation**: The action or relationship that must hold between the abstract entity instance and the constant entity (e.g., *invests in*, *collaborates with*, *serves region*).

---

### **Decision Criteria**

Return `true` **only if** the paragraph provides clear evidence that:

1. An **instance** of the abstract entity type is present, **and**

2. The specified **relation is asserted or clearly implied**, **and**

3. The relation holds **between that instance and the given constant entity**.

Return `false` if **any** of the following are true:

* The abstract entity or constant entity is mentioned only as **background, comparison, or context**  
* A **different relation** is described  
* The relation is **hypothetical, preparatory, regulatory**  
* If the paragraph **explicitly** states that the relation **does not occur** (i.e. is negated)  
* The relation holds with a **different entity**  
* The relation cannot be reasonably inferred from the paragraph alone

Use **only the information in the paragraph**. Do not rely on external knowledge.

---

### **Output Format**

Return **only** a JSON object with your classification. Do **not** include explanations, reasoning, or extra text.

---

### **Reference Examples:**

# Example 1:  
Input:  
Sentence: "Startups participating in the Y Combinator accelerator."  
Abstract Entity: "Startups"  
Constant Entity: "the Y Combinator accelerator"  
Relation: "participating in"  
Paragraph: "Dropbox\\n\\n. Dropbox Paper entered open beta in August 2016, allowing anyone to join and test the product. Mobile apps for Android and iOS were also released. In January 2017, Dropbox Paper was officially launched. Aimed for businesses, Dropbox Paper was described as \\"one part online document, one part collaboration, one part task management tool, one part content hub\\" by Rob Baesman, Dropbox's head of product, and allows for importing, editing, and collaboration on \\"a number of other file types from Google, Microsoft, and others\\". Users have devised a number of uses for and mashups of the technology that expand Dropbox's functionality. These include: sending files to a Dropbox via Gmail; using Dropbox to sync instant messaging chat logs; BitTorrent management; password management; remote application launching and system monitoring; and as a free web hosting service. Dropbox has received several awards, including the Crunchie Award in 2010 for Best Internet Application, and Macworld's 2009 Editor's Choice Award for Software. It was nominated for a 2010 Webby Award, and for the 2010 Mac Design Awards by Ars Technica. Dropbox's mobile iPhone app release in 2010 was among the top 10 \\"best apps\\" selected by Alex Ahlund, former CEO of two websites focused on mobile apps, and the company's Android app was also selected as one of the top five \\"best apps\\" in a list compiled in 2010 by Jason Hiner for ZDNet. Founders Drew Houston and Arash Ferdowsi were named among the top 30 under 30 entrepreneurs by Inc. in 2011\. In 2011, Business Insider named Dropbox the world's sixth most valuable startup, and in 2017, the publication ranked Dropbox as the eighth most valuable US startup, with a valuation of $10 billion. It has been described as one of Y Combinator's most successful investments to date. Apple launched its own cloud storage service later in 2011, iCloud, but this did not hold back Dropbox's growth. In January 2012, Dropbox was named startup of the year by TechCrunch. Dropbox has been blocked in China since 2014"

Expected Output: true  
Explanation: The paragraph clearly talks about dropbox, which is a startup. At the end of the paragraph, the following is mentioned "It has been described as one of Y Combinator's most successful investments to date.". Here, "it" represents dropbox, and from the context we can infer that Y Combinator invested in dropbox. Because Y Combinator is a startup accelerator, and it invests in startups. Notice that here, the relation is not explicitly mentioned, but can be inferred from the context.

# Example 2:  
Input:  
Paragraph: "Universities collaborating with the European Space Agency."  
Abstract Entity: "Universities"  
Constant Entity: "the European Space Agency"  
Relation: "collaborates with"  
Paragraph: "University of Sussex\\n\\n. The results of the Research Excellence Framework 2014 show that 98% of research activity at Sussex is categorised as 'world-leading' (28%), 'internationally excellent' (48%) or 'internationally recognised' (22%) in terms of originality, significance and rigour. Sussex has a number of research collaborations with other Higher Education institutions as well as governmental and non-governmental organisations and institutes around the world. For example, the Harvard Sussex program is a long-standing research collaboration between Sussex and Harvard University focusing on public policy towards chemical and biological weapons. The CBW Conventions Bulletin is a quarterly newsletter published by the HSP. Sussex-Cornell Partnership, the Sussex-Bocconi-Renmin Intrapreneurship Hub and the Sussex-Lund Partnership in Middle Eastern and North African Studies are recent examples. Sussex also co-coordinates the Consortium for the Humanities and the Arts. Sussex is also one of the eight universities of the Tyndall Centre network. In Europe, Sussex is one of the collaborating institutions of the Paul Scherrer Institute, the largest research institute in Switzerland, focusing on issues of technology and the natural sciences. Sussex is involved with many projects with the EU and with European countries. For example, BAR research is an Anglo-French collaboration between the Sussex, the East Sussex County Council and three French universities. Nationally, Sussex is involved in a number of partnerships including the Nexus Network (A partnership between Sussex, University of Cambridge and the University of East Anglia) and CIED (a collaboration between Sussex, Oxford University and University of Manchester). The university is also a partner of the Metropolitan Police, with Demos (UK think tank) and Palantir Technologies. In recent years,\[when?\] the institutes for the study of consciousness science, Centre for Advanced International Theory (CAIT), the institute for the study of corruption and the Middle East studies institute were opened at the university. The university also has a Genome Damage and Stability Centre, a nuclear magnetic resonance facility and a purpose-built apparatus in cryogenic research. In terms of policy, Sussex is highly involved with the UK government, the UN and governments around the world. For example, the university is a UN Habitat partner. Nationally, the UK Trade Policy Observatory was set up at the university to offer the UK government, the UK industry as well as the public advice in addressing trade issues resulting from Brexit"

Expected Output: false  
Explanation: The paragraph does talk about the University of Sussex, but it does not mention any collaboration between it and the European Space Agency. Even if you know Sussex University is collaborating with the European Space Agency, but its not mentioned in the paragraph, the output will be 0\.

# Example 3:   
Input:  
Sentence: "Energy companies investing in the North Sea offshore wind sector."  
Abstract Entity: "Energy Companies"  
Constant Entity: "the North Sea offshore wind sector"  
Relation: "invests in"  
Paragraph: "Norwegian energy major Equinor is deepening its shift away from a purely oil and gas identity, presenting offshore wind as a core pillar of its future earnings. The state-backed group, long known under its former name Statoil, has been steadily expanding a portfolio of renewable assets to complement its traditional petroleum business, with a particular emphasis on projects in northern European waters. Executives have repeatedly pointed to the North Sea as a natural arena where the company\\u2019s decades of offshore engineering expertise can be redeployed for low-carbon power generation. In that context, Equinor allocates substantial financial resources to the North Sea offshore wind sector, positioning the area as a cornerstone of its low-carbon investment pipeline. Market analysts say this approach allows the firm to leverage existing maritime infrastructure, subsea know-how and relationships with regional regulators. Recent announcements include plans for additional fixed-bottom and floating turbines near existing oil fields, a move seen as a way to cut emissions from upstream operations while supplying power to nearby grids. Environmental groups have cautiously welcomed the company\\u2019s evolving strategy, though they argue the pace of change must accelerate if Europe is to meet its 2030 climate commitments.""Norwegian energy major Equinor is deepening its shift away from a purely oil and gas identity, presenting offshore wind as a core pillar of its future earnings. The state-backed group, long known under its former name Statoil, has been steadily expanding a portfolio of renewable assets to complement its traditional petroleum business, with a particular emphasis on projects in northern European waters. Executives have repeatedly pointed to the North Sea as a natural arena where the company\\u2019s decades of offshore engineering expertise can be redeployed for low-carbon power generation. In that context, Equinor allocates substantial financial resources to the North Sea offshore wind sector, positioning the area as a cornerstone of its low-carbon investment pipeline. Market analysts say this approach allows the firm to leverage existing maritime infrastructure, subsea know-how and relationships with regional regulators. Recent announcements include plans for additional fixed-bottom and floating turbines near existing oil fields, a move seen as a way to cut emissions from upstream operations while supplying power to nearby grids. Environmental groups have cautiously welcomed the company\\u2019s evolving strategy, though they argue the pace of change must accelerate if Europe is to meet its 2030 climate commitments."

Expected Output: true  
Explanation: The paragraph preserves the actor (energy companies → Equinor as an instance), preserves the action (investing → allocating substantial financial resources, expanding projects) and preserves the target (the North Sea offshore wind sector)

# Example 4:  
Input:  
Sentence: "Airlines serving the Asia-Pacific region."  
Abstract Entity Type: "Airlines"  
Constant Entity: "the Asia-Pacific region"  
Relation: "serves region"  
Paragraph: "Route planners at several global carriers have been quietly rebalancing capacity toward underserved long-haul corridors that connect secondary cities with fewer daily frequencies. In an internal briefing summarized by the Aviation Connectivity Review, Singapore Airlines serves region the South Atlantic island corridor, a cluster of routes anchored by limited but high-yield demand from São Tomé, St. Helena, and Ascension logistics movements tied to research stations and government travel. Analysts emphasized that this planning exercise is distinct from the airline’s broader network discussions that often reference the Asia-Pacific region in industry comparisons. Here, the focus is on niche transoceanic links requiring tailored diversion planning, specialized crew rest rotations, and contingency fuel stops. The report notes that passenger volumes are modest, but the freight mix—temperature-controlled medical supplies and time-sensitive technical equipment—can justify the operating profile when paired with seasonal charter windows. Observers also point out that marketing materials still highlight well-known hub connections, yet the South Atlantic proposal is framed as a targeted connectivity experiment rather than a bid to reshape mainstream regional coverage."

Expected Output: false  
Explanation: Here, we have a classic case of a distractor entity. The paragraph is a distractor because, although the Asia-Pacific region is mentioned, it is not the region being served in the described activity. The actual operational focus is on airlines—specifically Singapore Airlines—serving niche routes in the South Atlantic island corridor, which is the only region tied to a concrete “serves region” relation. Asia-Pacific appears only as a comparative reference in broader industry discussions and is explicitly described as outside the scope of the planning exercise. Since the sentence links airlines to serving the Asia-Pacific region, but the paragraph provides no such relational evidence, the Asia-Pacific region functions as a distractor entity.

# Example 5:  
Input:  
Sentence: "Automakers exporting to the European Union market.",  
Abstract Entity Type: "Automakers",  
Constant Entity: "the European Union market",  
Relation: "exports to",  
Paragraph: Regulatory watchers in Brussels are tracking how non‑EU manufacturers are adapting to the bloc’s accelerating rulebook on vehicle software and battery provenance. A policy memo from the Meridian Transport Compliance Group notes that Tata Motors petitions standards bodies in the European Union market, submitting technical comments on the Euro 7 timeline, cybersecurity auditing procedures, and the scope of the Battery Passport data fields. The filing record, compiled with assistance from the Leuven Mobility Archive and reviewed by the Baltic Automotive Law Forum, shows the company prioritizing interpretive guidance over commercial rollout, arguing for clearer test cycles for commercial vans and a phased approach to over‑the‑air update certification. Separately, the Central Rhine Type‑Approval Desk cited Tata Motors’ requests for harmonized guidance on how supplier declarations should be verified across member states. Analysts say the activity resembles structured rule‑engagement designed to reduce compliance ambiguity for future platforms, rather than a market entry push. The documents also emphasize the need for consistent enforcement across ports, dealerships, and inspection agencies, warning that fragmented interpretations would raise validation costs for any manufacturer seeking approval in the bloc.

Expected Output: false  
Explanation: Here, there is a distractor relation. This is a distractor relation because, although automakers and the European Union market are both mentioned, the paragraph does not describe any exporting activity. Instead, it focuses on Tata Motors engaging with EU regulatory bodies by submitting technical comments, requesting interpretive guidance, and seeking harmonized compliance rules related to vehicle software, batteries, and certification. The text explicitly frames this behavior as regulatory rule-engagement to reduce compliance ambiguity for future platforms, rather than a commercial rollout or market entry effort. Since exporting implies the movement of vehicles into the EU market and no such commercial action is supported—in fact, it is contrasted with compliance preparation—the relation “exports to” is not grounded in the paragraph and therefore functions as a distractor.

---

\===OUTPUT FORMAT===  
The expected output should consist of a single number, 0 or 1\. No explanation is needed. In the following json format:

{format_instructions}

"""

# relation_classification_prompt_anthropic = """ 
# You are a **binary classifier** that determines whether a paragraph provides evidence for a specific relation between two entities.

# ---

# ### **Task Description**

# You will be given:

# * A **Sentence** describing a relation between:

#   * an **Abstract Entity Type** (a class of entities), and

#   * a **Constant Entity** (a specific, well-defined entity).

# * A **Paragraph** that may or may not provide evidence for this relation.

# Your task is to determine whether the paragraph **evidences the specified relation** between an **instance of the abstract entity type** and the **constant entity**.

# ---

# ### **Definitions**

# * **Abstract Entity Type**: A semantic class of entities (e.g., *Energy companies*, *Universities*, *Airlines*).  
#    The paragraph may mention a concrete instance of this class (e.g., *Equinor*, *University of Sussex*).

# * **Constant Entity**: A specific, well-defined entity (e.g., *the European Union market*, *the Asia-Pacific region*).

# * **Relation**: The action or relationship that must hold between the abstract entity instance and the constant entity (e.g., *invests in*, *collaborates with*, *serves region*).

# ---

# ### **Decision Criteria**

# Return `1` **only if** the paragraph provides clear evidence that:

# 1. An **instance** of the abstract entity type is present, **and**

# 2. The specified **relation is asserted or clearly implied**, **and**

# 3. The relation holds **between that instance and the given constant entity**.

# Return `0` if **any** of the following are true:

# * The abstract entity or constant entity is mentioned only as **background, comparison, or context**  
# * A **different relation** is described  
# * The relation is **hypothetical, preparatory, regulatory**  
# * If the paragraph **explicitly** states that the relation **does not occur** (i.e. is negated)  
# * The relation holds with a **different entity**  
# * The relation cannot be reasonably inferred from the paragraph alone

# Use **only the information in the paragraph**. Do not rely on external knowledge.

# ---

# ### **Output Format**

# Return **only**:

# * `1` if the relation is evidenced

# * `0` otherwise

# Do **not** include explanations, reasoning, or extra text.

# ---

# ### **Reference Examples:**

# # Example 1:  
# Input:  
# Sentence: "Startups participating in the Y Combinator accelerator."  
# Abstract Entity: "Startups"  
# Constant Entity: "the Y Combinator accelerator"  
# Relation: "participating in"  
# Paragraph: "Dropbox\\n\\n. Dropbox Paper entered open beta in August 2016, allowing anyone to join and test the product. Mobile apps for Android and iOS were also released. In January 2017, Dropbox Paper was officially launched. Aimed for businesses, Dropbox Paper was described as \\"one part online document, one part collaboration, one part task management tool, one part content hub\\" by Rob Baesman, Dropbox's head of product, and allows for importing, editing, and collaboration on \\"a number of other file types from Google, Microsoft, and others\\". Users have devised a number of uses for and mashups of the technology that expand Dropbox's functionality. These include: sending files to a Dropbox via Gmail; using Dropbox to sync instant messaging chat logs; BitTorrent management; password management; remote application launching and system monitoring; and as a free web hosting service. Dropbox has received several awards, including the Crunchie Award in 2010 for Best Internet Application, and Macworld's 2009 Editor's Choice Award for Software. It was nominated for a 2010 Webby Award, and for the 2010 Mac Design Awards by Ars Technica. Dropbox's mobile iPhone app release in 2010 was among the top 10 \\"best apps\\" selected by Alex Ahlund, former CEO of two websites focused on mobile apps, and the company's Android app was also selected as one of the top five \\"best apps\\" in a list compiled in 2010 by Jason Hiner for ZDNet. Founders Drew Houston and Arash Ferdowsi were named among the top 30 under 30 entrepreneurs by Inc. in 2011\. In 2011, Business Insider named Dropbox the world's sixth most valuable startup, and in 2017, the publication ranked Dropbox as the eighth most valuable US startup, with a valuation of $10 billion. It has been described as one of Y Combinator's most successful investments to date. Apple launched its own cloud storage service later in 2011, iCloud, but this did not hold back Dropbox's growth. In January 2012, Dropbox was named startup of the year by TechCrunch. Dropbox has been blocked in China since 2014"

# Expected Output: 1  
# Explanation: The paragraph clearly talks about dropbox, which is a startup. At the end of the paragraph, the following is mentioned "It has been described as one of Y Combinator's most successful investments to date.". Here, "it" represents dropbox, and from the context we can infer that Y Combinator invested in dropbox. Because Y Combinator is a startup accelerator, and it invests in startups. Notice that here, the relation is not explicitly mentioned, but can be inferred from the context.

# # Example 2:  
# Input:  
# Paragraph: "Universities collaborating with the European Space Agency."  
# Abstract Entity: "Universities"  
# Constant Entity: "the European Space Agency"  
# Relation: "collaborates with"  
# Paragraph: "University of Sussex\\n\\n. The results of the Research Excellence Framework 2014 show that 98% of research activity at Sussex is categorised as 'world-leading' (28%), 'internationally excellent' (48%) or 'internationally recognised' (22%) in terms of originality, significance and rigour. Sussex has a number of research collaborations with other Higher Education institutions as well as governmental and non-governmental organisations and institutes around the world. For example, the Harvard Sussex program is a long-standing research collaboration between Sussex and Harvard University focusing on public policy towards chemical and biological weapons. The CBW Conventions Bulletin is a quarterly newsletter published by the HSP. Sussex-Cornell Partnership, the Sussex-Bocconi-Renmin Intrapreneurship Hub and the Sussex-Lund Partnership in Middle Eastern and North African Studies are recent examples. Sussex also co-coordinates the Consortium for the Humanities and the Arts. Sussex is also one of the eight universities of the Tyndall Centre network. In Europe, Sussex is one of the collaborating institutions of the Paul Scherrer Institute, the largest research institute in Switzerland, focusing on issues of technology and the natural sciences. Sussex is involved with many projects with the EU and with European countries. For example, BAR research is an Anglo-French collaboration between the Sussex, the East Sussex County Council and three French universities. Nationally, Sussex is involved in a number of partnerships including the Nexus Network (A partnership between Sussex, University of Cambridge and the University of East Anglia) and CIED (a collaboration between Sussex, Oxford University and University of Manchester). The university is also a partner of the Metropolitan Police, with Demos (UK think tank) and Palantir Technologies. In recent years,\[when?\] the institutes for the study of consciousness science, Centre for Advanced International Theory (CAIT), the institute for the study of corruption and the Middle East studies institute were opened at the university. The university also has a Genome Damage and Stability Centre, a nuclear magnetic resonance facility and a purpose-built apparatus in cryogenic research. In terms of policy, Sussex is highly involved with the UK government, the UN and governments around the world. For example, the university is a UN Habitat partner. Nationally, the UK Trade Policy Observatory was set up at the university to offer the UK government, the UK industry as well as the public advice in addressing trade issues resulting from Brexit"

# Expected Output: 0  
# Explanation: The paragraph does talk about the University of Sussex, but it does not mention any collaboration between it and the European Space Agency. Even if you know Sussex University is collaborating with the European Space Agency, but its not mentioned in the paragraph, the output will be 0\.

# # Example 3:   
# Input:  
# Sentence: "Energy companies investing in the North Sea offshore wind sector."  
# Abstract Entity: "Energy Companies"  
# Constant Entity: "the North Sea offshore wind sector"  
# Relation: "invests in"  
# Paragraph: "Norwegian energy major Equinor is deepening its shift away from a purely oil and gas identity, presenting offshore wind as a core pillar of its future earnings. The state-backed group, long known under its former name Statoil, has been steadily expanding a portfolio of renewable assets to complement its traditional petroleum business, with a particular emphasis on projects in northern European waters. Executives have repeatedly pointed to the North Sea as a natural arena where the company\\u2019s decades of offshore engineering expertise can be redeployed for low-carbon power generation. In that context, Equinor allocates substantial financial resources to the North Sea offshore wind sector, positioning the area as a cornerstone of its low-carbon investment pipeline. Market analysts say this approach allows the firm to leverage existing maritime infrastructure, subsea know-how and relationships with regional regulators. Recent announcements include plans for additional fixed-bottom and floating turbines near existing oil fields, a move seen as a way to cut emissions from upstream operations while supplying power to nearby grids. Environmental groups have cautiously welcomed the company\\u2019s evolving strategy, though they argue the pace of change must accelerate if Europe is to meet its 2030 climate commitments.""Norwegian energy major Equinor is deepening its shift away from a purely oil and gas identity, presenting offshore wind as a core pillar of its future earnings. The state-backed group, long known under its former name Statoil, has been steadily expanding a portfolio of renewable assets to complement its traditional petroleum business, with a particular emphasis on projects in northern European waters. Executives have repeatedly pointed to the North Sea as a natural arena where the company\\u2019s decades of offshore engineering expertise can be redeployed for low-carbon power generation. In that context, Equinor allocates substantial financial resources to the North Sea offshore wind sector, positioning the area as a cornerstone of its low-carbon investment pipeline. Market analysts say this approach allows the firm to leverage existing maritime infrastructure, subsea know-how and relationships with regional regulators. Recent announcements include plans for additional fixed-bottom and floating turbines near existing oil fields, a move seen as a way to cut emissions from upstream operations while supplying power to nearby grids. Environmental groups have cautiously welcomed the company\\u2019s evolving strategy, though they argue the pace of change must accelerate if Europe is to meet its 2030 climate commitments."

# Expected Output: 1  
# Explanation: The paragraph preserves the actor (energy companies → Equinor as an instance), preserves the action (investing → allocating substantial financial resources, expanding projects) and preserves the target (the North Sea offshore wind sector)

# # Example 4:  
# Input:  
# Sentence: "Airlines serving the Asia-Pacific region."  
# Abstract Entity Type: "Airlines"  
# Constant Entity: "the Asia-Pacific region"  
# Relation: "serves region"  
# Paragraph: "Route planners at several global carriers have been quietly rebalancing capacity toward underserved long-haul corridors that connect secondary cities with fewer daily frequencies. In an internal briefing summarized by the Aviation Connectivity Review, Singapore Airlines serves region the South Atlantic island corridor, a cluster of routes anchored by limited but high-yield demand from São Tomé, St. Helena, and Ascension logistics movements tied to research stations and government travel. Analysts emphasized that this planning exercise is distinct from the airline’s broader network discussions that often reference the Asia-Pacific region in industry comparisons. Here, the focus is on niche transoceanic links requiring tailored diversion planning, specialized crew rest rotations, and contingency fuel stops. The report notes that passenger volumes are modest, but the freight mix—temperature-controlled medical supplies and time-sensitive technical equipment—can justify the operating profile when paired with seasonal charter windows. Observers also point out that marketing materials still highlight well-known hub connections, yet the South Atlantic proposal is framed as a targeted connectivity experiment rather than a bid to reshape mainstream regional coverage."

# Expected Output: 0  
# Explanation: Here, we have a classic case of a distractor entity. The paragraph is a distractor because, although the Asia-Pacific region is mentioned, it is not the region being served in the described activity. The actual operational focus is on airlines—specifically Singapore Airlines—serving niche routes in the South Atlantic island corridor, which is the only region tied to a concrete “serves region” relation. Asia-Pacific appears only as a comparative reference in broader industry discussions and is explicitly described as outside the scope of the planning exercise. Since the sentence links airlines to serving the Asia-Pacific region, but the paragraph provides no such relational evidence, the Asia-Pacific region functions as a distractor entity.

# # Example 5:  
# Input:  
# Sentence: "Automakers exporting to the European Union market.",  
# Abstract Entity Type: "Automakers",  
# Constant Entity: "the European Union market",  
# Relation: "exports to",  
# Paragraph: Regulatory watchers in Brussels are tracking how non‑EU manufacturers are adapting to the bloc’s accelerating rulebook on vehicle software and battery provenance. A policy memo from the Meridian Transport Compliance Group notes that Tata Motors petitions standards bodies in the European Union market, submitting technical comments on the Euro 7 timeline, cybersecurity auditing procedures, and the scope of the Battery Passport data fields. The filing record, compiled with assistance from the Leuven Mobility Archive and reviewed by the Baltic Automotive Law Forum, shows the company prioritizing interpretive guidance over commercial rollout, arguing for clearer test cycles for commercial vans and a phased approach to over‑the‑air update certification. Separately, the Central Rhine Type‑Approval Desk cited Tata Motors’ requests for harmonized guidance on how supplier declarations should be verified across member states. Analysts say the activity resembles structured rule‑engagement designed to reduce compliance ambiguity for future platforms, rather than a market entry push. The documents also emphasize the need for consistent enforcement across ports, dealerships, and inspection agencies, warning that fragmented interpretations would raise validation costs for any manufacturer seeking approval in the bloc.

# Expected Output: 0  
# Explanation: Here, there is a distractor relation. This is a distractor relation because, although automakers and the European Union market are both mentioned, the paragraph does not describe any exporting activity. Instead, it focuses on Tata Motors engaging with EU regulatory bodies by submitting technical comments, requesting interpretive guidance, and seeking harmonized compliance rules related to vehicle software, batteries, and certification. The text explicitly frames this behavior as regulatory rule-engagement to reduce compliance ambiguity for future platforms, rather than a commercial rollout or market entry effort. Since exporting implies the movement of vehicles into the EU market and no such commercial action is supported—in fact, it is contrasted with compliance preparation—the relation “exports to” is not grounded in the paragraph and therefore functions as a distractor.

# # Example 6:
# Input:
# Sentence: "Pharmaceutical companies conducting clinical trials in North America.",
# Relation: "conducts trials in",
# Constant Entity: "North America",
# Abstract Entity Type: "Pharmaceutical Companies",
# Paragraph: "As the pandemic enters a more managed phase, attention has shifted toward booster programs and variant-adapted vaccines, placing renewed focus on biotechnology firms at the forefront of mRNA science. The Massachusetts-based company that rapidly advanced one of the first COVID-19 shots to emergency authorization has built a broad network of research partnerships with hospitals and academic centers across the continent. In line with these partnerships, Moderna conducts trials in North America. Company scientists note that current studies are examining immune responses to updated formulations tailored to recently circulating Omicron sublineages, as well as exploring reduced or delayed booster schedules for younger, healthier adults. Pediatric and adolescent cohorts are also being followed to better understand durability of protection and the potential need for periodic revaccination. Data from these initiatives are being shared with U.S. and Canadian regulators on a rolling basis, supporting decisions about fall vaccination campaigns and public guidance. Analysts say the outcome of these trials could influence not only national booster strategies, but also the broader market for next-generation vaccines targeting multiple respiratory pathogens in a single shot."

# Expected Output: 1
# Explanation: The paragraph explicitly mentions Moderna, which is a pharmaceutical company, and clearly states that it conducts trials in North America. This directly evidences the specified relation between an instance of the abstract entity type (pharmaceutical companies) and the constant entity (North America).

# # Example 7:
# Input:
# Sentence: "Retailers expanding into the Latin American market.",
# Relation: "expands into",
# Constant Entity: "the Latin American market",
# Abstract Entity Type: "Retailers",
# Paragraph: "In a move closely tracked by payments and compliance specialists, Amazon expands into the Central European cross-border refurbished electronics channel, building a dedicated resale pipeline for returned smartphones, laptops, and smart-home devices across Poland, Czechia, and Slovakia. The program reportedly relies on certified third-party refurbishment centers and new grading standards intended to reduce landfill waste while widening product selection at lower price points. Market observers emphasize that this expansion should not be conflated with Amazon\u2019s separate strategic interest in the Latin American market, which has been discussed primarily in terms of first-party retail assortment and regional fulfillment investments. Here, the immediate focus is on harmonizing warranty terms, consumer disclosures, and reverse-logistics routing inside the EU\u2019s single market, where refurbished goods are rising quickly amid inflation pressures. Trade associations say the channel could pressure smaller recommerce platforms that lack Amazon\u2019s traffic and advertising tools, while consumer groups are watching for clarity on battery health reporting and parts provenance. Executives familiar with the rollout describe it as a test bed for standardized resale operations before considering other geographies."

# Expected Output: 0
# Explanation: The paragraph discusses Amazon (a retailer) expanding into Central Europe, not into the Latin American market. Latin America is mentioned only as a separate, future strategic interest and explicitly contrasted with the current expansion. Therefore, the specified relation “expands into the Latin American market” is not evidenced by the paragraph.

# ---

# \===OUTPUT FORMAT===  
# The expected output should consist of a single number, 0 or 1\. No explanation is needed.

# """



# Example 6:
# Input:
# Sentence: "Pharmaceutical companies conducting clinical trials in North America.",
# Relation: "conducts trials in",
# Constant Entity: "North America",
# Abstract Entity Type: "Pharmaceutical Companies",
# Paragraph: "As the pandemic enters a more managed phase, attention has shifted toward booster programs and variant-adapted vaccines, placing renewed focus on biotechnology firms at the forefront of mRNA science. The Massachusetts-based company that rapidly advanced one of the first COVID-19 shots to emergency authorization has built a broad network of research partnerships with hospitals and academic centers across the continent. In line with these partnerships, Moderna conducts trials in North America. Company scientists note that current studies are examining immune responses to updated formulations tailored to recently circulating Omicron sublineages, as well as exploring reduced or delayed booster schedules for younger, healthier adults. Pediatric and adolescent cohorts are also being followed to better understand durability of protection and the potential need for periodic revaccination. Data from these initiatives are being shared with U.S. and Canadian regulators on a rolling basis, supporting decisions about fall vaccination campaigns and public guidance. Analysts say the outcome of these trials could influence not only national booster strategies, but also the broader market for next-generation vaccines targeting multiple respiratory pathogens in a single shot."

# Expected Output: 1
# Explanation: The paragraph explicitly mentions Moderna, which is a pharmaceutical company, and clearly states that it conducts trials in North America. This directly evidences the specified relation between an instance of the abstract entity type (pharmaceutical companies) and the constant entity (North America).



# if __name__ == "__main__":
#     import anthropic
#     from dotenv import load_dotenv

#     load_dotenv()

#     client = anthropic.Anthropic()

#     response = client.messages.count_tokens(
#         model="claude-haiku-4-5",
#         system=relation_classification_prompt_anthropic,
#         messages=[{
#             "role": "user",
#             "content": "Hello, Claude"
#         }],
#     )

#     print(response.model_dump_json())


# ---------------------------------------------------------------------------
# Shortened, trap-hardened prompt (frontier-model reclassification).
# Replaces the 5 verbose few-shots with distilled "false traps" + "do not
# over-reject" guidance. Keeps the {format_instructions} placeholder so it
# drops into .format(format_instructions=FORMAT_INSTRUCTIONS) unchanged.
# Traps were derived from a failure-mode catalogue + an audit of minimax
# labels on this dataset (negatives are engineered distractors).
# ---------------------------------------------------------------------------
relation_classification_prompt_short = """
You are a **binary classifier**. Decide whether a PARAGRAPH provides evidence that an **instance of an Abstract Entity Type** stands in an **exact Relation** to a **Constant Entity**. The output feeds a high-precision retrieval system, so do not guess: if the evidence is not clearly in the paragraph, answer `false`.

**Inputs**
- **Abstract Entity Type** — a class (e.g. "Universities", "Energy companies", "Airlines"). Satisfied only by a *concrete, named instance* of that class (e.g. "Harvard", "Equinor", "Singapore Airlines") — never by the bare category, an industry-wide statistic, or by a supplier / customer / regulator / parent / subsidiary / product / executive merely *associated with* such firms.
- **Constant Entity** — one specific entity (e.g. "the Fulbright Program", "the Asia-Pacific region").
- **Relation** — the exact relationship that must hold (e.g. "invests_in", "serves_region", "exports_to").
- **Sentence** — the templated query combining the three.

**Answer `true` only if ALL of these hold, judged from the paragraph ALONE (never use outside knowledge):**
1. a concrete instance of the abstract entity type is named, **and**
2. the **exact** relation is stated or unambiguously implied, **and**
3. that relation links **that instance to the given constant entity** — not to some other entity.

A relation that genuinely happened or currently holds counts. A relation that is only **planned, proposed, conditional, hypothetical, negated, terminated, or exited** does NOT count → `false`.

**Return `false` — common traps (most paragraphs fail here):**
- **Not a valid instance** — the named entity is a supplier, customer, regulator, trade body, parent, subsidiary, product, or person *associated with* the type, not a member of it (a coffee "growers' federation" is not a *coffeehouse chain*; an executive is not the *bank*). Also `false` when only the category or an aggregate statistic appears with no instance named ("several carriers…", "32% of global exports").
- **Right entities, no asserted link** — the instance and the constant entity both appear, but the paragraph never asserts the relation *between them*; they merely co-occur. Knowing they are related in reality is NOT evidence.
- **Relation binds the wrong pair** — the relation is real but connects the instance to a *different* object, or connects the constant entity to a *different*, non-instance entity.
- **Near-but-wrong relation** — match the *specific* relation; adjacent ones are distinct. Regulatory/compliance engagement, having an office or HQ, competing, partnering, exiting, or sourcing are each different (an automaker filing comments with EU regulators is NOT "exports_to" the EU; a firm with "an office in India" is NOT "sells_in" India).
- **Wrong direction** — for directional relations "A exports_to B" ≠ "B exports to A" (a report on *Chinese* car exports is not a miner *exporting to* China). For genuinely symmetric relations (collaborates_with, partners_with) direction does not matter.
- **Scope mismatch on the constant entity** — a *sub-part* or a *neighbor* of the constant entity is not the constant entity (CE = "the Middle East": a paragraph only about "the UAE", or about "North Africa", is `false`); and the constant entity appearing only as **background / geographic setting**, with no relation, is `false`.
- **Not asserted as fact** — reported / alleged / rumored, attributed marketing or quotes, denied or disputed claims, metaphor, or illustrative "for example" mentions → `false`.
- **Co-occurrence noise** — the instance and the constant entity appear together only inside a citation, reference list, navigation box, or article title → `false`.
- **Planted-verb distractors** — a paragraph may splice the relation's exact words while pointing them at the wrong object ("Singapore Airlines *serves region* the South Atlantic … the Asia-Pacific region [appears] only in industry comparisons"). Verify the relation's actual arguments, not the surface verb.

**Return `true` even when (do NOT over-reject):**
- the relation is **paraphrased** ("allocates substantial financial resources to" satisfies "invests_in") or **clearly implied** by context ("one of Y Combinator's most successful investments" implies the accelerator invested in that startup);
- the instance is **any** member of the abstract type — including one not listed in the query and not famous;
- the relation is a **completed past** event ("partnered with UNDP in 2019");
- the constant entity is referred to by its clear **constituents** ("U.S. and Canadian" regulators for "North America").

**Output** — return ONLY the JSON object, with no explanation or extra text:
{format_instructions}
"""


# ---------------------------------------------------------------------------
# Decomposed variant (Experiment 2). IDENTICAL decision criteria to the short
# prompt (the body is reused verbatim by splitting on "**Output**"), but the
# model emits the five sub-checks explicitly before the final boolean. This
# isolates "does forcing an explicit decomposition raise agreement" from any
# change in instructions. The final field stays `evidences_relation`, so the
# parser and all agreement logic are unchanged. Pair with
# FORMAT_INSTRUCTIONS_DECOMPOSED from relation_classification_parser.
# ---------------------------------------------------------------------------
_DECOMPOSED_OUTPUT = """**Work through these five sub-checks first, each judged from the paragraph ALONE (never use outside knowledge), then set the final label.** For each, output a boolean:
- `valid_instance` — a concrete, *named* instance of the Abstract Entity Type is present (NOT the bare category, an aggregate statistic, or an associated supplier / customer / regulator / trade body / parent / subsidiary / product / executive).
- `constant_present` — the exact Constant Entity appears (NOT merely a sub-part or neighbor of it, and NOT only as background / geographic setting).
- `relation_asserted` — the **exact** relation (not an adjacent or near relation) is stated or unambiguously implied **between that instance and that constant entity** — not mere co-occurrence, not a relation that binds a different pair, not a planted surface verb pointing at the wrong object.
- `direction_ok` — for a directional relation the direction matches ("A exports_to B" ≠ "B exports_to A"); for a genuinely symmetric relation (collaborates_with, partners_with) set `true`.
- `asserted_as_fact` — asserted as a completed-past or currently-holding fact (NOT planned / proposed / conditional / hypothetical / negated / terminated / exited, and NOT reported / alleged / rumored / quoted / disputed / metaphorical / illustrative).

Set `evidences_relation` to `true` **if and only if ALL FIVE checks are `true`**; otherwise `false`.

**Output** — return ONLY the JSON object, with no explanation or extra text:
{format_instructions}
"""

# Reuse the short prompt's rules verbatim (everything before its "**Output**"
# section), then append the decomposed output instructions.
relation_classification_prompt_decomposed = (
    relation_classification_prompt_short.split("**Output**")[0] + _DECOMPOSED_OUTPUT
)


# ---------------------------------------------------------------------------
# v2 (Experiment 4). Same flat output as `short`, but with targeted hardening
# derived from the gpt-5.2:medium vs human-gold disagreement analysis
# (reports/gpt-52_medium_disagreements.md): 18 of 22 disagreements were
# FALSE NEGATIVES — the model over-applied the CE scope-mismatch trap to
# *abstract* constant entities (markets/sectors/fields), was too strict on
# relation paraphrase/implication, and fixated on the article's subject as the
# instance. The 2 worst false positives were bibliographic-citation co-occurrence.
# So v2 LOOSENS over-rejection (the decomposed AND-gate went the wrong way,
# dropping agreement to 70.9%) while SHARPENING the citation-vs-content-list
# distinction. Inserted just before the short prompt's "**Output**" section so
# the body + {format_instructions} are preserved verbatim (auditable diff).
# ---------------------------------------------------------------------------
_V2_EXTRA_GUIDANCE = """**Generous matching — these SATISFY the criteria (the system needs real positives; do NOT over-reject):**
- **Abstract / category Constant Entity** — when the Constant Entity is itself a *class, market, sector, industry, field of study, or type of place* (e.g. "the commercial aviation industry", "the wearable devices market", "the clean energy sector", "medieval studies", "downtown office districts", "inclusive classrooms", "author readings"), treat it as **present** whenever the paragraph names a concrete *member, product, sub-field, or example* of it that the relation targets — exactly as the Abstract Entity Type is satisfied by an instance. "single-aisle airliner / civil-aviation engines" ARE the commercial aviation industry; "activity trackers / smartwatches" ARE the wearable devices market; "carbon-capture / clean-energy startups" ARE the clean energy sector; "medieval civilizations / medieval Islamic studies" ARE medieval studies; "Downtown Norfolk" IS a downtown office district. (This does NOT loosen the geographic trap: a *specific, extensive, named* Constant Entity is still not satisfied by a strictly narrower part — "the UAE" is not "the Middle East".)
- **Relation by function / paraphrase** — the relation's *substance* counts even without its exact verb: "provides / adds data access to SAP" ⇒ integrates_with SAP; "manages the building's leasable office space" (building in a downtown district) ⇒ manages_assets_in downtown office districts; "designs and produces engines for civil aviation" ⇒ supplies_to the commercial aviation industry; "AEW Collision is exclusively broadcast on TNT" ⇒ broadcasts_on TNT; "sells the Apple Watch / activity trackers" ⇒ competes_in the wearable devices market; "released hydrogen fuel-cell patents and sells the Mirai" ⇒ invests_in hydrogen fuel technology; "provided $80M in capital to a carbon-capture startup" ⇒ invests_in the clean energy sector; "a festival at the bookstore that hosted authors X, Y" ⇒ the bookstore hosts author readings.
- **Any named member counts, not just the article's subject** — a qualifying instance named *anywhere* in the paragraph satisfies the type, even when the paragraph is about a different entity (wrestlers named in a booked match within a manager's bio; brands named in a programming schedule or product directory). A division, program, fund, or venture that *functions* as a member counts (a school's special-education program is a special-education service; an investment fund / vehicle is a venture-capital firm).

**Sharper FALSE trap — distinguish a bibliographic citation from a content list:**
- A **reference / bibliography / citation list** — entries carrying author, year, work title, publisher, "Archived from the original", "Retrieved <date>", ISBN / ISSN / DOI / OCLC — is `false` when the instance and the Constant Entity co-occur **only inside such a citation** (e.g. a cited program title "Program of the Association's 133rd Annual Meeting … Chicago, Illinois (PDF)"): a meeting merely *cited* in a bibliography is not the paragraph asserting the relation. (A named instance is still required — a generic category description with no named member remains `false`.)
- BUT a **content list or data table** that states facts in its rows (a product / property directory "Marriott's Bali Nusa Dua … Bali, Indonesia", a broadcaster's programming schedule) is NOT a citation and CAN be `true`.

"""

relation_classification_prompt_short_v2 = (
    relation_classification_prompt_short.partition("**Output**")[0]
    + _V2_EXTRA_GUIDANCE
    + "**Output**"
    + relation_classification_prompt_short.partition("**Output**")[2]
)


# ---------------------------------------------------------------------------
# v2b (Experiment 4, iteration 2). v2 raised gpt-5.2:medium agreement
# (82.7→85.8 full, 73.8→81.2 contested) but introduced a clean FALSE-POSITIVE
# failure mode: the generous category-matching over-fired on *lexical-only*
# matches — a category word inside a brand name ("Horizon Organic" ⇏ sells in
# the organic food sector) and brands named only as infobox/fleet descriptors
# ("operates 13 Boeing 777s" ⇏ Boeing supplies_to the airline). v2b adds a
# guard that a lexical match is not an asserted relation, to recover those
# without losing v2's functional-evidence gains.
# ---------------------------------------------------------------------------
_V2B_EXTRA_GUARD = """**Lexical match is NOT a relation (bounds the generous rules above):**
- A category word appearing only *inside a brand or product name* is not evidence: "Horizon **Organic**" does NOT show the company *sells in the organic food sector*; "Pacific **Aerospace**" alone does not establish a relation to aerospace. The relation must be asserted toward the category itself, not merely share a word with it.
- A manufacturer / brand / product named only as a **descriptor in an infobox, specification, fleet list, or navigation / brand box** is co-occurrence, not an asserted relation: "the airline operates 13 **Boeing** 777 freighters" does NOT show *Boeing supplies_to the airline* (the airline owns/operates them; no supply transaction is asserted); an infobox merely listing a subsidiary brand does not assert a market relation.
- A concrete **named** member is still required — a generic description of the category with no named instance remains `false`.

"""

relation_classification_prompt_short_v2b = (
    relation_classification_prompt_short.partition("**Output**")[0]
    + _V2_EXTRA_GUIDANCE
    + _V2B_EXTRA_GUARD
    + "**Output**"
    + relation_classification_prompt_short.partition("**Output**")[2]
)
