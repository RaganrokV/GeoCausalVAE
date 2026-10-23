# Spatially aware generative AI for urban transport scenario exploration

GeoCausalVAE integrates **causal representation learning, empirical structural priors, and spatial regularization** within a variational autoencoder to explore how hypothetical changes in urban form relate to electric-vehicle energy use. It connects generative modeling with spatial analysis to support interpretable, geographically explicit counterfactual scenario exploration.

## Framework Overview

<img width="631" height="270" alt="image" src="https://github.com/user-attachments/assets/344ff933-ab32-47c2-abd6-cbd27a08819d" />

GeoCausalVAE combines three key components:

- **Spatial econometrics** to capture spatial spillover effects between urban areas  
- **Causal inference** to model structural relationships between urban variables and energy use  
- **Generative modeling** to enable counterfactual scenario generation under hypothetical interventions

## Study at a glance

The Shanghai case study combines **more than 30 million GPS records from 1,000 electric vehicles** (encompassing sedans, buses, and logistics vehicles) with urban morphology and energy-consumption measurements.  
<img width="607" height="416" alt="image" src="https://github.com/user-attachments/assets/7e40d735-4b97-46a4-80a6-fd2fc7be4796" />

## From observations to scenarios
Analysis is conducted at the jiedao level—an administrative subdistrict containing multiple streets and communities.
<img width="652" height="581" alt="image" src="https://github.com/user-attachments/assets/aefcdd97-93aa-4d26-af73-bbb16bd15da6" />

1. **Characterize urban context:** Aggregate fleet observations and urban morphology at the subdistrict level; examine spatial dependence and local clusters.
2. **Learn structured representations:** Combine generative reconstruction with empirical structural guidance and spatial regularization.
3. **Explore hypothetical changes:** Compare simulated energy-use responses across urban-form scenarios, locations, and vehicle categories.


