
<h1 align="center">High-throughput fabrication of patient-specific vessel-on-chips for thrombosis prediction
</h1>
<p align="center">
<a href="[https://www.biorxiv.org/content/10.64898/2026.03.03.709446v1](https://www.biorxiv.org/content/10.64898/2026.03.03.709446v1)"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<h4 align="center">This is the official repository of the paper <a href="https://www.biorxiv.org/content/10.64898/2026.03.03.709446v1">High-throughput fabrication of patient-specific vessel-on-chips for thrombosis prediction</a>.</h4>
<h5 align="center"><em>Zihao Wang*, Yunduo Charles Zhao*, Haimei Zhao*, Arian Nasser, Nicole Alexis Yap, Yanyan Liu, Allan Sun, Wei Chen, Timothy Ang, Ken S Butcher, Lining Arnold Ju†
</em></h5>
<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#method">Method</a> |
  <a href="#results">Results</a> |
  <a href="#environment">Environment</a> |
  <a href="#code">Code</a> |
  <a href="#statement">Statement</a>
</p>

## News
- **(2026/03/05)** CLoT is released on [BioArXiv](https://www.biorxiv.org/content/10.64898/2026.03.03.709446v1).

## Abstract

The organ-on-a-chip field promises human-relevant disease modeling to replace animal testing, yet scale-up is limited by labor-intensive fabrication. Meanwhile, artificial intelligence (AI) has emerged as a transformative tool for disease prediction and drug screening, but its integration with organ-on-chips has been hindered by insufficient experimental datasets. Here, we introduce an automated fabrication platform capable of producing patient-specific vessel-on-chips at unprecedented throughput—80 fully functional chips within 20 hours of clinical image acquisition (10 hours fabrication + 10 hours biofunctionalization). Human blood-perfusion assays across these individualized chips create a high-fidelity “physical twin” library of thrombosis dynamics encompassing diverse anatomies, vascular injuries, and pharmacological interventions. Leveraging this dataset, we trained a generative AI “digital twin” (Cascade Learner of Thrombosis; CLoT) by fine-tuning a pretrained diffusion transformer via low-rank adaptation (LoRA) to synthesize realistic thrombosis videos. Compared with leading commercial video-generation models (Sora, Wan, Hunyuan, Kling, Seedance, Hailuo), CLoT achieves 5.3-fold higher vascular consistency and 7.38-fold greater thrombosis similarity. Prospective validation on unseen patient geometries and drug combinations yields >90% spatiotemporal agreement with experimental ground truth, establishing proof-of-concept for AI-driven personalized medicine while eliminating animal testing. This paradigm—automated “physical twin” production coupled with generative AI “digital twin”—represents a transformative approach for personalized thrombosis prediction and therapeutics.The organ-on-a-chip field promises human-relevant disease modeling to replace animal testing, yet scale-up is limited by labor-intensive fabrication. Meanwhile, artificial intelligence (AI) has emerged as a transformative tool for disease prediction and drug screening, but its integration with organ-on-chips has been hindered by insufficient experimental datasets. Here, we introduce an automated fabrication platform capable of producing patient-specific vessel-on-chips at unprecedented throughput—80 fully functional chips within 20 hours of clinical image acquisition (10 hours fabrication + 10 hours biofunctionalization). Human blood-perfusion assays across these individualized chips create a high-fidelity “physical twin” library of thrombosis dynamics encompassing diverse anatomies, vascular injuries, and pharmacological interventions. Leveraging this dataset, we trained a generative AI “digital twin” (Cascade Learner of Thrombosis; CLoT) by fine-tuning a pretrained diffusion transformer via low-rank adaptation (LoRA) to synthesize realistic thrombosis videos. Compared with leading commercial video-generation models (Sora, Wan, Hunyuan, Kling, Seedance, Hailuo), CLoT achieves 5.3-fold higher vascular consistency and 7.38-fold greater thrombosis similarity. Prospective validation on unseen patient geometries and drug combinations yields >90% spatiotemporal agreement with experimental ground truth, establishing proof-of-concept for AI-driven personalized medicine while eliminating animal testing. This paradigm—automated “physical twin” production coupled with generative AI “digital twin”—represents a transformative approach for personalized thrombosis prediction and therapeutics.



## Statement
@article {Wang2026.03.03.709446,
	author = {Wang, Zihao and Zhao, Yunduo Charles and Zhao, Haimei and Nasser, Arian and Yap, Nicole Alexis and Liu, Yanyan and Sun, Allan and Chen, Wei and Butcher, Ken S and Ang, Timothy and Ju, Lining Arnold},
	title = {Automated high-throughput fabrication of patient-specific vessel-on-chips enables a generative AI digital twin{\textemdash}Cascade Learner of Thrombosis (CLoT) for personalized thrombosis prediction},
	elocation-id = {2026.03.03.709446},
	year = {2026},
	doi = {10.64898/2026.03.03.709446},
	URL = {https://www.biorxiv.org/content/early/2026/03/05/2026.03.03.709446},
	eprint = {https://www.biorxiv.org/content/early/2026/03/05/2026.03.03.709446.full.pdf},
	journal = {bioRxiv}
}

