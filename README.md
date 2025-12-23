<h1 align="center">Interact2Ar: Full-Body Human-Human Interaction Generation via Autoregressive Diffusion Models</h1>

  <p align="center">
    <a href="https://www.pabloruizponce.com/papers/Interact2Ar"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2512.19692"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.19692-b31b1b.svg"></a> 
  </p>
  
<br>

## 🔎 About
<div style="text-align: center;">
    <img src="https://pycelle.com/papers/Interact2Ar/cover.png" align="center" width=100% >
</div>
</br>
Generating realistic human-human interactions is a challenging task that requires not only high-quality individual body and hand motions, but also coherent coordination among all interactants. Due to limitations in available data and increased learning complexity, previous methods tend to ignore hand motions, limiting the realism and expressivity of the interactions. Additionally, current diffusion-based approaches generate entire motion sequences simultaneously, limiting their ability to capture the reactive and adaptive nature of human interactions. To address these limitations, we introduce Interact2Ar, the first end-to-end text-conditioned autoregressive diffusion model for generating full-body, human-human interactions. Interact2Ar incorporates detailed hand kinematics through dedicated parallel branches, enabling high-fidelity full-body generation. Furthermore, we introduce an autoregressive pipeline coupled with a novel memory technique that facilitates adaptation to the inherent variability of human interactions using efficient large context windows. The adaptability of our model enables a series of downstream applications, including temporal motion composition, real-time adaptation to disturbances, and extension beyond dyadic to multi-person scenarios. To validate the generated motions, we introduce a set of robust evaluators and extended metrics designed specifically for assessing full-body interactions. Through quantitative and qualitative experiments, we demonstrate the state-of-the-art performance of Interact2Ar.

## 📌 News
- Our paper is available on [arXiv](https://arxiv.org/abs/2512.19692)

## 📝 TODO List
- [ ] Release code
- [ ] Release model weights
- [ ] Release evaluators weights
- [ ] Release visualization code.

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@misc{ruizponce2025interact2arfullbodyhumanhumaninteraction,
      title={Interact2Ar: Full-Body Human-Human Interaction Generation via Autoregressive Diffusion Models}, 
      author={Pablo Ruiz-Ponce and Sergio Escalera and José García-Rodríguez and Jiankang Deng and Rolandos Alexandros Potamias},
      year={2025},
      eprint={2512.19692},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.19692}, 
}
```
