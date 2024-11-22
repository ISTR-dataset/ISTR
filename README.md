

# ISTR: A Curve-Centric Dataset for Industrial Survey Table Recognition


<p align="center">
  <h3 align="center">"The ISTR dataset is available for download at the following link:"</h3>
</p>

## 📖 Introduction
ISTR (Industrial Survey Table Recognition) is the first dataset focused on curve elements in industrial survey tables, aiming to advance research on automatic interpretation in the industrial mapping field. The dataset is derived from 107 well log charts, spanning multiple regions and depths, and includes 10,453 high-quality annotated images. ISTR supports image-to-image translation tasks, particularly for transforming noisy images into clean, structured table-like outputs.
![image](https://github.com/ISTR-dataset/ISTR/blob/main/image/all.png)
![image](https://github.com/ISTR-dataset/ISTR/blob/main/image/detail.png)

## 🔑 Key Features
**（1）Domain-specific Focus**: Specially designed for industrial survey tables with real-world, complex well log curve images.  
**（2）Rich Visual Features**: Includes annotations for blue solid lines, red dashed lines, and green curves, covering scenarios like logarithmic scales, high-frequency oscillations, and overlapping curves.  
**（3）Beyond Traditional Datasets**: Unlike traditional table recognition datasets (e.g., TableBank, PubTabNet), ISTR is tailored to handle curve-specific visual features critical for industrial applications.  

## 🎯 Applications
**（1）Industrial Mapping Automation**: Parsing and cleaning well log curve images.  
**（2）Image-to-Image Translation**: Research on transforming noisy inputs into clean outputs.  
**（3）Expanding Table Recognition**: Extending table recognition to non-text, structured data.  

##  Setup
```
conda create -n ISTR python=3.9.2
conda activate ISTR
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

## Train
```
sh SD_finetune.sh
```

### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders)

<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian

## Disclaimer
We developed this repository for **research purposes only**, and it is strictly limited to personal, academic, or non-commercial use.



