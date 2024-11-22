

# ISTR: A Curve-Centric Dataset for Industrial Survey Table Recognition


<p align="center">
  <h3 align="center">"The ISTR dataset is available for download at the following [Quark](link:https://pan.quark.cn/s/cc8be9495958)"</h3>
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

## 🔧 Setup
```
conda create -n ISTR python=3.9.2
conda activate ISTR
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

## 💫 Train
```
sh SD_finetune.sh
```

## 🌟 Acknowledgments
This project was built upon the work of Linaqruf, whose [kohya-trainer](https://github.com/Linaqruf/kohya-trainer) served as a valuable resource for the development of our methods. We sincerely appreciate their contribution to the community.

## 💡 Disclaimer
We developed this repository for **research purposes only**, and it is strictly limited to personal, academic, or non-commercial use.



