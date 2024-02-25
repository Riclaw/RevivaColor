# RevivaColor
## Summary
The Image Colorization Project aims to bring old memories to life by colorizing monochromatic images, with a focus on photographs containing human subjects. Traditional colorization tools struggle with accurately representing skin tones and historical imagery, making them less appealing and authentic. By employing semi/self-supervised learning techniques, this project seeks to enhance the colorization quality of such images. This endeavor holds significance in various domains, including family history preservation, historical research, entertainment, and public safety. Through this project, we aim to provide a solution that restores authenticity and enhances the visual appeal of old photographs and historical footage, enriching our understanding and appreciation of the past.

## Proposed Solutions 
The project employs autoencoders and a conditional Generative Adversarial Network (cGAN) as the main solutions for colorizing monochromatic images, with a focus on images containing human subjects.

Autoencoders: These neural networks are utilized to learn efficient representations of input data without labels. The autoencoder comprises an Encoder and a Decoder. The Encoder projects the image into a lower-dimensional latent representation, while the Decoder translates this representation back into an image. The autoencoder is trained to minimize reconstruction errors, evaluated using Mean Absolute Error, Mean Squared Error, Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index Metrics (SSIM). Autoencoders are well-suited for deconstructing and reconstructing images based on important features, aligning with the project's goal of colorizing human subjects in images.

Conditional Generative Adversarial Network (cGAN): A cGAN is developed to improve performance. In a cGAN, the Generator learns to generate realistic images conditioned on a black-and-white canvas, while the Discriminator learns to distinguish between synthetic and real images. This architecture, inspired by pix2pix, aims to generate colored images directly, maintaining greater fidelity with the target image compared to autoencoders.

These solutions are trained, validated, and tested using a dataset of 4863 colored images containing human portraits. The results are evaluated both mathematically, through comparison of various loss metrics, and visually, by plotting test images against ground truth images.



## Results and Conclusions
The results obtained from both architectures, Autoencoders and cGAN, were deemed satisfactory. They exhibited good colorization performance, particularly evident when applied to real historical black-and-white images, which yielded visually pleasant outcomes. However, it is acknowledged that these results are not yet optimal, likely constrained by time and computational power limitations.

A deeper analysis reveals that Autoencoders slightly outperformed the cGAN across all metrics used for evaluation, including PSNR and SSIM. This performance difference could be attributed to the cGAN's need for more input images to reach its full potential, as well as its training methodology, which primarily focused on a mix of L1 and Adversarial Losses rather than specifically optimizing for PSNR or SSIM.

Despite the slightly better performance of Autoencoders according to metrics, both architectures provided satisfactory visual results, representing an improvement over existing freely available colorization tools.

Limitations and the Way Forward
While this project represents a promising initial approach to enhancing colorization techniques for human portraits, several limitations were identified, primarily related to hardware constraints. The project was restricted by GPU and RAM limitations, preventing the use of larger datasets or higher resolution images beyond 256x256 pixels.

### Moving forward, potential avenues for improvement include:

1) Utilizing more powerful computational tools: Overcoming hardware limitations to enable training with larger datasets and higher resolution images.
2) Further studies and optimization of architectures: Exploring alternative architectures such as Stable Diffusion and optimizing the cGAN architecture to improve performance.
3) Training cGAN based on SSIM or PSNR: Investigating whether training the cGAN with loss functions based on SSIM or PSNR instead of L1 loss could enhance performance.

## Real Result Example
Here is an example of a real historical black-and-white image that has been colorized using the models developed in this project:


## Repository stucture
```
│
├── 📝 RevivaColor.ipynb: implementation of the autoencoders
│
├── 📝 cGANs_implementation.ipynb: implementation of the pix2pix paper with some variations
│
├── 📄 utils.py: common functions used across the notebooks
│
├── 📁 models: folder with the different autoencoders and the cGAN
│
└── 📄 RevivaColor_Report.pdf: full report
```

## General requirements

```
tensorflow==2.14.0
scipy==1.11.3
matplotlib==3.7.1
numpy==1.23.5
scikit_learn==1.2.2
opencv-python==4.8.0
```

## Credits

| Author             | Contact                       
| ----------------   | ------------------------------
| Edoardo Morresi | edoardo.morresi@studbocconi.it 
| Lorenzo d'Imporzano |	lorenzo.dimporzano@studbocconi.it
| Riccardo Valdo  | riccardo.valdo@studbocconi.it  
| Benedikt Korbach | benedikt.korbach@studbocconi.it
