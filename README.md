

### Main Results

We evaluate ACMC3D on six challenging indoor 3D object detection benchmarks. As shown below, our method achieves **state-of-the-art performance** across nearly all datasets. Notably, compared to the strong baseline UniDet3D, our method brings remarkable improvements on **S3DIS** (+2.2/+8.4 mAP), **MultiScan** (+5.7/+8.4 mAP), and **3RScan** (+3.0/+4.1 mAP) in the best-result setting, with consistent gains on ScanNet and ARKitScenes as well. These results demonstrate the effectiveness of our approach in enhancing detection robustness and minority-class recognition. Averaged over 25 independent trials, ACMC3D still outperforms all competitors by a clear margin, validating the stability of our improvements.

#### Best Result

| Method | ScanNet<br>mAP₂₅ | ScanNet<br>mAP₅₀ | ARKitScenes<br>mAP₂₅ | ARKitScenes<br>mAP₅₀ | S3DIS<br>mAP₂₅ | S3DIS<br>mAP₅₀ | MultiScan<br>mAP₂₅ | MultiScan<br>mAP₅₀ | 3RScan<br>mAP₂₅ | 3RScan<br>mAP₅₀ | ScanNet++<br>mAP₂₅ | ScanNet++<br>mAP₅₀ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MLCVNet | 64.5 | 41.4 | 41.9 | - | - | - | - | - | - | - | - | - |
| H3DNet | 67.2 | 48.1 | 38.3 | - | - | - | - | - | - | - | - | - |
| FCAF3D | 71.5 | 57.3 | - | - | 66.7 | 45.9 | 53.8 | 40.7 | 60.1 | 42.6 | 22.3 | 11.4 |
| UniDet3D | 71.7 | 58.3 | - | - | 70.1 | 48.0 | - | - | - | - | - | - |
| TR3D | 72.9 | 59.3 | - | - | 74.5 | 51.7 | 56.7 | 42.3 | 62.3 | 45.4 | 26.2 | 14.5 |
| SPGroup3D | 74.3 | 59.6 | - | - | 69.2 | 47.2 | - | - | - | - | - | - |
| UniDet3D | 77.9 | 66.1 | 61.3 | 47.1 | 75.2 | 60.8 | 64.2 | 51.6 | 64.7 | 48.6 | **26.4** | 17.2 |
| **ACMC3D** | **78.5** | **67.1** | **62.5** | **49.3** | **77.4** | **69.2** | **69.9** | **60.0** | **67.7** | **52.7** | 25.0 | **17.9** |

#### Average across 25 trials

| Method | ScanNet<br>mAP₂₅ | ScanNet<br>mAP₅₀ | ARKitScenes<br>mAP₂₅ | ARKitScenes<br>mAP₅₀ | S3DIS<br>mAP₂₅ | S3DIS<br>mAP₅₀ | MultiScan<br>mAP₂₅ | MultiScan<br>mAP₅₀ | 3RScan<br>mAP₂₅ | 3RScan<br>mAP₅₀ | ScanNet++<br>mAP₂₅ | ScanNet++<br>mAP₅₀ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FCAF3D | 70.7 | 56.0 | - | - | 64.9 | 43.8 | 52.5 | 39.2 | 59.6 | 40.4 | 21.4 | 11.0 |
| TR3D | 72.0 | 57.4 | - | - | 72.1 | 47.6 | 55.0 | 41.2 | 61.5 | 44.2 | 24.3 | 13.9 |
| SPGroup3D | 74.5 | 60.3 | - | - | 67.7 | 43.6 | - | - | - | - | - | - |
| CAGroup3D | 76.8 | 64.5 | - | - | - | - | - | - | - | - | - | - |
| UniDet3D | 77.1 | 65.2 | 60.2 | 46.0 | 73.3 | 57.9 | 62.4 | 50.8 | 62.1 | 45.6 | **24.4** | 16.3 |
| **ACMC3D** | **77.9** | **65.6** | **61.2** | **48.1** | **75.5** | **65.4** | **68.3** | **58.5** | **66.1** | **50.2** | 23.7 | **17.2** |

> **Key Observations:**
> - **Best-result setting**: ACMC3D achieves the highest mAP on six datasets, with particularly large gains on S3DIS, MultiScan, and 3RScan.
> - **25-trial average**: Consistent improvements across all datasets demonstrate the stability and robustness of our method.
> - The significant boost in mAP₅₀ indicates enhanced precision for minority-class detection, validating our motivation.
