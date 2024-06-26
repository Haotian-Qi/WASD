# Wilder Active Speaker Detection (WASD) Dataset ([Paper](https://arxiv.org/pdf/2303.05321.pdf))

*To view further details about WASD go to [dataset website](https://tiago-roxo.github.io/WASD/)*

## Metadata

The [metadata file](https://github.com/Tiago-Roxo/WASD/blob/main/eval/wasd_metadata.csv) contains all the features described in **Section D** of [WASD Supplementary Materials](https://drive.google.com/file/d/1rxBelPZDB_aQ9Et2PAQqBNG-3Y0_i6Hf/view?usp=drive_link) for each video of WASD dataset. 

The categories are represented by numbers from 0 to 4, refering to *Optimal Conditions*, *Speech Impairment*, *Facial Occlusion*, *Human Voice Noise*, and *Surveillance Settings*, respectively. The following table relates the numerical values with the considered possibilities for certain features in the metadata file:

| Feature                         | Possibilities                            |
|:--------------------------------|:----------------------------------------:|
| Facial Occlusion                | No (0) or Yes (1)                        |
| Human Voice as Background Noise | No (0) or Yes (1)                        |
| Speech Overlap                  | None-Low (0) or Medium-High (1)          |
| Delayed Speech                  | No (0) or Yes (1)                        |
| Surveillance Settings           | No (0) or Yes (1)                        |
| Proximity to Camera             | Low (0) or Medium (1) or High (2)        |
| Audio Quality                   | Low (0) or High (1)                      |
| Face Availability               | Non-Guaranteed (0.0) or Guaranteed (1.0) |

## Columbia

The results for [Columbia dataset](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_18) were obtained using the evaluation tool available at [Light-ASD](https://github.com/Junhua-Liao/Light-ASD).

## Cite

```bibtex
@article{roxo2023wasd,
    title={WASD: A Wilder Active Speaker Detection Dataset},
    author={Roxo, Tiago and Costa, Joana C and In{\'a}cio, Pedro RM and Proen{\c{c}}a, Hugo},
    journal={arXiv preprint arXiv:2303.05321},
    year={2023}
}
```
