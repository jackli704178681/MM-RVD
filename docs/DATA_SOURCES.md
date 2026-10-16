# Data Sources

Raw neural datasets are not redistributed in this repository.

The Allen Institute publishes [terms of use](https://alleninstitute.org/terms-of-use/) and a [citation policy](https://www.allencell.org/citation-policy.html). CRCNS [terms of use](https://crcns.org/terms) prohibit public redistribution of downloaded data except to specified direct collaborators or students; obtain pvc-11 from CRCNS instead of mirroring it here. Dryad describes its datasets as [CC0](https://datadryad.org/help/guides/reuse), but the RS source file is also left with the original provider to avoid duplicating a large binary in this code repository. Follow each provider's current terms when acquiring data.

## Allen Visual Behavior Neuropixels
This study used 14 Allen Visual Behavior Neuropixels sessions:

1069458330, 1072567062, 1104058216, 1104289498, 1105798776, 1108531612, 1044016459, 1044385384, 1044389060, 1044594870, 1049514117, 1051155866, 1053718935, and 1056495334.

Official access is through the AllenSDK `VisualBehaviorNeuropixelsProjectCache` and Allen Institute data resources:

- AllenSDK documentation: https://alleninstitute.github.io/AllenSDK/
- Visual Behavior Neuropixels data book: https://allenswdb.github.io/physiology/ephys/visual-behavior/VBN-SessionData.html
- Allen Brain Observatory Open Data registry: https://registry.opendata.aws/allen-brain-observatory/

## CRCNS pvc-11
This study used three CRCNS pvc-11 sessions:

`data_monkey1_gratings`, `data_monkey2_gratings`, and `data_monkey3_gratings`.

Official access:

- CRCNS pvc-11 landing page: https://crcns.org/data-sets/vc/pvc-11/about/
- CRCNS/NERSC download endpoint: https://portal.nersc.gov/project/crcns/download/pvc-11

A CRCNS account may be required by the provider. Use the session manifests in this repository to reproduce the exact analysis scope.

## RS Center-Out

The current three-dataset study also uses the `RS_CenterOut_data_10msBin_with500buffer.pkl` file from the Dryad dataset [Dynamic structure of motor cortical neuron co-activity carries behaviorally relevant information](https://doi.org/10.5061/dryad.9p8cz8wm6). It is one macaque motor-cortex recording with eight reach directions. The current analysis uses 391 classification trials, 142 sorted units, and movement-onset-aligned 0-0.5 s responses. Five fixed trial partitions are sensitivity repeats, not five additional animals.

The provider file is not redistributed here. Channel-to-unit mapping supports random occupied-channel loss (RC); reliable physical electrode geometry for an adjacent-channel (AC) test is not established in this study. Aggregate-only RS results are in `source_data/current_20261016/`. The earlier 17-session manifests in this repository do not include RS.
