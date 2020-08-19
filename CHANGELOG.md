# Change Log
All notable changes to this project will be documented in this file. Project first added to GitHub on
2020-02-20. 

## unreleased

## [0.1.2] - 2020-08-19
### Fixed
- Bug fix for `connectivity_features`; was: if channel labels list is empty
then would assume (incorrectly) only 2 channels and would use index [1,2] for left, right
hemisphere. Now: if no channel labels throw warning and returns nans, 
if there are less than 2 channels it returns nans and
if there isn't at least a left and right hemisphere channel it returns nan
  

## [0.1.1] - 2020-06-26
### Added
- Added option to pass in parameters so NEURAL_parameters.py does not need to be altered
### Fixed
- Fixed bug when user wants to generate 1 frequency band instead of multiple


### Changed 
### Removed
### Fixed
### Added
- Added entire NEURAL_py_EEG_Features repository 
- Using [semantic versioning](http://semver.org/) 
## [0.1.0] - 2020-02-20



[0.1.0]: https://github.com/BrianMur92/NEURAL_py_EEG_feature_set/releases/tag/0.1.0
[0.1.1]: https://github.com/BrianMur92/NEURAL_py_EEG_feature_set/releases/tag/0.1.1
[0.1.2]: https://github.com/BrianMur92/NEURAL_py_EEG_feature_set/releases/tag/0.1.2

