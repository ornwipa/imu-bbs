# article-code
# Postural Balance Assessment in Elderly Using Inertial Measurement Units

## Project Overview
This project aims to evaluate postural balance in older adults using Inertial Measurement Units (IMUs). It involves data collection from multiple body locations during various balance tasks, data processing, feature extraction, and statistical analysis to identify balance indicators and fall risk factors.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Data Collection](#data-collection)
5. [Data Processing](#data-processing)
6. [Analysis](#analysis)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Installation
To set up the project environment:

1. Ensure you have Python 3.x installed.
2. Clone the repository:
git clone https://github.com/yourusername/postural-balance-assessment.git
cd postural-balance-assessment
3. Install required dependencies:
pip install -r requirements.txt
## Project Structure
- `organize-data.py`: Script for organizing raw sensor data.
- `database1.py`: Processes organized data and extracts features.
- `addional-features with bbs.py`: Merges features with Berg Balance Scale scores.
- `Identification of Poor Balance Indicators from Lower Back Data.py`: Analyzes lower back sensor data.
- `Impact of Sensor Placement on Balance Predictors for Static and Dynamic Tasks.py`: Investigates sensor placement effects.

## Usage
Follow these steps to run the analysis:

1. Organize raw data:
python organize-data.py
2. Process data and extract features:
python database1.py
3. Merge features with BBS scores:
python "addional-features with bbs.py"
4. Run analysis scripts:
python "Identification of Poor Balance Indicators from Lower Back Data.py"
python "Impact of Sensor Placement on Balance Predictors for Static and Dynamic Tasks.py"
## Data Collection
- Participants: 14 individuals (6 women, 8 men), average age 59 years.
- Equipment: IMUs (SXT model, NexGen) placed on head, sternum, and lower back.
- Tasks: 14 tasks derived from the Berg Balance Scale.

## Data Processing
- Preprocessing: Linear interpolation, low-pass filtering.
- Feature Extraction: Total path length, jerk, RMS acceleration and angular velocity, area measures, and volume.

## Analysis
- Statistical Methods: Logistic regression, correlation analysis.
- Key Metrics: Area under the curve, RMS angular velocity, total path length, movement volume.

## Results
- Lower back sensor data most indicative of balance status.
- Larger movement volume and total path length associated with better balance.
- Task-specific and sensor location-specific balance indicators identified.

## Contributing
We welcome contributions to this project. Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License


