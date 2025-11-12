# STGCN Project Context and Progress Report

## Project Overview

This project focuses on developing a Spatio-Temporal Graph Convolutional Network (STGCN) for epidemic forecasting, specifically targeting dengue-like disease spread. The work has progressed through two main phases: (1) testing on real data, and (2) generating and testing on artificial data due to limitations in the real dataset.

---

## Phase 1: Real Data Implementation (`code.ipynb`)

### Data Characteristics
- **Source**: Real dengue case data from CSV files
- **Dimensions**: 228 timesteps × 558 localities
- **Data Split**: 
  - Training/Validation: 200 timesteps
  - Hold-out Test: 28 timesteps
- **Challenge**: Limited timesteps (228) insufficient for robust model training

### Model Architecture

#### STGCN Components:
1. **Graph Construction**:
   - Gravity model-based sparse adjacency matrix
   - Uses population and geographic coordinates (latitude/longitude)
   - Formula: `gravity_matrix = (pop_i × pop_j) / (distance²)`
   - Sparsification: Keeps top 3% of edges (97th percentile cutoff)
   - Row-normalized for weighted aggregation
   - Result: 558 nodes, 9,326 edges

2. **GraphConv Layer**:
   - Weighted aggregation of neighbor features
   - Formula: `sum(features × weights) / sum(weights)`
   - Combination type: concatenation of node and aggregated neighbor representations
   - Activation: ReLU

3. **LSTMGC Layer**:
   - Combines GraphConv with LSTM
   - Architecture: GraphConv → LSTM (32 units) → Dense (output sequence length)
   - Processes spatio-temporal patterns simultaneously

#### Hyperparameters:
- Input sequence length: 12 timesteps
- Forecast horizon: 1 timestep
- LSTM units: 32
- GraphConv: in_feat=1, out_feat=4
- Batch size: 32
- Optimizer: Adam (learning rate: 0.001)
- Loss: Mean Absolute Error

### Baseline Models Implemented:
1. **Multi-Variate LSTM**: Standard LSTM without graph structure
2. **Ensemble LSTM**: Individual LSTM models for selected localities (4 localities, then expanded to 70)

### Results on Real Data:
- **STGCN Performance**:
  - MAE: 141.75 cases
  - sMAPE: 106.74%
  - MASE: 2.21 (worse than naive forecast)
  
- **Multi-Variate LSTM Performance**:
  - MAE: 160.11 cases
  - sMAPE: 126.14%
  - MASE: 2.50

- **Ensemble LSTM Performance** (4 localities):
  - MAE: 161.85 cases
  - sMAPE: 121.85%
  - MASE: 2.53

**Conclusion**: Poor performance on real data, likely due to insufficient timesteps (228) for learning complex spatio-temporal patterns.

### Visualizations:
- Comparative plots for 4 selected localities showing actual vs. predicted cases for all three models
- Test set predictions visualized for localities: 10, 150, 300, 500

---

## Phase 2: Artificial Data Implementation (`artificial.ipynb`)

### Motivation
Due to limited real data timesteps, we transitioned to generating artificial epidemic data using an ODE-based epidemic model to:
- Generate longer time series (4000 days simulation)
- Study model behavior under controlled conditions
- Test STGCN architecture with sufficient data

### Artificial Data Generation Model

#### Epidemic Model: Modulated SEIRS-SEI with Cross-Immunity

**Model Type**: 8-compartment ODE system

**Compartments**:
1. **Human (Host) Compartments**:
   - `Sh`: Susceptible humans
   - `Eh`: Exposed humans
   - `Ih`: Infected humans
   - `Rh`: Recovered humans
   - `C`: Cross-immunity factor (slow-moving)

2. **Vector (Mosquito) Compartments**:
   - `Sv`: Susceptible vectors
   - `Ev`: Exposed vectors
   - `Iv`: Infected vectors

**Model Parameters**:
- **Host Parameters**:
  - `SIGMA_H = 1/5.5`: Latency rate (exposed → infected)
  - `GAMMA_H = 1/4.0`: Recovery rate
  - `MU_H = 1/(70*365)`: Natural mortality rate
  - `ALPHA = 1/(365*2)`: Waning immunity rate (2 years)

- **Vector Parameters**:
  - `SIGMA_V = 1/5.5`: Latency rate
  - `MU_V = 1/10.5`: Natural mortality rate
  - `VECTOR_RATIO = 3.0`: Vector-to-human population ratio

- **Transmission Parameters**:
  - `BETA_H = 0.5`: Base vector-to-host transmission rate
  - `BETA_V_AVG = 0.5`: Base host-to-vector average transmission
  - `BETA_V_AMP = 0.15`: Amplitude of seasonal forcing

- **Cross-Immunity Parameters**:
  - `IMMUNITY_PROD = 1e-10`: Rate at which infections build cross-immunity
  - `IMMUNITY_DECAY = 1/(365*3)`: Rate at which cross-immunity wanes (3 years)

**Key Features**:
1. **Seasonal Forcing**: Vector transmission modulated by cosine function
   - `beta_v_seasonal = beta_v_avg × (1 + beta_v_amp × cos(2πt/365.25))`

2. **Cross-Immunity Modulation**: Human transmission reduced by cross-immunity factor
   - `beta_h_modulated = beta_h × (1 - clip(C, 0, 1))`

3. **Spatial Flow**: Physical flow matrix `F_ij` based on gravity model
   - Same gravity model as used in STGCN graph construction
   - Flow rate: `G_CONSTANT = 5e-5`
   - Cutoff: `P_CUTOFF = 95` (top 5% of edges)

**Simulation Details**:
- Total simulation: 4000 days (~11 years)
- Spin-up period: 1500 days (discarded to reach endemic equilibrium)
- Final data: 2500 timesteps
- Seed: Initial infection in highest population node (20 infected humans, 60 infected vectors)
- Output: Daily new exposed human cases per node

**Data Split**:
- Training/Validation: 2000 timesteps (80%)
- Hold-out Test: 500 timesteps (20%)

### STGCN Implementation on Artificial Data

**Same Architecture as Real Data**:
- Graph: Same gravity model (97th percentile cutoff, 558 nodes, 9,326 edges)
- GraphConv + LSTMGC layers
- Same hyperparameters

### Results on Artificial Data:
- **STGCN Performance**:
  - MAE: 19.01 cases
  - sMAPE: 4.85%
  - MASE: 6.29 (worse than naive, but much better than real data)

- **Multi-Variate LSTM Performance**:
  - MAE: 3.22 cases
  - sMAPE: 11.56%
  - MASE: 1.07 (better than naive forecast)

**Observation**: STGCN performs significantly better on artificial data than real data, but interestingly, simple LSTM outperforms STGCN on artificial data.

### Visualizations:
- Full simulation history plot for seed node showing spin-up period
- Artificial data visualization for selected localities
- Comparative model performance plots for multiple localities

---

## Current Issues and Limitations

### 1. Constant Peak Amplitude Problem
**Issue**: In the artificial data, each node exhibits a constant peak amplitude over time. This creates unrealistic regularity that doesn't reflect real-world epidemic dynamics where outbreak magnitudes vary.

**Impact**: 
- Reduces data realism
- May lead to overfitting to periodic patterns
- Doesn't capture stochastic variability in epidemic spread

### 2. Network Structure Bias
**Critical Concern**: The same gravity model network structure is used for:
- **Data Generation**: Physical flow matrix `F_ij` in the ODE system
- **STGCN Graph**: Adjacency matrix for graph convolutions

This creates an "unfair advantage" where the model has perfect knowledge of the underlying network structure that generated the data. In real-world scenarios, the true transmission network is unknown and must be inferred or approximated.

**Implications**:
- Model performance on artificial data may be artificially inflated
- Results may not generalize to scenarios with network mismatch
- Need to test with different network structures for generation vs. prediction

### 3. Epidemic-Guided E-STGCN Design Challenge
**End Goal**: Develop an **Epidemic-Guided E-STGCN** that incorporates ODE model information.

**What "Epidemic-Guided" Means**:
- The standard approach: Fit an ODE model (e.g., SEIR, SEIRS) to **real epidemic data**
- Use the fitted ODE model to generate epidemic predictions/features (e.g., force of infection, transmission rates, compartmental states)
- Pass these ODE-derived features as additional inputs to the STGCN alongside the raw time-series case data
- This allows the neural network to leverage epidemiological knowledge encoded in the ODE model

**Current Dilemma with Artificial Data**:
- The artificial data is **already generated from an ODE model** (the same ODE system we would fit)
- There is **no real data** to fit the ODE model to
- **Question**: What epidemic-guided information should be passed to STGCN when:
  - The data itself is ODE-generated (not real observations)?
  - There's no real data to fit an ODE model to?
  - The ODE parameters are already known (since we generated the data)?

**The Core Problem**:
- Epidemic-guided architecture requires fitting ODE models to **real data** to extract epidemiological insights
- With artificial data, we're missing the "real data" component that makes epidemic-guidance meaningful
- Fitting an ODE model to ODE-generated data would be circular and doesn't provide the intended epidemiological guidance

**Potential Workarounds** (if continuing with artificial data):
- Use the known ODE parameters directly as features (but this doesn't test the fitting process)
- Generate ODE-derived features from the known model (force of infection, transmission rates) as auxiliary inputs
- Add noise/perturbations to ODE parameters and see if STGCN can learn to correct for them
- However, these approaches don't address the fundamental issue: epidemic-guidance is meant to bridge the gap between real data and ODE models, which doesn't exist with artificial data

---

## Future Work and Improvements

Two potential paths forward have been identified:

### Path 1: Continue with Artificial Dengue Data
**Approach**: Work around the identified problems while maintaining focus on dengue-like diseases.

**Planned Workarounds**:
- **Address Constant Peak Amplitude**:
  - Add stochastic noise to break constant amplitude patterns
    - Gaussian noise on case counts
    - Poisson sampling for discrete case counts
    - Environmental noise affecting transmission rates
  - Introduce variable amplitude over time
    - Stochastic transmission rate variations
    - Random external forcing events
    - Time-varying population dynamics
  - Create irregular patterns
    - Random seeding events
    - Stochastic migration events
    - Environmental stochasticity

- **Mitigate Network Structure Bias**:
  - Generate data with one network structure, train STGCN with different network
  - Test robustness to network perturbations
  - Explore network inference methods
  - Compare performance with mismatched networks
  - Use different network construction methods for generation vs. prediction

- **Epidemic-Guided Architecture Development** (Limited with Artificial Data):
  - **Note**: True epidemic-guided architecture requires fitting ODE models to real data, which doesn't exist with artificial data
  - **Workarounds** (if continuing with artificial data):
    - Use known ODE parameters directly as features (doesn't test fitting process)
    - Generate ODE-derived features from the known model (force of infection, transmission rates) as auxiliary inputs
    - Add noise/perturbations to simulate parameter uncertainty
  - **Limitation**: These approaches don't truly test epidemic-guided architecture, which is meant to bridge real data and ODE models
  - **Architecture Modifications** (for future real data):
    - Multi-input STGCN accepting both time-series and ODE-fitted features
    - Physics-informed loss terms incorporating ODE constraints
    - Hybrid model combining neural network and ODE solver

### Path 2: Switch to Alternative Periodic Disease
**Approach**: Transition to a disease with more readily available real-world data, such as influenza (flu) or measles.

**Advantages**:
- **More Real Data Available**: Diseases like flu and measles have extensive historical surveillance data
  - Longer time series (years/decades of data)
  - Better temporal coverage for training robust models
  - Multiple geographic regions with consistent reporting

- **Periodic Nature**: Both flu and measles exhibit strong seasonal/periodic patterns
  - Well-suited for STGCN temporal modeling
  - Clear epidemic cycles to learn from
  - Established epidemiological understanding

- **Data Quality**: 
  - Standardized reporting systems (e.g., CDC flu surveillance, WHO measles data)
  - More consistent data collection across regions
  - Better data completeness and reliability

**Implementation Considerations**:
- Adapt ODE model to flu/measles dynamics (SIR/SEIR models)
- Modify transmission parameters for different disease characteristics
- Utilize real surveillance data for training and validation
- Maintain same STGCN architecture for comparability
- **Enable True Epidemic-Guided Architecture**:
  - Fit ODE models to real flu/measles data to extract epidemiological parameters
  - Use fitted ODE model outputs (force of infection, transmission rates, compartmental states) as features
  - Develop epidemic-guided E-STGCN that combines real data with ODE-derived epidemiological insights
  - This addresses the core limitation: with real data, we can properly implement epidemic-guided architecture

**Potential Data Sources**:
- **Influenza**: 
  - CDC FluView data (weekly, multiple seasons)
  - WHO Global Influenza Surveillance
  - Regional health department datasets
- **Measles**:
  - WHO Measles surveillance data
  - National health surveillance systems
  - Historical outbreak records

### Decision Factors
The choice between paths depends on:
- **Research Goals**: Whether focus should remain on dengue or generalize to periodic diseases
- **Data Availability**: Access to flu/measles datasets with sufficient temporal coverage
- **Problem Complexity**: Whether addressing artificial data limitations is more valuable than working with real data
- **Generalization**: Whether results should be disease-specific or broadly applicable

---

## Technical Implementation Details

### Data Preprocessing
- **Normalization**: MinMaxScaler (fit on training, transform validation/test)
- **Windowing**: Sliding window approach
  - Input: 12 timesteps
  - Output: 1 timestep (forecast horizon)
- **TensorFlow Datasets**: Cached and prefetched for performance

### Model Training
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Absolute Error (MAE)
- **Early Stopping**: Patience of 5-10 epochs
- **Epochs**: 20-100 (depending on model)

### Evaluation Metrics
1. **MAE (Mean Absolute Error)**: Average absolute difference
2. **sMAPE (Symmetric Mean Absolute Percentage Error)**: Percentage error
3. **MASE (Mean Absolute Scaled Error)**: Relative to naive forecast (value < 1 is better)

---

## Summary of Work Completed

### Code Files

#### `code.ipynb` (Real Data):
- ✅ Data loading and preprocessing
- ✅ Gravity model graph construction
- ✅ GraphConv layer implementation
- ✅ LSTMGC layer implementation
- ✅ STGCN model training and evaluation
- ✅ Baseline models (Multi-variate LSTM, Ensemble LSTM)
- ✅ Comparative visualizations
- ✅ Performance metrics calculation

#### `artificial.ipynb` (Artificial Data):
- ✅ ODE-based epidemic model implementation (SEIRS-SEI with cross-immunity)
- ✅ Artificial data generation pipeline
- ✅ Same STGCN architecture implementation
- ✅ Model training on artificial data
- ✅ Performance evaluation
- ✅ Comparative model analysis
- ✅ Data visualization

### Key Achievements
1. Successfully implemented STGCN architecture for epidemic forecasting
2. Developed comprehensive ODE-based epidemic simulator
3. Generated realistic artificial epidemic data
4. Established baseline comparisons with multiple models
5. Created visualization framework for model evaluation

### Current State
- STGCN architecture is functional and tested
- Artificial data generation pipeline is operational
- Identified key limitations and areas for improvement
- Ready for next phase: data enhancement and epidemic-guided architecture development

---

## End Goals

### Primary Objective
Develop an **Epidemic-Guided E-STGCN** that:
1. **Fits ODE models to real epidemic data** to extract epidemiological parameters and insights
2. Incorporates ODE-fitted features (force of infection, transmission rates, compartmental states) into the neural network architecture
3. Achieves robust forecasting performance by combining real time-series data with ODE-derived epidemiological knowledge
4. Generalizes to scenarios with network structure uncertainty
5. Handles irregular, noisy epidemic patterns

**Note**: True epidemic-guided architecture requires real data to fit ODE models to. With artificial data, this fundamental component is missing, making it difficult to properly develop and test the epidemic-guided approach.

### Success Criteria
- Improved forecasting accuracy compared to baseline STGCN
- Robustness to network structure mismatches
- Realistic handling of stochastic epidemic dynamics
- Effective integration of ODE model information
- Validation on real-world epidemic data

---

## Concerns and Open Questions

1. **Network Bias**: How to test and mitigate the unfair advantage from using the same network for generation and prediction?

2. **Epidemic-Guided Architecture with Artificial Data**: Epidemic-guided E-STGCN requires fitting ODE models to real data, but with artificial data there is no real data to fit to. How can we meaningfully develop epidemic-guided architecture without real data? This is a fundamental limitation that may require switching to diseases with real data (flu/measles).

3. **Data Realism**: How to balance realistic irregularity with controlled experimental conditions?

4. **Generalization**: Will improvements on artificial data translate to real-world performance?

5. **Architecture Design**: What is the optimal way to combine neural networks with ODE-fitted models in the epidemic-guided framework? This requires real data to fit ODE models to, which is currently unavailable for dengue.

---

## Next Steps

1. **Immediate**: Add noise and variability to artificial data generation
2. **Short-term**: Test network structure mismatch scenarios
3. **Medium-term**: Design and implement epidemic-guided architecture
4. **Long-term**: Validate on real data and refine for deployment

---

*Last Updated: Based on work completed in `code.ipynb` and `artificial.ipynb`*

