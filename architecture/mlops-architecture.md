%%{init: {
  'theme': 'neutral',
  'themeVariables': {
    'primaryColor': '#5D8AA8',
    'primaryTextColor': '#fff',
    'primaryBorderColor': '#5D8AA8',
    'lineColor': '#5D8AA8',
    'secondaryColor': '#006400',
    'tertiaryColor': '#fff'
  },
  'flowchart': {
    'curve': 'basis',
    'htmlLabels': true
  }
}}%%

flowchart TB
    %% Main System Components
    classDef dataLayer fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,color:#0D47A1
    classDef trainingLayer fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px,color:#1B5E20
    classDef registryLayer fill:#FFF3E0,stroke:#FF9800,stroke-width:2px,color:#E65100
    classDef inferenceLayer fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px,color:#4A148C
    classDef monitoringLayer fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px,color:#006064
    classDef cicdLayer fill:#FCE4EC,stroke:#E91E63,stroke-width:2px,color:#880E4F
    classDef containerLayer fill:#FFFDE7,stroke:#FFC107,stroke-width:2px,color:#FF6F00
    
    %% Title
    TITLE["Production-Grade MLOps System<br>Satellite Image Segmentation"]
    style TITLE fill:none,stroke:none,color:#212529,font-size:20px
    
    %% Data Layer
    subgraph DataLayer["Data Management Layer"]
        direction TB
        RawData["Raw Satellite Images<br><small>DVC Tracked</small>"]
        Annotations["Image Annotations<br><small>DVC Tracked (JSON)</small>"]
        ProcessedData["Processed Data<br><small>Patches & Binary Masks</small>"]
        DataRegistry["Data Registry<br><small>PostgreSQL</small>"]
        DataValidation["Data Validation<br><small>Great Expectations</small>"]
        DataVersioning["Version Control<br><small>DVC</small>"]
        
        RawData --> DataVersioning
        Annotations --> DataVersioning
        DataVersioning --> ProcessedData
        ProcessedData --> DataRegistry
        DataRegistry --> DataValidation
    end
    class DataLayer dataLayer
    
    %% Training Pipeline
    subgraph TrainingPipeline["Training Pipeline"]
        direction TB
        DataPrep["Data Preparation<br><small>OpenCV, NumPy</small>"]
        FeatureEng["Feature Engineering<br><small>Patch Generation</small>"]
        DataSplit["Train/Val Split<br><small>Stratified Sampling</small>"]
        Training["Model Training<br><small>TensorFlow, LinkNet</small>"]
        HyperOpt["Hyperparameter Tuning<br><small>Bayesian Optimization</small>"]
        Evaluation["Model Evaluation<br><small>IoU, Dice Coefficient</small>"]
        ExpTracking["Experiment Tracking<br><small>MLflow</small>"]
        
        DataPrep --> FeatureEng
        FeatureEng --> DataSplit
        DataSplit --> Training
        HyperOpt --> Training
        Training --> Evaluation
        Training --> ExpTracking
        Evaluation --> ExpTracking
    end
    class TrainingPipeline trainingLayer
    
    %% Automated Retraining
    subgraph AutoRetraining["Automated Retraining System"]
        direction TB
        RetrainingTrigger["Retraining Triggers<br><small>Scheduled/Drift-Based</small>"]
        DataSelection["Incremental Data<br>Selection"]
        AutoTraining["Automated Training<br><small>Parameterized Pipeline</small>"]
        ModelComparison["Model Comparison<br><small>A/B Testing</small>"]
        DeployDecision{"Deploy Decision<br><small>Performance Metrics</small>"}
        
        RetrainingTrigger --> DataSelection
        DataSelection --> AutoTraining
        AutoTraining --> ModelComparison
        ModelComparison --> DeployDecision
    end
    class AutoRetraining trainingLayer
    
    %% Model Registry
    subgraph ModelRegistry["Model Registry"]
        direction TB
        ModelVersions["Model Versions<br><small>MLflow Registry</small>"]
        ModelMetadata["Model Metadata<br><small>Performance Metrics</small>"]
        ArtifactStorage["Artifact Storage<br><small>Local File System</small>"]
        
        ModelVersions --- ModelMetadata
        ModelVersions --- ArtifactStorage
    end
    class ModelRegistry registryLayer
    
    %% Inference Pipeline
    subgraph InferencePipeline["Inference Pipeline"]
        direction TB
        ModelLoading["Model Loading<br><small>TensorFlow Serving</small>"]
        PredictionAPI["Prediction API<br><small>FastAPI</small>"]
        Postprocessing["Post-processing<br><small>Contour Detection</small>"]
        BatchPrediction["Batch Prediction<br><small>Parallel Processing</small>"]
        GeoConversion["Geographic Conversion<br><small>Coordinate Mapping</small>"]
        ResultStorage["Result Storage<br><small>GeoJSON/CSV</small>"]
        
        ModelLoading --> PredictionAPI
        PredictionAPI --> Postprocessing
        ModelLoading --> BatchPrediction
        BatchPrediction --> Postprocessing
        Postprocessing --> GeoConversion
        GeoConversion --> ResultStorage
    end
    class InferencePipeline inferenceLayer
    
    %% Monitoring System
    subgraph MonitoringSystem["Monitoring System"]
        direction TB
        PerformanceMonitor["Model Performance<br><small>Metrics Dashboard</small>"]
        DataDrift["Data Drift Detection<br><small>Statistical Tests</small>"]
        ConceptDrift["Concept Drift<br><small>Performance Degradation</small>"]
        Alerting["Alerting System<br><small>Prometheus/Grafana</small>"]
        Logging["Centralized Logging<br><small>ELK Stack</small>"]
        
        PerformanceMonitor --> Alerting
        DataDrift --> Alerting
        ConceptDrift --> Alerting
        Alerting --> Logging
    end
    class MonitoringSystem monitoringLayer
    
    %% CI/CD Pipeline
    subgraph CICDPipeline["CI/CD Pipeline"]
        direction TB
        CIPipeline["CI Pipeline<br><small>GitHub Actions</small>"]
        CDPipeline["CD Pipeline<br><small>GitHub Actions</small>"]
        CodeQuality["Code Quality<br><small>SonarQube</small>"]
        UnitTests["Unit Tests<br><small>PyTest</small>"]
        IntegrationTests["Integration Tests<br><small>PyTest</small>"]
        ArtifactPublication["Artifact Publication<br><small>Docker Registry</small>"]
        
        CIPipeline --> CodeQuality
        CIPipeline --> UnitTests
        CIPipeline --> IntegrationTests
        CDPipeline --> ArtifactPublication
    end
    class CICDPipeline cicdLayer
    
    %% Containerization
    subgraph Containerization["Containerization"]
        direction TB
        TrainingContainer["Training Container<br><small>Docker</small>"]
        InferenceContainer["Inference Container<br><small>Docker</small>"]
        MLflowContainer["MLflow Container<br><small>Docker</small>"]
        MonitoringContainer["Monitoring Container<br><small>Docker</small>"]
        APIContainer["API Container<br><small>Docker</small>"]
        DBContainer["Database Container<br><small>Docker</small>"]
        
        TrainingContainer --- InferenceContainer
        MLflowContainer --- MonitoringContainer
        APIContainer --- DBContainer
    end
    class Containerization containerLayer
    
    %% Main Connections with Animation
    ProcessedData ==>|"DVC Pipeline"| DataPrep
    Annotations ==>|"DVC Pipeline"| DataPrep
    
    ExpTracking ==>|"Register Model"| ModelVersions
    
    Evaluation ==>|"Performance<br>Metrics"| PerformanceMonitor
    Evaluation ==>|"Trigger<br>Retraining"| RetrainingTrigger
    
    DeployDecision ==>|"Yes"| ModelVersions
    DeployDecision ==>|"No"| DataSelection
    
    ModelVersions ==>|"Load<br>Production Model"| ModelLoading
    
    ResultStorage ==>|"Feedback<br>Loop"| DataDrift
    
    PredictionAPI ==>|"Real-time<br>Inference"| PerformanceMonitor
    BatchPrediction ==>|"Batch<br>Results"| DataDrift
    
    DataDrift ==>|"Detect<br>Drift"| RetrainingTrigger
    ConceptDrift ==>|"Detect<br>Degradation"| RetrainingTrigger
    
    CIPipeline ==>|"Validate"| CDPipeline
    CDPipeline ==>|"Deploy"| InferenceContainer
    CDPipeline ==>|"Deploy"| TrainingContainer
    CDPipeline ==>|"Deploy"| MLflowContainer
    CDPipeline ==>|"Deploy"| MonitoringContainer
    CDPipeline ==>|"Deploy"| APIContainer
    
    %% Add legend
    LEGEND["Legend:
      🔵 Data Management
      🟢 Training & Retraining
      🟠 Model Registry
      🟣 Inference Pipeline
      🔷 Monitoring
      🔴 CI/CD
      🟡 Containerization"]
    style LEGEND fill:white,stroke:#ccc,stroke-width:1px,color:#333,font-size:12px