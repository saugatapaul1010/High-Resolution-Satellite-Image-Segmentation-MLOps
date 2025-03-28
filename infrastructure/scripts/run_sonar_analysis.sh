#!/bin/bash
# scripts/run_sonar_analysis.sh

# Make sure SonarQube is running
echo "Checking if SonarQube is running..."
if ! curl -s http://localhost:9000 > /dev/null; then
    echo "SonarQube is not running. Start it with 'docker-compose up -d sonarqube'"
    exit 1
fi

# Generate coverage report
echo "Running tests and generating coverage report..."
pytest --cov=. --cov-report=xml --junitxml=test-results.xml

# Run SonarQube analysis
echo "Running SonarQube analysis..."
sonar-scanner \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=$SONAR_TOKEN

echo "SonarQube analysis complete. View results at http://localhost:9000"